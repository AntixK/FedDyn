import os
import shutil
import torch
from pathlib import Path
import torch.nn as nn
from torch.optim import SGD, Adadelta
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt

plt.style.use('seaborn')


class Server:
    def __init__(self,
                 model: nn.Module,
                 weights_dir: Path,
                 data_dir: Path,
                 alpha: float,
                 num_clients: int):
        self.model = model
        self.weights_dir = weights_dir
        self.data_dir = data_dir
        self.client_state_dicts = []
        self.alpha = alpha
        self.num_clients = num_clients
        self.h = self.model.state_dict().copy()

        self.criterion = nn.NLLLoss()
        data = torch.load(self.data_dir / "test_data.pth")
        self.test_loader = DataLoader(data,
                                 batch_size=1000,
                                 shuffle=True, num_workers=4)

        # Metrics
        self.test_loss_log = []
        self.accuracy_log = []

    def receiveMessage(self, participant_ids: torch.Tensor) -> None:
        self.client_state_dicts = []
        for p_ids in participant_ids:
            self.client_state_dicts.append(
                torch.load(self.weights_dir / f'client_{p_ids}_model.pth')["model_state_dict"])
        print("Server: Received Message.")

    def sendMessage(self) -> None:
        torch.save({"model_state_dict": self.model.state_dict()},
                   self.weights_dir / f"server_model.pth")
        print("Server: Sent Message.")

    def updateModel(self):
        print(f"Server: Updating model...")
        num_participants = len(self.client_state_dicts)
        sum_theta = self.client_state_dicts[0]
        for client_theta in self.client_state_dicts[1:]:
            for key in client_theta.keys():
                sum_theta[key] += client_theta[key]

        delta_theta = {}
        for key in self.model.state_dict().keys():
            delta_theta[key] = sum_theta[key] - self.model.state_dict()[key]

        for key in self.h.keys():
            self.h[key] -= self.alpha * (1./self.num_clients) * delta_theta[key]

        for key in self.model.state_dict().keys():
            self.model.state_dict()[key] = (1./num_participants) * sum_theta[key] - (1./self.alpha) *  self.h[key]
        print("Server: Updated model.")

    def evaluate(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc = "Evaluating")
            for data, labels in pbar:
                data, labels = data.cuda(), labels.cuda()
                y = self.model(data)
                test_loss += self.criterion(y, labels).item()  # sum up batch loss
                _, predicted = torch.max(y.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        test_loss /= len(self.test_loader)
        accuracy = 100. * correct / total
        self.test_loss_log.append(test_loss)
        self.accuracy_log.append(accuracy)

        print(f"\nTest dataset: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({accuracy:.2f}%)\n")

    def _plot_results(self):
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(self.test_loss_log, lw=2, color='k', label='Test Loss')
        ax.set_ylabel('Test Loss', fontsize=13, color='k')

        ax2 = ax.twinx()
        ax2.plot(self.accuracy_log, lw=2, color='tab:blue', label="Accuracy")
        ax2.set_ylabel('Accuracy %', fontsize=13, color='tab:blue')

        ax.set_xticks(range(len(self.test_loss_log)))
        # ax.set_xlim([-1, self.num_classes])
        ax.set_title("Results", fontsize=15)
        ax.set_xlabel('Rounds', fontsize=13)
        plt.tight_layout()
        plt.savefig("results.png", dpi=200)
        # plt.show()

class ClientNode:
    def __init__(self,
                 model: nn.Module,
                 weights_dir: Path,
                 learning_rate: float,
                 batch_size: int,
                 data_dir: Path,
                 alpha: float,
                 id: str):
        self.model = model
        self.weights_dir = weights_dir
        self.id = id
        self.alpha = alpha

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.data_dir = data_dir
        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.NLLLoss()
        self.optim = Adadelta(self.model.parameters(), lr=self.learning_rate)

        self.server_state_dict = None

    def gatherData(self) -> None:
        data = torch.load(self.data_dir / self.id / "data.pth")
        self.train_loader = DataLoader(data,
                                       batch_size=self.batch_size,
                                       shuffle=True, num_workers=4)
        print(f"Client {self.id}: Gathered data.")

    def receiveMessage(self):
        self.model.load_state_dict(torch.load(self.weights_dir / 'server_model.pth')["model_state_dict"])
        self.server_state_dict = torch.load(self.weights_dir / 'server_model.pth')["model_state_dict"]
        print(f"Client {self.id}: Received message from server.")

    def sendMessage(self):
        torch.save({"model_state_dict": self.model.state_dict()},
                   self.weights_dir / f"client_{self.id}_model.pth")
        print(f"Client {self.id}: Sent message to server.")

    def trainModel(self,
                   num_epochs:int):
        # print(f"Client {self.id}: Training model...")

        self.model.train()
        pbar = tqdm(range(num_epochs), desc=f"Client {self.id} Training")
        for epoch in pbar:
            epoch_loss = 0.0
            for data, labels in self.train_loader:
                # print(labels)
                self.optim.zero_grad()

                y = self.model(data.cuda())
                # print(y.shape, labels.shape)
                loss = self.criterion(y, labels.cuda())

                # Dynamic regularization
                reg = 0.0
                for name, param in self.model.named_parameters():
                    reg += F.mse_loss(param, self.server_state_dict[name], reduction='sum')

                loss += self.alpha/2.0 * reg

                epoch_loss += loss.item()
                loss.backward()
                self.optim.step()
            pbar.set_postfix({"Loss":epoch_loss/len(self.train_loader)})
        self.model.eval()
        # print(f"Client {self.id}: Training done.")

class FedDyn:
    def __init__(self,
                 model: nn.Module,
                 num_clients: int,
                 data_dir: Path,
                 alpha:float,
                 batch_size: int,
                 learning_rate: float,
                 seed: int = 4864,
                 task: str = "classification"):

        torch.manual_seed(seed)
        assert num_clients > 0, "num_clients must be positive."
        self.weights_dir = Path("tmp")
        self.weights_dir.mkdir(exist_ok=True)
        self.num_clients = num_clients
        if os.path.exists(self.weights_dir):
            shutil.rmtree(self.weights_dir)
        self.weights_dir.mkdir()

        # Initialize Server and Clients
        self.server = Server(model,
                             self.weights_dir,
                             alpha = alpha,
                             num_clients=num_clients,
                             data_dir=data_dir)
        self.clients = []
        for i in range(num_clients):
            self.clients.append(ClientNode(model,
                                           self.weights_dir,
                                           batch_size=batch_size,
                                           learning_rate=learning_rate,
                                           data_dir=data_dir,
                                           id=str(i),
                                           alpha = alpha))

    def run(self,
            num_epochs: int,
            num_rounds: int,
            participation_level: float):

        assert num_rounds > 0, "num_rounds must be positive."
        if participation_level <= 0 or participation_level > 1.0:
            raise ValueError("participation_level must be in the range (0, 1].")

        num_active_clients = int(participation_level*self.num_clients)
        for t in range(num_rounds):
            print("="*30)
            print(" "*10 + f"Round {t+1}")

            # Get participants
            participant_ids = torch.randperm(self.num_clients)[:num_active_clients]

            # Send weights to all participants
            self.server.sendMessage()
            for p_id in participant_ids:
                self.clients[p_id].gatherData()

            # Train the participant models
            for p_id in participant_ids:
                self.clients[p_id].receiveMessage()
                self.clients[p_id].trainModel(num_epochs)
                self.clients[p_id].sendMessage()

            # Receive participant models
            self.server.receiveMessage(participant_ids)

            # Update the Server
            self.server.updateModel()

            # Test the server model
            self.server.evaluate()

            # clean up client tmp folder
            print("="*30)

        self.server._plot_results()

if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from models import MLP, Net
    from prepare_data import dataPrep
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms

    # d = dataPrep("MNIST", root_dir =Path("../Data/"))
    # d.make(0, 10, dir_alpha=0.7, lognorm_std=0.0, show_plots=False)

    f = FedDyn(model = MLP().cuda(),
               num_clients = 10,
               data_dir= Path("../Data/client_data"),
               batch_size = 128,
               learning_rate = 0.003,
               alpha=0.4)

    f.run(num_epochs=100,
          num_rounds=10,
          participation_level=0.5)


