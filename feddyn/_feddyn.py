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
from joblib import Parallel, delayed
from joblib.externals.loky.backend.context import get_context

plt.style.use('seaborn')

__all__ = ['FedDyn']

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

    def _plot_results(self, exp_name: str):
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(self.test_loss_log, lw=2, color='k', label='Test Loss')
        ax.set_ylabel('Test Loss', fontsize=11, color='k')

        ax2 = ax.twinx()
        ax2.plot(self.accuracy_log, lw=2, color='tab:blue', label="Accuracy")
        ax2.set_ylabel('Accuracy %', fontsize=11, color='tab:blue')

        # ax.set_xticks(range(len(self.test_loss_log)))
        # ax.set_xlim([-1, self.num_classes])
        ax.set_title(f"{exp_name} Results", fontsize=13)
        ax.set_xlabel('Communication Rounds', fontsize=11)
        plt.tight_layout()
        plt.savefig(f"{exp_name}_results.png", dpi=200)
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
        self.optim = SGD(self.model.parameters(),
                         lr=self.learning_rate,
                         weight_decay=1e-4)

        self.server_state_dict = None

        self.prev_grads = None
        for param in self.model.parameters():
            if not isinstance(self.prev_grads, torch.Tensor):
                self.prev_grads = torch.zeros_like(param.view(-1))
            else:
                self.prev_grads = torch.cat((self.prev_grads, torch.zeros_like(param.view(-1))), dim=0)


    def gatherData(self) -> None:
        data = torch.load(self.data_dir / self.id / "data.pth")
        self.train_loader = DataLoader(data,
                                       batch_size=self.batch_size,
                                       shuffle=True, num_workers=4,
                                       multiprocessing_context=get_context('loky'))
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
                epoch_loss = {}
                loss = self.criterion(y, labels.cuda())
                epoch_loss['Task Loss'] = loss.item()
                #=== Dynamic regularization === #
                # Linear penalty
                lin_penalty = 0.0
                curr_params = None
                for name, param in self.model.named_parameters():
                    if not isinstance(curr_params, torch.Tensor):
                        curr_params = param.view(-1)
                    else:
                        curr_params = torch.cat((curr_params, param.view(-1)), dim=0)

                lin_penalty = torch.sum(curr_params * self.prev_grads)
                loss -= lin_penalty
                epoch_loss['Lin Penalty'] = lin_penalty.item()

                # Quadratic Penalty
                quad_penalty = 0.0
                for name, param in self.model.named_parameters():
                    quad_penalty += F.mse_loss(param, self.server_state_dict[name], reduction='sum')

                loss += self.alpha/2.0 * quad_penalty
                epoch_loss['Quad Penalty'] = quad_penalty.item()
                loss.backward()

                # Update the previous gradients
                self.prev_grads = None
                for param in self.model.parameters():
                    if not isinstance(self.prev_grads, torch.Tensor):
                        self.prev_grads = param.grad.view(-1).clone()
                    else:
                        self.prev_grads = torch.cat((self.prev_grads, param.grad.view(-1).clone()), dim=0)

                self.optim.step()
            pbar.set_postfix(epoch_loss) #{"Loss":epoch_loss/len(self.train_loader)})
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

    def _client_run(self, client_id: int, num_epochs: int):
        self.clients[client_id].receiveMessage()
        self.clients[client_id].trainModel(num_epochs)
        self.clients[client_id].sendMessage()

    def run(self,
            num_epochs: int,
            num_rounds: int,
            participation_level: float,
            exp_name: str):

        assert num_rounds > 0, "num_rounds must be positive."
        if participation_level <= 0 or participation_level > 1.0:
            raise ValueError("participation_level must be in the range (0, 1].")

        num_active_clients = int(participation_level*self.num_clients)

        for p_id in range(self.num_clients):
            self.clients[p_id].gatherData()

        for t in range(num_rounds):
            print("="*30)
            print(" "*10 + f"Round {t+1}")

            # Get participants
            participant_ids = torch.randperm(self.num_clients)[:num_active_clients]

            # Send weights to all participants
            self.server.sendMessage()

            # Train the participant models
            for p_id in participant_ids:
                self._client_run(p_id, num_epochs=num_epochs)
            # Parallel(n_jobs=self.num_clients)(delayed(self._client_run)(p_id, num_epochs)
            #                                   for p_id in participant_ids)

            # Receive participant models
            self.server.receiveMessage(participant_ids)

            # Update the Server
            self.server.updateModel()

            # Test the server model
            self.server.evaluate()

            # clean up client tmp folder
            print("="*30)

        self.server._plot_results(exp_name)

if __name__ == "__main__":
    import sys
    sys.path.append("..")
    from models import MLP, Net
    from prepare_data import dataPrep
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    #
    # d = dataPrep("MNIST", root_dir =Path("../Data/"))
    # d.make(1, 10, dir_alpha=0.7, lognorm_std=0.3, show_plots=False)

    f = FedDyn(model = MLP().cuda(),
               num_clients = 10,
               data_dir= Path("../Data/client_data"),
               batch_size = 128,
               learning_rate = 0.1,
               alpha=0.01)

    f.run(num_epochs=50,
          num_rounds=100,
          participation_level=0.5,
          exp_name=r"MNIST 50% Non-IID balanced")

