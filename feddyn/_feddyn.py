import os
import shutil
import torch
from pathlib import Path
import torch.nn as nn
from torch.optim import SGD
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader


class Server:
    def __init__(self,
                 model: nn.Module,
                 weights_dir: Path,
                 alpha: float,
                 num_clients: int):
        self.model = model
        self.weights_dir = weights_dir
        self.client_state_dicts = []
        self.alpha = alpha
        self.num_clients = num_clients
        self.h = self.model.state_dict().copy()

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


class ClientNode:
    def __init__(self,
                 model: nn.Module,
                 weights_dir: Path,
                 learning_rate: float,
                 batch_size: int,
                 data_dir: Path,
                 id: str):
        self.model = model
        self.weights_dir = weights_dir
        self.id = id

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.data_dir = data_dir
        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = nn.NLLLoss()
        self.optim = SGD(self.model.parameters(), lr=self.learning_rate)

        self.server_weights = None

    def gatherData(self) -> None:
        data = torch.load(self.data_dir / self.id / "data.pth")
        self.train_loader = DataLoader(data,
                                       batch_size=self.batch_size,
                                       shuffle=True, num_workers=4)
        print(f"Client {self.id}: Gathered data.")

    def receiveMessage(self):
        self.model.load_state_dict(torch.load(self.weights_dir / 'server_model.pth')["model_state_dict"])
        print(f"Client {self.id}: Received message from server.")

    def sendMessage(self):
        torch.save({"model_state_dict": self.model.state_dict()},
                   self.weights_dir / f"client_{self.id}_model.pth")
        print(f"Client {self.id}: Sent message to server.")

    def trainModel(self,
                   num_epochs:int):
        # print(f"Client {self.id}: Training model...")

        self.model.train()
        pbar = tqdm(range(num_epochs), desc=f"Client {self.id} Training:")
        epoch_loss = 0.0
        for epoch in pbar:
            for data, labels in self.train_loader:
                # print(labels)
                self.optim.zero_grad()

                y = self.model(data.cuda())
                # print(y.shape, labels.shape)
                loss = self.criterion(y, labels.cuda())
                # loss = F.nll_loss(y, labels)
                # Dynamic regularization
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
                             alpha = 0.4,
                             num_clients=num_clients)
        self.clients = []
        for i in range(num_clients):
            self.clients.append(ClientNode(model,
                                           self.weights_dir,
                                           batch_size=batch_size,
                                           learning_rate=learning_rate,
                                           data_dir=data_dir,
                                           id=str(i)))

    def run(self,
            num_epochs: int,
            num_rounds: int,
            participation_level: float,
            alpha: float):

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
            print("="*30)

            # Test the server model

            # clean up client tmp folder

if __name__ == "__main__":
    from models import MLP, Net
    from prepare_data import dataPrep
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms

    # d = dataPrep("MNIST", root_dir =Path("../Data/"))
    # d.make(0, 1, dir_alpha=0.7, lognorm_std=0.0, show_plots=False)
    #
    # f = FedDyn(model = MLP().cuda(),
    #        num_clients = 10,
    #        data_dir= Path("../Data/client_data"),
    #        batch_size = 48,
    #        learning_rate = 0.0001)
    #
    # f.run(100,2,participation_level=0.3, alpha=0.4)

    # data = torch.load("../Data/client_data/0/data.pth")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        # transforms.Lambda(lambda x: torch.flatten(x,start_dim=1).squeeze())
    ])

    data = datasets.MNIST(Path("../Data/"),
                                     train=True,
                                     download=True,
                                     transform=transform)
    criterion = nn.NLLLoss()
    train_loader = DataLoader(data,
                                   batch_size=64,
                                   shuffle=True, num_workers=4, pin_memory=True)

    model = Net().cuda()
    optim = SGD(model.parameters(), lr=0.0001)
    model.train()
    pbar = tqdm(range(50), desc=f"Training:")
    epoch_loss = 0.0
    for epoch in pbar:
        for data, labels in train_loader:
            # print(labels)
            optim.zero_grad()

            y = model(data.cuda())
            # print(y.shape, labels.shape)
            loss = criterion(y, labels.cuda())
            # loss = F.nll_loss(y, labels)
            # Dynamic regularization
            epoch_loss += loss.detach().item()
            loss.backward()
            optim.step()
        pbar.set_postfix({"Loss": epoch_loss/len(train_loader)})

