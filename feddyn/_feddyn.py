import torch
from pathlib import Path
import torch.nn as nn
import logging


class Server:
    def __init__(self,
                 model: nn.Module,
                 weights_dir: str):
        self.model = model
        self.weights_dir = weights_dir

    def receiveMessage(self, participant_ids):
        pass

    def sendMessage(self):
        pass

    def updateModel(self):
        pass


class ClientNode:
    def __init__(self,
                 model: nn.Module,
                 weights_dir: str,
                 learning_rate: float,
                 batch_size: int,
                 data_dir: str,
                 id: str):
        self.model = model
        self.weights_dir = weights_dir
        self.id = id

    def gatherData(self):
        pass

    def receiveMessage(self):
        pass

    def sendMessage(self):
        pass

    def trainModel(self,
                   num_epochs:int):

        for epoch in range(num_epochs):

            self.optim.zero_grad()

            loss.backward()
            self.optim.step()


class FedDyn:
    def __init__(self,
                 model: nn.Module,
                 num_clients: int,
                 data_dir: str,
                 batch_size: int,
                 learning_rate: float,
                 task: str = "classification"):

        assert num_clients > 0, "num_clients must be positive."
        self.weights_dir = Path("tmp")
        self.weights_dir.mkdir(exist_ok=True)
        self.num_clients = num_clients
        # Initialize Server and Clients
        self.server = Server(model, self.weights_dir)
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

            # Get participants
            participant_ids = torch.randperm(self.num_clients)[:num_active_clients]

            # Send weights to all participants
            self.server.sendMessage()

            # Train the participant models
            for p_id in participant_ids:
                self.clients[p_id].receiveMessage()
                self.clients[p_id].gatherData()
                self.clients[p_id].trainModel(num_epochs)
                self.clients[p_id].sendMessage()

            # Receive participant models
            self.server.receiveMessage(participant_ids)

            # Update the Server
            self.server.updateModel()

            # clean up client tmp folder




