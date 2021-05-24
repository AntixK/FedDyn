import os
import torch
import numpy as np
import shutil
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.distributions import Dirichlet, Categorical, LogNormal

from typing import List
plt.style.use('seaborn')


class dataPrep:
    def __init__(self,
                 dataset_name: str,
                 root_dir: Path) -> None:

        self.root_dir = root_dir
        self.dataset_name = dataset_name

        if self.dataset_name == "MNIST":
            ## Reference: https://stackoverflow.com/a/66816284
            new_mnist_mirror = 'https://ossci-datasets.s3.amazonaws.com/mnist'
            datasets.MNIST.resources = [
                ('/'.join([new_mnist_mirror, url.split('/')[-1]]), md5)
                for url, md5 in datasets.MNIST.resources
            ]

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

            self.train_data = datasets.MNIST(root_dir / "raw/",
                                             train=True,
                                             download=True,
                                             transform=transform)
            self.test_data = datasets.MNIST(root_dir / "raw/",
                                            train=False,
                                            download=True,
                                            transform=transform)
            self.num_train_data = len(self.train_data)
            self.num_classes = 10
        elif self.dataset_name == "EMNIST":
            self.train_data = datasets.EMNIST(root_dir / "raw/", split="letters", train=True, download=True)
            self.test_data = datasets.EMNIST(root_dir / "raw/", split="letters", train=False, download=True)
        elif self.dataset_name == "CIFAR10":
            transform = transforms.Compose(
                            [transforms.ToTensor(),
                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

            self.train_data = datasets.CIFAR10(root_dir / "raw/",
                                               train=True,
                                               download=True,
                                               transform=transform)
            self.test_data = datasets.CIFAR10(root_dir / "raw/",
                                              train=False,
                                              download=True,
                                              transform=transform)

            self.num_train_data = len(self.train_data)
            self.num_classes = 10
        else:
            raise ValueError("Unknown dataset_name")

    def make(self,
             mode: int,
             num_clients: int,
             show_plots: bool = False,
             **kwargs) -> None:

        if os.path.exists(self.root_dir / "client_data"):
            shutil.rmtree(self.root_dir / "client_data")
        client_data_path = Path(self.root_dir / "client_data")
        client_data_path.mkdir()

        if mode == 0:       # IID
            # Shuffle data
            data_ids = torch.randperm(self.num_train_data, dtype=torch.int32)
            num_data_per_client = self.num_train_data// num_clients

            if not isinstance(self.train_data.targets, torch.Tensor):
                self.train_data.targets = torch.tensor(self.train_data.targets)

            pbar = tqdm(range(num_clients), desc = f"{self.dataset_name} Non-IID Unbalanced: ")
            for i in pbar:
                client_path = Path(client_data_path / str(i))
                client_path.mkdir()

                # TODO: Make this parallel for large number of clients & large datasets (Maybe not required)
                train_data = [self.train_data[j]
                              for j in data_ids[i*num_data_per_client: (i+1)*num_data_per_client]]

                pbar.set_postfix({'# data / Client': num_data_per_client})

                if show_plots:
                    self._plot(train_data, title=f"Client {i+1} Data Distribution")

                # Split data equally and send to the client
                torch.save(train_data,
                           client_data_path / str(i) / "data.pth")
        elif mode == 1:     # Non IID Balanced
            num_data_per_client = self.num_train_data//num_clients
            classs_sampler = Dirichlet(torch.empty(self.num_classes).fill_(kwargs.get('dir_alpha')))
            # print(torch.empty(self.num_classes).fill_(2.0))
            if not isinstance(self.train_data.targets, torch.Tensor):
                self.train_data.targets = torch.tensor(self.train_data.targets)

            assigned_ids = []
            pbar = tqdm(range(num_clients), desc = f"{self.dataset_name} Non-IID Balanced: ")
            for i in pbar:

                client_path = Path(client_data_path / str(i))
                client_path.mkdir()
                # Compute class prior probabilities for each client
                p_ij = classs_sampler.sample()  # Share of jth class for ith client (always sums to 1)
                # print(p_ij)
                weights = torch.zeros(self.num_train_data)
                # print(torch.nonzero(self.train_data.targets == 9))
                for c_id in range(self.num_classes):
                    weights[self.train_data.targets == c_id] = p_ij[c_id]
                weights[assigned_ids] = 0.0 # So that previously assigned data are not sampled again

                # Sample each data point uniformly without replacement based on
                # the sampling probability assigned based on its class
                data_ids = torch.multinomial(weights, num_data_per_client, replacement=False)

                train_data = [self.train_data[j] for j in data_ids]
                # print(f"Client {i} has {len(train_data)} data points.")
                pbar.set_postfix({'# data / Client': len(train_data)})

                assigned_ids += data_ids.tolist()

                torch.save(train_data,
                           client_data_path / str(i) / "data.pth")

                if show_plots:
                    self._plot(train_data, title=f"Client {i+1} Data Distribution")
        elif mode == 2:     # Non IID Unbalanced
            num_data_per_client = self.num_train_data // num_clients
            num_data_per_class = self.num_train_data / (self.num_classes * num_clients)
            classs_sampler = Dirichlet(torch.empty(self.num_classes).fill_(kwargs.get('dir_alpha')))

            assigned_ids = []
            pbar = tqdm(range(num_clients), desc = f"{self.dataset_name} Non-IID Unbalanced: ")

            if not isinstance(self.train_data.targets, torch.Tensor):
                self.train_data.targets = torch.tensor(self.train_data.targets)

            for i in pbar:
                train_data = []
                client_path = Path(client_data_path / str(i))
                client_path.mkdir()
                # Compute class prior probabilities for each client
                p_ij = classs_sampler.sample()  # Share of jth class for ith client (always sums to 1)
                c_sampler = Categorical(p_ij)
                data_sampler = LogNormal(torch.tensor(num_data_per_class).log(),
                                         kwargs.get('lognorm_std'))

                while(True):
                    num_data_left = num_data_per_client - len(train_data)
                    c = c_sampler.sample()
                    num_data_c = int(data_sampler.sample())
                    # print(c, num_data_c, len(train_data))
                    data_ids = torch.nonzero(self.train_data.targets == c.item()).flatten()
                    # data_ids = [x for x in data_ids if x not in assigned_ids] # Remove duplicated ids
                    # print(data_ids.shape)
                    num_data_c = min(num_data_c, data_ids.shape[0])
                    if num_data_c >= num_data_left :
                        train_data += [self.train_data[j] for j in data_ids[:num_data_left]]
                        break
                    else:
                        train_data += [self.train_data[j] for j in data_ids[:num_data_c]]
                        assigned_ids += data_ids[:num_data_c].tolist()

                pbar.set_postfix({'# data / Client': len(train_data)})
                torch.save(train_data,
                           client_data_path / str(i) / "data.pth")
                if show_plots:
                    self._plot(train_data, title=f"Client {i+1} Data Distribution")

        else:
            raise ValueError("Unknown mode. Mode must be {0,1}")

    def _plot(self, data: List, title: str = None) -> None:
        labels = [int(d[1]) for d in data]
        # print(labels)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.hist(labels, bins=np.arange(self.num_classes + 1) - 0.5)
        ax.set_xticks(range(self.num_classes))
        ax.set_xlim([-1, self.num_classes])
        ax.set_title(title, fontsize=15)
        ax.set_xlabel('Class ID', fontsize=13)
        ax.set_ylabel('# samples', fontsize=13)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    d = dataPrep("MNIST", root_dir =Path("Data/"))
    d.make(1, 10, dir_alpha=0.7, lognorm_std=0.0, show_plots=True)

