import os
import torch
import numpy as np
import torchvision.datasets as datasets
from pathlib import Path
import shutil
from tqdm import tqdm
import torchvision.transforms as transforms
import torch.distributions as dist

class dataPrep:
    def __init__(self,
                 dataset: str,
                 root_dir: Path):

        self.root_dir = root_dir
        self.dataset = dataset

        if dataset == "MNIST":
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
        elif dataset == "EMNIST":
            self.train_data = datasets.EMNIST(root_dir / "raw/", split="letters", train=True, download=True)
            self.test_data = datasets.EMNIST(root_dir / "raw/", split="letters", train=False, download=True)
        elif dataset == "CIFAR10":
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
            raise ValueError("Unknown dataset")

    def make(self,
             mode: int,
             num_clients: int,
             **kwargs):

        if os.path.exists(self.root_dir / "client_data"):
            shutil.rmtree(self.root_dir / "client_data")
        client_data_path = Path(self.root_dir / "client_data")
        client_data_path.mkdir()

        if mode == 0:       # IID
            # Shuffle data
            arr = np.arange(self.num_train_data)
            np.random.shuffle(arr)
            num_data_per_client = int(self.num_train_data/num_clients)
            for i in range(num_clients):
                client_path = Path(client_data_path / str(i))
                client_path.mkdir()

                # TODO: Make this parallel for large number of clients & large datasets (Maybe not required)
                train_data = [self.train_data[j]
                              for j in tqdm(arr[i*num_data_per_client: (i+1)*num_data_per_client],
                                            desc="Splitting Data")]
                # Split data equally and send to the client
                torch.save(train_data,
                           client_data_path / str(i) / "data.pth")
        elif mode == 1:     # Non IID Balanced
            sampler = dist.dirichlet.Dirichlet(torch.empty(self.num_classes).fill_(kwargs.get('dir_alpha')))

            for i in range(num_clients):
                p_ij = sampler.sample()  # Share of jth class for ith client (always sums to 1)
                
        elif mode == 2:     # Non IID Unbalanced
            pass
        else:
            raise ValueError("Unknown mode")


d = dataPrep("MNIST", root_dir =Path("Data/"))
d.make(0, 100)




