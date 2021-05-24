import torch.nn as nn
import torch
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

class MLP(nn.Module):
    def __init__(self,
                 in_feat: int = 784):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
                        nn.Linear(in_feat, 200),
                        nn.ReLU(inplace=True),
                        nn.Linear(200, 100),
                        nn.ReLU(inplace=True),
                        nn.Linear(100, 10),
                        nn.LogSoftmax(dim=1),
        )

    @property
    def num_param(self):
        return sum([p.numel() for p in self.parameters() if p.requires_grad])

    def forward(self, x):
        return self.model(x)

class CNN(nn.Module):
    def __init__(self,
                 in_ch :int = 3):
        super(CNN, self).__init__()

        self.model = nn.Sequential(
                        nn.Conv2d(in_ch, 64, 5),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(64, 64, 5),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
                            nn.Linear(1024, 512),
                            nn.ReLU(),
                            nn.Linear(512, 10),
                            nn.LogSoftmax(dim=1)
        )

    @property
    def num_param(self):
        return sum([p.numel() for p in self.parameters() if p.requires_grad])

    def forward(self, x):
        x = self.model(x)
        x = x.flatten(1)
        print(x.shape)
        return self.classifier(x)

if __name__ == "__main__":
    pass
    # x = torch.randn(2, 3, 28, 28)
    # m = CNN()
    # n = CNN()
    # y = m(x)
    # print(list(m.named_parameters()))
    # print(y.sum(1))
    # print(m.num_param)
    # print(list(m.state_dict().values()) + list(n.state_dict().values()))
