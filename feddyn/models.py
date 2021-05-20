import torch.nn as nn
import torch


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
                        nn.Softmax(dim=1),
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
                            nn.Softmax(dim=1)
        )

    @property
    def num_param(self):
        return sum([p.numel() for p in self.parameters() if p.requires_grad])

    def forward(self, x):
        x = self.model(x)
        x = x.flatten(1)
        print(x.shape)
        return self.classifier(x)

# x = torch.randn(2, 3, 28, 28)
# m = CNN()
# y = m(x)
# print(y.sum(1))
# print(m.num_param)
