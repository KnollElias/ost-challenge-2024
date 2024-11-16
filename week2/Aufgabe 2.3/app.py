import torch.nn as nn

class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()
        self.linear1 = nn.Linear(784, 128)
        self.relu1 = nn.ReLU()

        # hidden layer
        self.linear2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()

        self.linear_out = nn.Linear(64, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear_out(x)
        x = self.softmax(x)
        return x
