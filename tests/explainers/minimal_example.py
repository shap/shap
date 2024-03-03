import torch
from torch import nn


class Net(nn.Module):
    """ Basic conv net.
    """

    def __init__(self):
        super().__init__()
        # Testing several different activations
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.Tanh(),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.ConvTranspose2d(20, 20, 1),
            nn.AdaptiveAvgPool2d(output_size=(4, 4)),
            nn.Softplus(),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(320, 50),
            nn.BatchNorm1d(50),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.ELU(),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        """ Run the model.
        """
        x = self.conv_layers(x)
        x = x.view(-1, 320)
        x = self.fc_layers(x)
        return x

model = Net()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

device = torch.device("cpu")

model.to(device)
data = torch.randn(32, 1, 28, 28)
model(data)
