import torch
from torch.nn import functional as F
from torch import nn
from torch.optim import Adam
import os

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 3),
            nn.Linear(3, 3)
        )
        self.fc = nn.Linear(3, 3)

    def forward(self, x):
        return self.model(x)


model = Model()
optimizer = Adam(model.parameters(), lr=0.001)
dir = "./checkpoints"
name = "model.pth"
path = os.path.join(dir, name)
print(os.path.exists(dir))
