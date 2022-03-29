import torch
from torch.nn import functional as F
from torch import nn
from torch.optim import Adam
import os

# class Model(nn.Module):
#     def __init__(self):
#         super(Model, self).__init__()
#         self.model = nn.Sequential(
#             nn.Linear(3, 3),
#             nn.Linear(3, 3)
#         )
#         self.fc = nn.Linear(3, 3)
#
#     def forward(self, x):
#         return self.model(x)
#
#
# model = Model()
# optimizer = Adam(model.parameters(), lr=0.001)
# dir = "./checkpoints"
# name = "model.pth"
# path = os.path.join(dir, name)
# print(os.path.exists(dir))

# class LaneDecoder(nn.Module):
#     def __init__(self,
#                  input_size: int = EMBED_SIZE,
#                  hidden_size: int = 256
#                  ):
#         super().__init__()
#
#         self.fc = nn.Linear(input_size, hidden_size)
#         self.offset_head = nn.Linear(
#             hidden_size, 1
#         )  # Classification: has offset or not.
#         self.yaw_head = nn.Linear(
#             hidden_size, 1
#         )  # Classification: has yaw or not.
#
#     def forward(self, x):
#         x = F.elu(self.fc(x))
#         offset = self.offset_head(x).squeeze(dim=-1)
#         yaw = self.yaw_head(x).squeeze(dim=-1)
#
#         return offset, yaw
from tools.misc import RoadOption
import numpy as np

a = np.array([[RoadOption.LEFT, RoadOption.LANEFOLLOW], [RoadOption.LEFT, RoadOption.LANEFOLLOW]])
