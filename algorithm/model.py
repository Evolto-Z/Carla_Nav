import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch import distributions as td
from torchvision import models
from algorithm.nn_modules import TanhBijector
from typing import Any, List, Tuple
from torch import Tensor
from tools.misc import RoadOption

ActFunc = Any
DYNAMIC_EMBED_SIZE = 2048
STATIC_EMBED_SIZE = 6144
EMBED_SIZE = DYNAMIC_EMBED_SIZE + STATIC_EMBED_SIZE


class ConvEncoder(nn.Module):
    """
    Input shape: (3, 216, 288)
    Output shape: (512, 3, 4)  (not flattened)
                  (6144,)  (flattened)
    """

    def __init__(self):
        super().__init__()
        resnet18 = models.resnet18(pretrained=False)

        # See https://arxiv.org/abs/1606.02147v1 section 4: Information-preserving
        # dimensionality changes
        #
        # "When downsampling, the first 1x1 projection of the convolutional branch is performed
        # with a stride of 2 in both dimensions, which effectively discards 75% of the input.
        # Increasing the filter size to 2x2 allows to take the full input into consideration,
        # and thus improves the information flow and accuracy."

        assert resnet18.layer2[0].downsample[0].kernel_size == (1, 1)
        assert resnet18.layer3[0].downsample[0].kernel_size == (1, 1)
        assert resnet18.layer4[0].downsample[0].kernel_size == (1, 1)
        assert resnet18.layer2[0].downsample[0].stride == (2, 2)
        assert resnet18.layer3[0].downsample[0].stride == (2, 2)
        assert resnet18.layer4[0].downsample[0].stride == (2, 2)

        resnet18.layer2[0].downsample[0].kernel_size = (2, 2)
        resnet18.layer3[0].downsample[0].kernel_size = (2, 2)
        resnet18.layer4[0].downsample[0].kernel_size = (2, 2)

        assert resnet18.layer2[0].downsample[0].kernel_size == (2, 2)
        assert resnet18.layer3[0].downsample[0].kernel_size == (2, 2)
        assert resnet18.layer4[0].downsample[0].kernel_size == (2, 2)
        assert resnet18.layer2[0].downsample[0].stride == (2, 2)
        assert resnet18.layer3[0].downsample[0].stride == (2, 2)
        assert resnet18.layer4[0].downsample[0].stride == (2, 2)

        new_conv1 = nn.Conv2d(
            3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        resnet18.conv1 = new_conv1

        self.encoder = nn.Sequential(
            *(list(resnet18.children())[:-2]),
        )  # resnet18_no_fc_no_avgpool
        self.last_conv_downsample = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=(2, 2), stride=(2, 2), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        )

    def forward(self, x):
        # reshape to [batch * horizon, C, H, W]
        orig_shape = list(x.size())
        x = x.view([-1] + orig_shape[-3:])

        # resnet18 without last fc and avg pooling
        x = self.encoder(x)
        x = self.last_conv_downsample(x)

        # reshape to [batch, horizon, D]
        x = x.view(orig_shape[:-3] + [-1])

        return x


class TrafficLightDecoder(nn.Module):
    def __init__(self,
                 input_size: int = EMBED_SIZE,
                 hidden_size: int = 256):
        super().__init__()

        self.fc = nn.Linear(input_size, hidden_size)
        self.affected_head = nn.Linear(
            hidden_size, 1
        )  # Classification: affected or not
        self.signal_head = nn.Linear(
            hidden_size, 3
        )  # Classification: red, orange or green

    def forward(self, x):
        x = F.elu(self.fc(x))
        affected = self.affected_head(x).squeeze(dim=-1)
        signal = self.signal_head(x)

        return affected, signal


class JunctionDecoder(nn.Module):
    def __init__(self,
                 input_size: int = STATIC_EMBED_SIZE,
                 hidden_size: int = 256,
                 ):
        super().__init__()

        self.fc = nn.Linear(input_size, hidden_size)
        self.presence_head = nn.Linear(
            hidden_size, 1
        )  # Classification present or not

    def forward(self, x):
        x = F.elu(self.fc(x))
        presence = self.presence_head(x).squeeze(dim=-1)

        return presence


class SpeedDecoder(nn.Module):
    def __init__(self,
                 input_size: int = DYNAMIC_EMBED_SIZE,
                 hidden_size: int = 256,
                 ):
        super().__init__()

        self.fc = nn.Linear(input_size, hidden_size)
        self.speed_head = nn.Linear(
            hidden_size, 1
        )  # km/h

    def forward(self, x):
        x = F.elu(self.fc(x))
        speed = self.speed_head(x).squeeze(dim=-1)

        return speed


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


class ActionDecoder(nn.Module):
    """
    It outputs a distribution parameterized by mean and std, later to be
    transformed by a TanhBijector.
    """

    def __init__(self,
                 action_size: int,
                 input_size: int = EMBED_SIZE,
                 units: Tuple[int] = (1024, 256),
                 num_head: int = 6,
                 act: ActFunc = None,
                 min_std: float = 1e-4,
                 init_std: float = 5.0,
                 mean_scale: float = 5.0
                 ):
        """Initializes Policy

        Args:
            action_size (int): Action space size
            input_size (int): Input size to network
            units (Tuple[int,]): Size of the hidden layers
            act (Any): Activation function
            min_std (float): Minimum std for output distribution
            init_std (float): Intitial std
            mean_scale (float): Augmenting mean output from FC network
        """
        super().__init__()
        self.units = units
        self.num_head = num_head
        self.act = act
        if not act:
            self.act = nn.ReLU
        self.min_std = min_std
        self.init_std = init_std
        self.mean_scale = mean_scale
        self.action_size = action_size

        self.heads = []
        self.softplus = nn.Softplus()

        # MLP Construction
        self.fc = nn.Sequential(
            nn.Linear(input_size, self.units[0]),
            self.act()
        )
        for _ in range(num_head):
            layers = []
            cur_size = self.units[0]
            for unit in self.units[1:]:
                layers.extend([nn.Linear(cur_size, unit), self.act()])
                cur_size = unit
            layers.append(nn.Linear(cur_size, 2 * action_size))
            head = nn.Sequential(*layers)
            self.heads.append(head)

    # Returns distribution
    def forward(self, x: Tensor, command: List[RoadOption]):
        x = self.fc(x)
        x = torch.split(x, 1, dim=0)
        x = list(map(lambda item: self.heads[item[0].value - 1](item[1]), zip(command, x)))
        x = torch.concat(x, dim=0)

        mean, std = torch.chunk(x, 2, dim=-1)
        mean = self.mean_scale * torch.tanh(mean / self.mean_scale)
        raw_init_std = np.log(np.exp(self.init_std) - 1)
        std = self.softplus(std + raw_init_std) + self.min_std
        dist = td.Normal(mean, std)
        transforms = [TanhBijector()]
        dist = td.transformed_distribution.TransformedDistribution(dist, transforms)
        dist = td.Independent(dist, 1)

        return dist


class ValueDecoder(nn.Module):
    """
    It outputs the mean of cumulative return.
    """

    def __init__(self,
                 action_size: int = 2,
                 input_size: int = EMBED_SIZE,
                 units: Tuple[int] = (1024, 256),
                 act: ActFunc = None
                 ):
        """Initializes Policy

        Args:
            input_size (int): Input size to network
            units (Tuple[int,]): Size of the hidden layers
            act (Any): Activation function
        """
        super().__init__()
        self.units = units
        self.act = act
        if not act:
            self.act = nn.ELU

        self.softplus = nn.Softplus()

        # MLP Construction
        self.fc = nn.Sequential(
            nn.Linear(input_size, self.units[0]),
            self.act()
        )
        self.layers = []
        cur_size = self.units[0] + action_size
        for unit in self.units[1:]:
            self.layers.extend([nn.Linear(cur_size, unit), self.act()])
            cur_size = unit

        self.layers.append(nn.Linear(cur_size, 1))

        self.model = nn.Sequential(*self.layers)

    # Returns distribution
    def forward(self, x: Tensor, action: Tensor):
        x = self.fc(x)
        x = torch.concat([x, action], dim=-1)
        value = self.model(x).squeeze(dim=-1)

        return value


class MyModel(nn.Module):
    def __init__(self,
                 action_dim: int = 2,
                 cell_size: int = DYNAMIC_EMBED_SIZE,
                 ):
        super().__init__()

        self.cell_size = cell_size
        self.cell_state = None

        self.encoder = ConvEncoder()
        self.cell = nn.GRU(STATIC_EMBED_SIZE, cell_size, batch_first=True)
        self.tl_decoder = TrafficLightDecoder(EMBED_SIZE)
        self.junction_decoder = JunctionDecoder(STATIC_EMBED_SIZE)
        self.speed_decoder = SpeedDecoder(DYNAMIC_EMBED_SIZE)
        self.actor = ActionDecoder(action_dim)
        self.critic = ValueDecoder()

    def reset_cell(self):
        self.cell_state = None

    def encode(self, x: Tensor):
        orig_shape = list(x.size())

        static_embed = self.encoder(x)
        if self.cell_state is None:
            state_shape = [1, orig_shape[0], self.cell_size]
            self.cell_state = torch.zeros(state_shape).to(x.device)

        flag = False
        if static_embed.ndim == 2:
            static_embed = static_embed.unsqueeze(dim=1)
            flag = True
        dynamic_embed, self.cell_state = self.cell(static_embed.detach(), self.cell_state)
        if flag:
            static_embed = static_embed.squeeze(dim=1)
            dynamic_embed = dynamic_embed.squeeze(dim=1)

        return dynamic_embed, static_embed

    def auxiliary_decode(self,
                         dynamic_embed: Tensor,
                         static_embed: Tensor
                         ):
        embed = torch.concat([dynamic_embed, static_embed], dim=-1)
        tl_pred = self.tl_decoder(embed)
        junction_pred = self.junction_decoder(static_embed)
        speed_pred = self.speed_decoder(dynamic_embed)

        return tl_pred, junction_pred, speed_pred

    def actor_decode(self,
                     dynamic_embed: Tensor,
                     static_embed: Tensor,
                     command: [List[RoadOption], RoadOption]
                     ):
        embed = torch.concat([dynamic_embed, static_embed], dim=-1)
        if not isinstance(command, list):
            command = [command] * len(dynamic_embed)
        action_dist = self.actor(embed, command)

        return action_dist

    def critic_decode(self,
                      dynamic_embed: Tensor,
                      static_embed: Tensor,
                      action: Tensor
                      ):
        embed = torch.concat([dynamic_embed, static_embed], dim=-1)
        value = self.critic(embed, action)

        return value

    def forward(self, x: Tensor, command: List[RoadOption]):
        dynamic_embed, static_embed = self.encode(x)

        return self.actor_decode(dynamic_embed, static_embed, command)
