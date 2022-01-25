import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch import distributions as td
from torchvision import models
from algorithm.utils import create_resnet_basic_block, TanhBijector
from typing import Any, List, Tuple
from torch import TensorType

ActFunc = Any


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


# class SegDecoder(nn.Module):
#     """
#     Input shape: (6144,)
#     Output shape: (num_class, 74, 128)
#     """
#
#     def __init__(self,
#                  num_class: int = 6,  # moving obstacles, traffic lights, road markers, road, sidewalk and background.
#                  shape: Tuple[int, int, int] = (512, 3, 4)
#                  ):
#         super().__init__()
#
#         self.shape = shape
#
#         # We will upsample image with nearest neightboord interpolation between each umsample block
#         # https://distill.pub/2016/deconv-checkerboard/
#         self.up_sampled_block_0 = create_resnet_basic_block(6, 8, 512, 512)
#         self.up_sampled_block_1 = create_resnet_basic_block(12, 16, 512, 256)
#         self.up_sampled_block_2 = create_resnet_basic_block(24, 32, 256, 128)
#         self.up_sampled_block_3 = create_resnet_basic_block(48, 64, 128, 64)
#         self.up_sampled_block_4 = create_resnet_basic_block(74, 128, 64, 32)
#
#         self.last_conv_segmentation = nn.Conv2d(
#             32,
#             num_class,
#             kernel_size=(1, 1),
#             stride=(1, 1),
#             bias=False,
#         )
#
#     def forward(self, x):
#         # Flatten to [batch * horizon, C, H, W]
#         orig_shape = list(x.size())
#         x = x.view(-1, *self.shape)
#
#         # Segmentation branch
#         x = self.up_sampled_block_0(x)  # 512*8*8
#         x = self.up_sampled_block_1(x)  # 256*16*16
#         x = self.up_sampled_block_2(x)  # 128*32*32
#         x = self.up_sampled_block_3(x)  # 64*64*64
#         x = self.up_sampled_block_4(x)  # 32*128*128
#
#         x = F.softmax(self.last_conv_segmentation(x), -1)
#
#         new_shape = orig_shape[:-1] + list(x.size())[-3:]
#         x = x.view(*new_shape)
#
#         return x


class TrafficLightDecoder(nn.Module):
    def __init__(self,
                 input_size: int = 1024,
                 hidden_size: int = 256):
        super().__init__()

        self.fc = nn.Linear(input_size, hidden_size)
        self.presence_head = nn.Linear(
            hidden_size, 1
        )  # Classification: present or not
        self.signal_head = nn.Linear(
            hidden_size, 3
        )  # Classification: red, orange or green
        self.distance_head = nn.Linear(
            hidden_size, 1
        )  # Classification: near to traffic_light or not

    def forward(self, x):
        # reshape to [batch * horizon, D]
        orig_shape = list(x.size())
        x = x.contiguous().view([-1] + orig_shape[-1:])

        x = F.elu(self.fc(x))
        presence = torch.sigmoid(self.presence_head(x)).squeeze(dim=-1)
        signal = F.softmax(self.signal_head(x), dim=-1)
        distance = torch.sigmoid(self.distance_head(x)).squeeze(dim=-1)

        # reshape to [batch, horizon, D]
        presence = presence.view(orig_shape[:-1] + [-1])
        signal = signal.view(orig_shape[:-1] + [-1])
        distance = distance.view(orig_shape[:-1] + [-1])

        return presence, signal, distance


class JunctionDecoder(nn.Module):
    def __init__(self,
                 input_size: int = 1024,
                 hidden_size: int = 256,
                 ):
        super().__init__()

        self.fc = nn.Linear(input_size, hidden_size)
        self.presence_head = nn.Linear(
            hidden_size, 1
        )  # Classification present or not

    def forward(self, x):
        # reshape to [batch * horizon, D]
        orig_shape = list(x.size())
        x = x.contiguous().view([-1] + orig_shape[-1:])

        x = F.elu(self.fc(x))
        presence = torch.sigmoid(self.presence_head(x))
        presence = presence.squeeze(dim=-1)

        # reshape to [batch, horizon, D]
        presence = presence.view(orig_shape[:-1] + [-1])

        return presence


class LaneDecoder(nn.Module):
    def __init__(self,
                 input_size: int = 1024,
                 hidden_size: int = 256
                 ):
        super().__init__()

        self.fc = nn.Linear(input_size, hidden_size)
        self.offset_head = nn.Linear(
            hidden_size, 1
        )  # Classification: closely in the middle of the lane or not.
        self.yaw_head = nn.Linear(
            hidden_size, 1
        )  # Classification: closely follow the lane or not

    def forward(self, x):
        # reshape to [batch * horizon, D]
        orig_shape = list(x.size())
        x = x.contiguous().view([-1] + orig_shape[-1:])

        x = F.elu(self.fc(x))
        offset = torch.sigmoid(self.offset_head(x))
        yaw = torch.sigmoid(self.yaw_head(x))

        # reshape to [batch, horizon, D]
        offset = offset.view(orig_shape[:-1] + [-1])
        yaw = yaw.view(orig_shape[:-1] + [-1])

        return offset, yaw


class ActionDecoder(nn.Module):
    """
    It outputs a distribution parameterized by mean and std, later to be
    transformed by a TanhBijector.
    """

    def __init__(self,
                 action_size: int,
                 input_size: int = 7168,
                 units: Tuple[int,] = (1024, 256),
                 act: ActFunc = None,
                 min_std: float = 1e-4,
                 init_std: float = 5.0,
                 mean_scale: float = 5.0):
        """Initializes Policy

        Args:
            input_size (int): Input size to network
            action_size (int): Action space size
            units (Tuple[int,]): Size of the hidden layers
            act (Any): Activation function
            min_std (float): Minimum std for output distribution
            init_std (float): Intitial std
            mean_scale (float): Augmenting mean output from FC network
        """
        super().__init__()
        self.units = units
        self.act = act
        if not act:
            self.act = nn.ReLU
        self.min_std = min_std
        self.init_std = init_std
        self.mean_scale = mean_scale
        self.action_size = action_size

        self.layers = []
        self.softplus = nn.Softplus()

        # MLP Construction
        cur_size = input_size
        for unit in self.units:
            self.layers.extend([nn.Linear(cur_size, unit), self.act()])
            cur_size = unit

        self.layers.append(nn.Linear(cur_size, 2 * action_size))

        self.model = nn.Sequential(*self.layers)

    # Returns distribution
    def forward(self, x):
        # reshape to [batch * horizon, D]
        orig_shape = list(x.size())
        x = x.contiguous().view([-1] + orig_shape[-1:])

        raw_init_std = np.log(np.exp(self.init_std) - 1)
        x = self.model(x)

        # reshape to [batch, horizon, D]
        x = x.view(orig_shape[:-1] + [-1])

        mean, std = torch.chunk(x, 2, dim=-1)
        mean = self.mean_scale * torch.tanh(mean / self.mean_scale)
        std = self.softplus(std + raw_init_std) + self.min_std
        dist = td.Normal(mean, std)
        transforms = [TanhBijector()]
        dist = td.transformed_distribution.TransformedDistribution(
            dist, transforms)
        dist = td.Independent(dist, 1)

        return dist


class MyModel(nn.Module):
    def __init__(self,
                 num_output: int = 3,
                 cell_size: int = 1024
                 ):
        super().__init__()

        self.cell_size = cell_size
        self.state = None

        self.encoder = ConvEncoder()
        self.cell = nn.GRU(6144, cell_size, batch_first=True)
        self.tl_decoder = TrafficLightDecoder()
        self.juction_decoder = JunctionDecoder()
        self.lane_decoder = LaneDecoder()
        self.action_decoder = ActionDecoder(num_output)

    def reset(self):
        self.state = None

    def encode(self, x):
        orig_shape = list(x.size())

        static_embed = self.encoder(x)

        if self.state is None:
            state_shape = [1, orig_shape[0], self.cell_size]
            self.state = torch.zeros(state_shape).to(x.device)
        if static_embed.ndim == 2:
            static_embed = static_embed.unsqueeze(dim=1)
        dynamic_embed, self.state = self.cell(static_embed, self.state)

        return dynamic_embed, static_embed

    def decode(self, dynamic_embed, static_embed):
        tl_pred = self.tl_decoder(dynamic_embed)
        junction_pred = self.juction_decoder(dynamic_embed)
        lane_pred = self.lane_decoder(dynamic_embed)
        embed = torch.concat([dynamic_embed, static_embed], dim=-1)
        action_dist = self.action_decoder(embed)
        return tl_pred, junction_pred, lane_pred, action_dist

    def forward(self, x):
        dynamic_embed, static_embed = self.encode(x)
        embed = torch.concat([dynamic_embed, static_embed], dim=-1)

        action_dist = self.action_decoder(embed)
        action = action_dist.rsample()

        return action
