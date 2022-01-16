import numpy as np
from typing import Any, List, Tuple
import torch
from torch import nn
from torch import distributions as td
from torchvision import models
from torch import TensorType
from torch.nn import functional as F

ActFunc = Any


class Reshape(nn.Module):
    """
    Standard module that reshapes/views a tensor
    """

    def __init__(self, shape: List):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


def create_resnet_basic_block(width_output_feature_map, height_output_feature_map, nb_channel_in, nb_channel_out):
    basic_block = nn.Sequential(
        nn.Upsample(size=(width_output_feature_map, height_output_feature_map), mode="nearest"),
        nn.Conv2d(
            nb_channel_in,
            nb_channel_out,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        ),
        nn.BatchNorm2d(
            nb_channel_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        ),
        nn.ReLU(inplace=True),
        nn.Conv2d(
            nb_channel_out,
            nb_channel_out,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        ),
        nn.BatchNorm2d(
            nb_channel_out, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        ),
    )
    return basic_block


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
        # Flatten to [batch * horizon, 3, H, W]
        orig_shape = list(x.size())
        x = x.view(-1, *(orig_shape[-3:]))

        # resnet18 without last fc and avg pooling
        x = self.encoder(x)
        x = self.last_conv_downsample(x)

        new_shape = orig_shape[:-3] + [-1]
        x = x.view(*new_shape)
        return x


class SegDecoder(nn.Module):
    """
    Input shape: (6144,)
    Output shape: (num_class, 74, 128)
    """

    def __init__(self, num_class, shape=(512, 3, 4)):
        super().__init__()

        self.shape = shape

        # We will upsample image with nearest neightboord interpolation between each umsample block
        # https://distill.pub/2016/deconv-checkerboard/
        self.up_sampled_block_0 = create_resnet_basic_block(6, 8, 512, 512)
        self.up_sampled_block_1 = create_resnet_basic_block(12, 16, 512, 256)
        self.up_sampled_block_2 = create_resnet_basic_block(24, 32, 256, 128)
        self.up_sampled_block_3 = create_resnet_basic_block(48, 64, 128, 64)
        self.up_sampled_block_4 = create_resnet_basic_block(74, 128, 64, 32)

        self.last_conv_segmentation = nn.Conv2d(
            32,
            num_class,
            kernel_size=(1, 1),
            stride=(1, 1),
            bias=False,
        )

    def forward(self, x):
        # Flatten to [batch * horizon, C, H, W]
        orig_shape = list(x.size())
        x = x.view(-1, *self.shape)

        # Segmentation branch
        x = self.up_sampled_block_0(x)  # 512*8*8
        x = self.up_sampled_block_1(x)  # 256*16*16
        x = self.up_sampled_block_2(x)  # 128*32*32
        x = self.up_sampled_block_3(x)  # 64*64*64
        x = self.up_sampled_block_4(x)  # 32*128*128

        x = F.softmax(self.last_conv_segmentation(x), -1)

        new_shape = orig_shape[:-1] + list(x.size())[-3:]
        x = x.view(*new_shape)

        return x


class TrafficLightDecoder(nn.Module):
    def __init__(self, hidden_size=1024):
        super().__init__()

        self.state_size = 6144

        self.fc = nn.Linear(self.state_size, hidden_size)
        self.presence_head = nn.Linear(
            hidden_size, 1
        )  # Classification: present or not
        self.signal_head = nn.Linear(
            hidden_size, 3
        )  # Classification: red, orange or green
        self.distance_head = nn.Linear(
            hidden_size, 2
        )  # Classification: near or far to traffic_light

    def forward(self, x):
        # Flatten to [batch * horizon, N]
        orig_shape = list(x.size())
        x = x.view(-1, self.state_size)

        x = F.elu(self.fc(x))

        new_shape = orig_shape[:-1] + [-1]

        presence = torch.sigmoid(self.presence_head(x))
        signal = F.softmax(self.signal_head(x).view(*new_shape), dim=-1)
        distance = F.softmax(self.distance_head(x).view(*new_shape), dim=-1)

        presence = presence.view(*new_shape)
        signal = signal.view(*new_shape)
        distance = distance.view(*new_shape)

        return presence, signal, distance


class JunctionDecoder(nn.Module):
    def __init__(self, hidden_size=1024, num_dist_class=2):
        super().__init__()

        self.state_size = 6144

        self.fc = nn.Linear(self.state_size, hidden_size)
        self.presence_head = nn.Linear(
            hidden_size, 1
        )  # Classification present or not

    def forward(self, x):
        # Flatten to [batch * horizon, N]
        orig_shape = list(x.size())
        x = x.view(-1, self.state_size)

        x = F.elu(self.fc(x))

        new_shape = orig_shape[:-1] + [-1]

        presence = torch.sigmoid(self.presence_head(x))

        presence = presence.view(*new_shape)

        return presence


class LaneDecoder(nn.Module):
    def __init__(self, hidden_size=1024, num_dist_class=2):
        super().__init__()

        self.state_size = 6144

        self.fc = nn.Linear(self.state_size, hidden_size)
        self.offset_head = nn.Linear(
            hidden_size, 3
        )  # Classification: left, middle or right
        self.yaw_head = nn.Linear(
            hidden_size, 3
        )  # Classification: left, middle, or right

    def forward(self, x):
        # Flatten to [batch * horizon, N]
        orig_shape = list(x.size())
        x = x.view(-1, self.state_size)

        x = F.elu(self.fc(x))

        new_shape = orig_shape[:-1] + [-1]

        offset = F.softmax(self.offset_head(x).view(*new_shape), dim=-1)
        yaw = F.softmax(self.yaw_head(x).view(*new_shape), dim=-1)

        offset = offset.view(*new_shape)
        yaw = yaw.view(*new_shape)

        return offset, yaw


# Represents dreamer policy
class ActionDecoder(nn.Module):
    """
    It outputs a distribution parameterized by mean and std, later to be
    transformed by a TanhBijector.
    """

    def __init__(self,
                 input_size: int,
                 action_size: int,
                 layers: int,
                 units: int,
                 dist: str = "tanh_normal",
                 act: ActFunc = None,
                 min_std: float = 1e-4,
                 init_std: float = 5.0,
                 mean_scale: float = 5.0):
        """Initializes Policy

        Args:
            input_size (int): Input size to network
            action_size (int): Action space size
            layers (int): Number of layers in network
            units (int): Size of the hidden layers
            dist (str): Output distribution, with tanh_normal implemented
            act (Any): Activation function
            min_std (float): Minimum std for output distribution
            init_std (float): Intitial std
            mean_scale (float): Augmenting mean output from FC network
        """
        super().__init__()
        self.layrs = layers
        self.units = units
        self.dist = dist
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
        for _ in range(self.layrs):
            self.layers.extend([nn.Linear(cur_size, self.units), self.act()])
            cur_size = self.units
        if self.dist == "tanh_normal":
            self.layers.append(nn.Linear(cur_size, 2 * action_size))
        elif self.dist == "onehot":
            self.layers.append(nn.Linear(cur_size, action_size))
        self.model = nn.Sequential(*self.layers)

    # Returns distribution
    def forward(self, x):
        raw_init_std = np.log(np.exp(self.init_std) - 1)
        x = self.model(x)
        if self.dist == "tanh_normal":
            mean, std = torch.chunk(x, 2, dim=-1)
            mean = self.mean_scale * torch.tanh(mean / self.mean_scale)
            std = self.softplus(std + raw_init_std) + self.min_std
            dist = td.Normal(mean, std)
            transforms = [nn.TanhBijector()]
            dist = td.transformed_distribution.TransformedDistribution(
                dist, transforms)
            dist = td.Independent(dist, 1)
        elif self.dist == "onehot":
            dist = td.OneHotCategorical(logits=x)
            raise NotImplementedError("Atari not implemented yet!")
        return dist


# Represents TD model in PlaNET
class RSSM(nn.Module):
    """RSSM is the core recurrent part of the PlaNET module. It consists of
    two networks, one (obs) to calculate posterior beliefs and states and
    the second (img) to calculate prior beliefs and states. The prior network
    takes in the previous state and action, while the posterior network takes
    in the previous state, action, and a latent embedding of the most recent
    observation.
    """

    def __init__(self,
                 action_size: int,
                 embed_size: int,
                 stoch: int = 30,
                 deter: int = 200,
                 hidden: int = 200,
                 act: ActFunc = None):
        """Initializes RSSM

        Args:
            action_size (int): Action space size
            embed_size (int): Size of ConvEncoder embedding
            stoch (int): Size of the distributional hidden state
            deter (int): Size of the deterministic hidden state
            hidden (int): General size of hidden layers
            act (Any): Activation function
        """
        super().__init__()
        self.stoch_size = stoch
        self.deter_size = deter
        self.hidden_size = hidden
        self.act = act
        if act is None:
            self.act = nn.ELU

        self.obs1 = nn.Linear(embed_size + deter, hidden)
        self.obs2 = nn.Linear(hidden, 2 * stoch)

        self.cell = nn.GRUCell(self.hidden_size, hidden_size=self.deter_size)
        self.img1 = nn.Linear(stoch + action_size, hidden)
        self.img2 = nn.Linear(deter, hidden)
        self.img3 = nn.Linear(hidden, 2 * stoch)

        self.softplus = nn.Softplus

        self.device = (torch.device("cuda")
                       if torch.cuda.is_available() else torch.device("cpu"))

    def get_initial_state(self, batch_size: int) -> List[TensorType]:
        """Returns the inital state for the RSSM, which consists of mean,
        std for the stochastic state, the sampled stochastic hidden state
        (from mean, std), and the deterministic hidden state, which is
        pushed through the GRUCell.

        Args:
            batch_size (int): Batch size for initial state

        Returns:
            List of tensors
        """
        return [
            torch.zeros(batch_size, self.stoch_size).to(self.device),
            torch.zeros(batch_size, self.stoch_size).to(self.device),
            torch.zeros(batch_size, self.stoch_size).to(self.device),
            torch.zeros(batch_size, self.deter_size).to(self.device),
        ]

    def observe(self,
                embed: TensorType,
                action: TensorType,
                state: List[TensorType] = None
                ) -> Tuple[List[TensorType], List[TensorType]]:
        """Returns the corresponding states from the embedding from ConvEncoder
        and actions. This is accomplished by rolling out the RNN from the
        starting state through each index of embed and action, saving all
        intermediate states between.

        Args:
            embed (TensorType): ConvEncoder embedding
            action (TensorType): Actions
            state (List[TensorType]): Initial state before rollout

        Returns:
            Posterior states and prior states (both List[TensorType])
        """
        if state is None:
            state = self.get_initial_state(action.size()[0])

        if embed.dim() <= 2:
            embed = torch.unsqueeze(embed, 1)

        if action.dim() <= 2:
            action = torch.unsqueeze(action, 1)

        embed = embed.permute(1, 0, 2)
        action = action.permute(1, 0, 2)

        priors = [[] for i in range(len(state))]
        posts = [[] for i in range(len(state))]
        last = (state, state)
        for index in range(len(action)):
            # Tuple of post and prior
            last = self.obs_step(last[0], action[index], embed[index])
            [o.append(s) for s, o in zip(last[0], posts)]
            [o.append(s) for s, o in zip(last[1], priors)]

        prior = [torch.stack(x, dim=0) for x in priors]
        post = [torch.stack(x, dim=0) for x in posts]

        prior = [e.permute(1, 0, 2) for e in prior]
        post = [e.permute(1, 0, 2) for e in post]

        return post, prior

    def imagine(self, action: TensorType,
                state: List[TensorType] = None) -> List[TensorType]:
        """Imagines the trajectory starting from state through a list of actions.
        Similar to observe(), requires rolling out the RNN for each timestep.

        Args:
            action (TensorType): Actions
            state (List[TensorType]): Starting state before rollout

        Returns:
            Prior states
        """
        if state is None:
            state = self.get_initial_state(action.size()[0])

        action = action.permute(1, 0, 2)

        indices = range(len(action))
        priors = [[] for _ in range(len(state))]
        last = state
        for index in indices:
            last = self.img_step(last, action[index])
            [o.append(s) for s, o in zip(last, priors)]

        prior = [torch.stack(x, dim=0) for x in priors]
        prior = [e.permute(1, 0, 2) for e in prior]
        return prior

    def obs_step(
            self, prev_state: TensorType, prev_action: TensorType,
            embed: TensorType) -> Tuple[List[TensorType], List[TensorType]]:
        """Runs through the posterior model and returns the posterior state

        Args:
            prev_state (TensorType): The previous state
            prev_action (TensorType): The previous action
            embed (TensorType): Embedding from ConvEncoder

        Returns:
            Post and Prior state
      """
        prior = self.img_step(prev_state, prev_action)
        x = torch.cat([prior[3], embed], dim=-1)
        x = self.obs1(x)
        x = self.act()(x)
        x = self.obs2(x)
        mean, std = torch.chunk(x, 2, dim=-1)
        std = self.softplus()(std) + 0.1
        stoch = self.get_dist(mean, std).rsample()
        post = [mean, std, stoch, prior[3]]
        return post, prior

    def img_step(self, prev_state: TensorType,
                 prev_action: TensorType) -> List[TensorType]:
        """Runs through the prior model and returns the prior state

        Args:
            prev_state (TensorType): The previous state
            prev_action (TensorType): The previous action

        Returns:
            Prior state
        """
        x = torch.cat([prev_state[2], prev_action], dim=-1)
        x = self.img1(x)
        x = self.act()(x)
        deter = self.cell(x, prev_state[3])
        x = deter
        x = self.img2(x)
        x = self.act()(x)
        x = self.img3(x)
        mean, std = torch.chunk(x, 2, dim=-1)
        std = self.softplus()(std) + 0.1
        stoch = self.get_dist(mean, std).rsample()
        return [mean, std, stoch, deter]

    def get_feature(self, state: List[TensorType]) -> TensorType:
        # Constructs feature for input to reward, decoder, actor, critic
        return torch.cat([state[2], state[3]], dim=-1)

    def get_dist(self, mean: TensorType, std: TensorType) -> TensorType:
        return td.Normal(mean, std)


# Represents all models in Dreamer, unifies them all into a single interface
class DreamerModel(nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):

        super().__init__()
        self.depth = model_config["depth_size"]
        self.deter_size = model_config["deter_size"]
        self.stoch_size = model_config["stoch_size"]
        self.hidden_size = model_config["hidden_size"]

        self.action_size = action_space.shape[0]

        self.encoder = ConvEncoder(self.depth)
        self.decoder = ConvDecoder(
            self.stoch_size + self.deter_size, depth=self.depth)
        self.reward = DenseDecoder(self.stoch_size + self.deter_size, 1, 2,
                                   self.hidden_size)
        self.dynamics = RSSM(
            self.action_size,
            32 * self.depth,
            stoch=self.stoch_size,
            deter=self.deter_size)
        self.actor = ActionDecoder(self.stoch_size + self.deter_size,
                                   self.action_size, 4, self.hidden_size)
        self.value = DenseDecoder(self.stoch_size + self.deter_size, 1, 3,
                                  self.hidden_size)
        self.state = None

        self.device = (torch.device("cuda")
                       if torch.cuda.is_available() else torch.device("cpu"))

    def policy(self, obs: TensorType, state: List[TensorType], explore=True
               ) -> Tuple[TensorType, List[float], List[TensorType]]:
        """Returns the action. Runs through the encoder, recurrent model,
        and policy to obtain action.
        """
        if state is None:
            self.state = self.get_initial_state(batch_size=obs.shape[0])
        else:
            self.state = state
        post = self.state[:4]
        action = self.state[4]

        embed = self.encoder(obs)
        post, _ = self.dynamics.obs_step(post, action, embed)
        feat = self.dynamics.get_feature(post)

        action_dist = self.actor(feat)
        if explore:
            action = action_dist.sample()
        else:
            action = action_dist.mean
        logp = action_dist.log_prob(action)

        self.state = post + [action]
        return action, logp, self.state

    def imagine_ahead(self, state: List[TensorType],
                      horizon: int) -> TensorType:
        """Given a batch of states, rolls out more state of length horizon.
        """
        start = []
        for s in state:
            s = s.contiguous().detach()
            shpe = [-1] + list(s.size())[2:]
            start.append(s.view(*shpe))

        def next_state(state):
            feature = self.dynamics.get_feature(state).detach()
            action = self.actor(feature).rsample()
            next_state = self.dynamics.img_step(state, action)
            return next_state

        last = start
        outputs = [[] for i in range(len(start))]
        for _ in range(horizon):
            last = next_state(last)
            [o.append(s) for s, o in zip(last, outputs)]
        outputs = [torch.stack(x, dim=0) for x in outputs]

        imag_feat = self.dynamics.get_feature(outputs)
        return imag_feat

    def get_initial_state(self) -> List[TensorType]:
        self.state = self.dynamics.get_initial_state(1) + [
            torch.zeros(1, self.action_space.shape[0]).to(self.device)
        ]
        return self.state

    def value_function(self) -> TensorType:
        return None