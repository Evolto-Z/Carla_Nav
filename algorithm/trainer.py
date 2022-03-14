import torch.cuda
from torch.optim import Adam
from torch.nn import functional as F
from algorithm.model import ValueDecoder, MyModel
from rllib_integration.helper import join_dicts
from typing import Any, List, Tuple
import os

DEFAULT_CONFIG = {
    "gamma": 0.99,  # reward discount
    "alpha": 0.2,  # entropy weight
    "polyak": 0.995,  # synchronize weight
    "lr_il": 0.001,
    "loss_weights": [1, 1, 1, 1, 1],  # traffic lights, junction, lane, actor and critic losses
    "lr_rl_actor": 0.001,
    "lr_rl_critic": 0.001
}


class MyTrainer:
    def __init__(self, model: MyModel, config={}):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)
        self.critic = None
        self.target_critic = ValueDecoder().to(self.device)
        for p in self.target_critic.parameters():
            p.requires_grad = False

        # SAC parameters
        config = join_dicts(DEFAULT_CONFIG, config)
        self.gamma = config["gamma"]
        self.alpha = config["alpha"]
        self.polyak = config["polyak"]

        self.optimizer_il = Adam(self.model.parameters(), lr=config["lr_il"])
        self.loss_weights = torch.FloatTensor(config["loss_weights"])

        self.optimizer_rl_actor = Adam(self.model.actor.parameters(), lr=config["lr_rl_actor"])
        self.optimizer_rl_critic = Adam(self.model.critic.parameters(), lr=config["lr_rl_critic"])

        self.il_phase = True  # False for rl phase

        self.critic = self.model.critic
        self.target_critic.load_state_dict(self.critic.state_dict())

    def set_il_phase(self, value: bool):
        self.il_phase = value

        self.model.encoder.requires_grad_(value)
        self.model.tl_decoder.requires_grad_(value)
        self.model.juction_decoder.requires_grad_(value)
        self.model.lane_decoder.requires_grad_(value)

        self.model.encoder.train(value)
        self.model.tl_decoder.train(value)
        self.model.juction_decoder.train(value)
        self.model.lane_decoder.train(value)

    @torch.no_grad()
    def sync_value_decoder(self):
        for p_target, p in zip(self.target_critic.parameters(), self.critic.parameters()):
            p_target.data.mul_(self.polyak)
            p_target.data.add_((1 - self.polyak) * p)

    def compute_td_estimate(self, state, action):
        if self.il_phase:
            obs, command = state
            embed = self.model.encode(obs)
            td_estimate = self.model.critic_decode(*embed, action)
        else:
            dynamic_embed, static_embed, command = state
            td_estimate = self.model.critic_decode(dynamic_embed, static_embed, action)

        return td_estimate

    @torch.no_grad()
    def compute_td_target(self, reward, next_state, done):
        if self.il_phase:
            obs, command = next_state
            embed = self.model.encode(obs)
            embed = torch.concat(embed, dim=-1)
        else:
            dynamic_embed, static_embed, command = next_state
            embed = torch.concat([dynamic_embed, static_embed], dim=-1)

        action_dist = self.model(embed, command)
        action = action_dist.sample()
        entropy = -action_dist.log_prob(action)
        next_value = self.target_critic(embed, action)
        td_target = reward + (1 - done) * self.gamma * (next_value + self.alpha * entropy)

        return td_target

    def il_loss_func(self, pred, label):
        tl_pred, junction_pred, speed_pred, action_dist, td_estimate = pred
        tl_label, junction_label, speed_label, action_label, td_target = label

        # traffic light loss
        tl_loss = F.binary_cross_entropy_with_logits(tl_pred[0], tl_label[0])  # affected
        tl_loss += (F.cross_entropy(tl_pred[1], tl_label[1], reduction="none") * tl_label[0]).mean()  # signal
        # junction loss
        junction_loss = F.binary_cross_entropy_with_logits(junction_pred, junction_label)  # presence
        # lane loss
        speed_loss = F.mse_loss(speed_pred, speed_label)

        # actor loss
        actor_loss = -torch.mean(action_dist.log_prob(action_label))  # acceleration & steering
        # critic loss
        critic_loss = F.mse_loss(td_estimate, td_target)

        return tl_loss, junction_loss, speed_loss, actor_loss, critic_loss

    def il_train(self, batch: List, label: List):
        """
        Train in il phase.

        :param batch: List of (s, a, r, s', d), where s means (obs, command)
        :param label: List of (tl_label, junction_label, speed_label)
        :return: List of losses.
        """
        assert self.il_phase

        self.model.reset()

        state, action, reward, next_state, done = list(zip(*batch))
        state = list(zip(*state))
        state[0] = torch.FloatTensor(state[0])
        obs, command = state
        reward = torch.FloatTensor(reward)
        next_state = list(zip(*next_state))
        next_state[0] = torch.FloatTensor((next_state[0]))
        done = torch.FloatTensor(done)

        tl_label, junction_label, speed_label = list(zip(*label))
        tl_label = list(zip(*tl_label))
        tl_label = [torch.FloatTensor(item) for item in tl_label]
        junction_label = torch.FloatTensor(junction_label)
        speed_label = torch.FloatTensor(speed_label)
        action_label = torch.FloatTensor(action)

        # pred
        obs = torch.tensor(obs)
        dynamin_embed, static_embed = self.model.encode(obs)
        tl_pred, junction_pred, speed_pred = self.model.auxiliary_decode(dynamin_embed, static_embed)
        action_dist = self.model.actor_decode(dynamin_embed, static_embed, command)
        td_estimate = self.compute_td_estimate(state, action_label)
        pred = [tl_pred, junction_pred, speed_pred, action_dist, td_estimate]

        # label
        td_target = self.compute_td_target(reward, next_state, done)
        label = [tl_label, junction_label, speed_label, action_label, td_target]

        loss = self.il_loss_func(pred, label)
        memo = [value.item() for value in loss]
        loss = torch.tensor(loss) * self.loss_weights
        self.optimizer_il.zero_grad()
        loss.backward()
        self.optimizer_il.step()
        memo.append(loss.item())

        return memo

    def rl_train(self, batch: List):
        """
        Train in rl phase.

        :param batch: List of (s, a, r, s', d), where s means (dynamic_embed, static_embed, command)
        :return:
        """
        assert not self.il_phase

        self.model.reset()

        state, action, reward, next_state, done = list(zip(*batch))
        state = list(zip(*state))
        state[0] = torch.FloatTensor(state[0])
        state[1] = torch.FloatTensor(state[1])
        dynamic_embed, static_embed, command = state
        reward = torch.FloatTensor(reward)
        next_state = list(zip(*next_state))
        next_state[0] = torch.FloatTensor((next_state[0]))
        next_state[1] = torch.FloatTensor((next_state[1]))
        done = torch.FloatTensor(done)

        # update critic
        td_target = self.compute_td_target(reward, next_state, done)
        td_estimate = self.compute_td_estimate(state, action)
        critic_loss = F.mse_loss(td_estimate, td_target)
        self.optimizer_rl_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_rl_critic.step()

        # update actor
        action_dist = self.model.actor_decode(dynamic_embed, static_embed, command)
        action = action_dist.rsample()
        entropy = -torch.mean(action_dist.log_prob(action))
        with torch.no_grad:
            value = self.model.critic_decode(dynamic_embed, static_embed, action)
        actor_loss = -(value + self.alpha * entropy)
        self.optimizer_rl_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_rl_actor.step()

        # synchronize target critic with current one
        self.sync_value_decoder()

        memo = [actor_loss.item(), critic_loss.item()]
        return memo

    def store(self, path, epoch, iteration):
        checkpoint = {
            "model": self.model.state_dict(),
            "optim_il": self.optimizer_il.state_dict(),
            "optim_rl_actor": self.optimizer_rl_actor.state_dict(),
            "optim_rl_critic": self.optimizer_rl_critic.state_dict(),
            "epoch": epoch,
            "iteration": iteration
        }
        torch.save(checkpoint, path)

    def restore(self, path):
        checkpoint = torch.load(path)

        self.model.load_state_dict(torch.load(checkpoint["model"]))
        self.critic = self.model.critic
        self.target_critic.load_state_dict(self.critic.state_dict)

        self.optimizer_il.load_state_dict(torch.load(checkpoint["optim_il"]))
        self.optimizer_rl_actor.load_state_dict(torch.load(checkpoint["optim_rl_actor"]))
        self.optimizer_rl_critic.load_state_dict(torch.load(checkpoint["optim_rl_critic"]))

        return checkpoint["epoch"], checkpoint["iteration"]
