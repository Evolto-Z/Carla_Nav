import torch
from algorithm.model import *
from tools.misc import RoadOption


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def debug_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyModel().to(device)
    command = RoadOption.STRAIGHT
    obs = torch.rand(8, 3, 216, 288)
    dynamic_embed, static_embed = model.encode(obs)
    print("-------------- embed --------------")
    print(dynamic_embed.shape)
    print(static_embed.shape)
    tl_pred, junction_pred, lane_pred = model.auxiliary_decode(dynamic_embed, static_embed)
    print("-------------- auxiliary --------------")
    print(tl_pred[0])
    print(tl_pred[1])
    print(junction_pred)
    print(lane_pred[0])
    print(lane_pred[1])
    print("-------------- actor-critic --------------")
    action_dist = model.actor_decode(dynamic_embed, static_embed, command)
    action = action_dist.rsample()
    print(action)
    print(action_dist.log_prob(action))
    value_pred = model.critic_decode(dynamic_embed, static_embed, action)
    print(value_pred)
    print("-------------- model --------------")
    print(get_parameter_number(model))


if __name__ == "__main__":
    debug_model()
