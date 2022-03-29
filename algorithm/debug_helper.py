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
    command = np.array([[RoadOption.STRAIGHT, RoadOption.LANEFOLLOW, RoadOption.LEFT],
                        [RoadOption.CHANGELANERIGHT, RoadOption.STRAIGHT, RoadOption.RIGHT]])
    env_state = torch.rand(2, 3, 3, 216, 288)
    ego_state = torch.rand(2, 3, 3)
    print("-------------- embed --------------")
    dynamic_embed, static_embed = model.encode(env_state)
    print("dynamic embed shape:", dynamic_embed.shape)
    print("static embed shape:", static_embed.shape)
    print("-------------- auxiliary --------------")
    tl_pred, junction_pred, speed_pred = model.auxiliary_decode(dynamic_embed, static_embed)
    print("tl affected:", tl_pred[0])
    print("tl signal:", tl_pred[1])
    print("junction present:", junction_pred)
    print("speed:", speed_pred)
    print("-------------- actor-critic --------------")
    action_dist = model.actor_decode(dynamic_embed, static_embed, command, ego_state)
    action = action_dist.rsample()
    print("sampled action:", action)
    print("log prob:", action_dist.log_prob(action))
    value_pred = model.critic_decode(dynamic_embed, static_embed, action, ego_state)
    print("return:", value_pred)
    print("-------------- model --------------")
    print("num of parameters:", get_parameter_number(model))


if __name__ == "__main__":
    debug_model()
