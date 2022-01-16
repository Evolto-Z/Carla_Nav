import torch
from torchvision import models
from algorithm.model import ConvEncoder, SegDecoder, TrafficLightDecoder


encoder = ConvEncoder()
input = torch.rand(1, 4, 3, 216, 288)
embed = encoder(input)
print(embed.shape)
decoder = SegDecoder(3)
seg = decoder(embed)
print(seg.shape)
decoder = TrafficLightDecoder()
presence, signal, distance = decoder(embed)
print(presence.shape)
print(signal.shape)
print(distance.shape)
