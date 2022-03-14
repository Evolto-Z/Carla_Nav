import numpy as np
import carla
import torch
from torchvision import transforms as T

from algorithm.model import MyModel


class MyController():
    def __init__(self):
        self._vehicle = None
        self._world = None
        self._past_steering = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = MyModel().to(self.device)
        self._training = False
        self._exploration = False
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize([216, 288]),
            T.Normalize([0.471, 0.448, 0.408], [0.234, 0.239, 0.242])
        ])

    def reset(self, vehicle: carla.Vehicle):
        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        self._past_steering = self._vehicle.get_control().steer
        self._model.reset_cell()

    def set_training(self, value=True):
        self._training = value
        self._model.train(value)

    def set_exploration(self, value=True):
        self._exploration = value

    def get_model(self):
        return self._model

    def preprocess(self, obs):
        obs = self.transform(np.ascontiguousarray(obs))
        if obs.ndim == 3:
            obs = obs.unsqueeze(0)
        obs = obs.to(self.device)
        return obs

    def postprocess(self, acceleration, current_steering):
        control = carla.VehicleControl()

        if acceleration >= 0.0:
            control.throttle = acceleration
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = -acceleration

        # Steering regulation: changes cannot happen abruptly, can't steer too much.
        if current_steering > self._past_steering + 0.1:
            current_steering = self._past_steering + 0.1
        elif current_steering < self._past_steering - 0.1:
            current_steering = self._past_steering - 0.1
        control.steer = current_steering

        control.hand_brake = False
        control.manual_gear_shift = False

        self.past_steering = current_steering

        return control

    def run_step(self, obs, command):
        """
        Execute one step of control invoking both lateral and longitudinal.

        :param obs: the image from the camera
        :param command: the road option
        """
        obs = self.preprocess(obs)
        action_dist = self._model(obs, command)

        # explore and exploit
        if self._exploration:
            action = action_dist.rsample()
        else:
            action = action_dist.mean

        # batch size is 1
        action = action.squeeze(0)
        control = self.postprocess(action[0].item(), action[1].item())

        return control, action
