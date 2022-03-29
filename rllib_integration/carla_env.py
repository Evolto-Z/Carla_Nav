#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


from __future__ import print_function

import gym

from rllib_integration.carla_core import CarlaCore
from tools.displayer import DisplayManager


class CarlaEnv(gym.Env):
    """
    This is a carla environment, responsible for handling all the CARLA related steps of the training.
    """

    def __init__(self, config, seed=None):
        """Initializes the environment"""
        self.config = config
        self.seed(seed)

        self.experiment = self.config["experiment"]["type"](self.config["experiment"])
        self.action_space = self.experiment.get_action_space()
        self.observation_space = self.experiment.get_observation_space()

        self.core = CarlaCore(self.config["carla"], seed=seed)
        self.core.setup(self.experiment.config)

        self.hero = None

        image_shape = self.observation_space["cameraRGB"].shape
        self.displayer = DisplayManager((image_shape[1], image_shape[0]))
        self.render_cache = None

    def reset(self):
        self.experiment.reset()
        self.core.reset()
        self.hero = self.core.hero

        # Tick once and get the observations
        sensor_data = self.core.tick(None)
        obs, _ = self.experiment.get_observation(sensor_data, self.hero)

        self.render_cache = obs["cameraRGB"]

        return obs

    def step(self, control):
        """Computes one tick of the environment in order to return the new observation,
        as well as the rewards"""
        self.displayer.step()

        sensor_data = self.core.tick(control)

        obs, info = self.experiment.get_observation(sensor_data, self.hero)
        done = self.experiment.get_done_status(obs, self.core)
        reward = self.experiment.compute_reward(obs, self.core)

        self.render_cache = obs["cameraRGB"]

        return obs, reward, done, info

    def render(self, mode="human"):
        self.displayer.render(self.render_cache)
