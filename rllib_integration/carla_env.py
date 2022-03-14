#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


from __future__ import print_function

import gym

from rllib_integration.carla_core import CarlaCore


class CarlaEnv(gym.Env):
    """
    This is a carla environment, responsible of handling all the CARLA related steps of the training.
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

    def reset(self):
        self.experiment.reset()
        self.core.reset()
        self.hero = self.core.hero

        # Tick once and get the observations
        sensor_data = self.core.tick(None)
        observation, _ = self.experiment.get_observation(sensor_data)

        return observation

    def step(self, control):
        """Computes one tick of the environment in order to return the new observation,
        as well as the rewards"""
        sensor_data = self.core.tick(control)

        observation, info = self.experiment.get_observation(sensor_data)
        done = self.experiment.get_done_status(observation, self.core)
        reward = self.experiment.compute_reward(observation, self.core)

        return observation, reward, done, info
