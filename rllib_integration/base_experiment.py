#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import carla
from rllib_integration.helper import join_dicts

BASE_EXPERIMENT_CONFIG = {
    "hero": {
        "blueprint": "vehicle.lincoln.mkz_2017",
        "sensors": {  # Go to sensors/factory.py to check all the available sensors
            # "sensor_name1": {
            #     "type": blueprint,
            #     "attribute1": attribute_value1,
            #     "attribute2": attribute_value2
            # }
            # "sensor_name2": {
            #     "type": blueprint,
            #     "attribute_name1": attribute_value1,
            #     "attribute_name2": attribute_value2
            # }
        }
    },
    "background_activity": {
        "n_vehicles": 0,
        "n_walkers": 0,
        "tm_hybrid_mode": True
    },
    "weather": ["ClearNight", "ClearNoon", "ClearSunset", "CloudyNight", "CloudyNoon", "CloudySunset",
                "Default", "HardRainNight", "HardRainNoon", "HardRainSunset", "MidRainSunset", "MidRainyNight",
                "MidRainyNoon", "SoftRainNight", "SoftRainNoon", "SoftRainSunset", "WetCloudyNight",
                "WetCloudyNoon", "WetCloudySunset", "WetNight", "WetNoon", "WetSunset"]
}


class BaseExperiment(object):
    def __init__(self, config):
        self.config = join_dicts(BASE_EXPERIMENT_CONFIG, config)

    def reset(self):
        """Called at the beginning and each time the simulation is reset"""
        pass

    def get_action_space(self):
        """Returns the action space"""
        raise NotImplementedError

    def get_observation_space(self):
        """Returns the observation space"""
        raise NotImplementedError

    def get_observation(self, sensor_data, hero: carla.Vehicle):
        """
        Function to do the postprocessing of the sensor data and the ego state of the hero.

        :param sensor_data: dictionary {sensor_name: sensor_data}
        :param hero: carla.Vehicle

        Should return a tuple or list with two items, the processed observations,
        as well as a variable with additional information about such observation.
        The information variable can be empty
        """
        return NotImplementedError

    def get_done_status(self, observation, core):
        """Returns whether the experiment has to end"""
        return NotImplementedError

    def compute_reward(self, observation, core):
        """Computes the reward"""
        return NotImplementedError
