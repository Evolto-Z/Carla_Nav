from rllib_integration.base_experiment import BaseExperiment
import carla
from gym.spaces import Box, Dict
import numpy as np
import math


class MyExperiment(BaseExperiment):
    def __init__(self, config={}):
        super().__init__(config)  # Creates a self.config with the experiment configuration

        self.allowed_types = [carla.LaneType.Driving, carla.LaneType.Parking]
        self.last_heading_deviation = 0
        self.last_action = None

    def reset(self):
        """Called at the beginning and each time the simulation is reset"""

        # hero variables
        self.last_location = None
        self.last_velocity = 0

        self.last_heading_deviation = 0

    def get_action_space(self):
        return Dict({
            "throttle": Box(0, 1, (1,)),
            "steer": Box(-1, 1, (1,)),
            "brake": Box(0, 1, (1,))
        })

    def get_observation_space(self):
        num_of_channels = 3
        image_space = Box(
            low=0.0,
            high=255.0,
            shape=(
                self.config["hero"]["sensors"]["camera_rgb"]["image_size_y"],
                self.config["hero"]["sensors"]["camera_rgb"]["image_size_x"],
                num_of_channels
            ),
            dtype=np.uint8,
        )
        return image_space

    def compute_action(self, action):
        """Given the action, returns a carla.VehicleControl() which will be applied to the hero"""
        action = carla.VehicleControl()
        action.throttle = action["throttle"]
        action.steer = action["steer"]
        action.brake = action["brake"]
        action.reverse = False
        action.hand_brake = False

        self.last_action = action

        return action

    def get_observation(self, sensor_data):
        """Function to do all the post processing of observations (sensor data).

        :param sensor_data: dictionary {sensor_name: sensor_data}

        Should return a tuple or list with two items, the processed observations,
        as well as a variable with additional information about such observation.
        The information variable can be empty
        """
        image = sensor_data["camera_rgb"][1]
        # image = (image.astype(np.float32) - 128) / 128

        return image, {}

    def get_speed(self, hero):
        """Computes the speed of the hero vehicle in Km/h"""
        vel = hero.get_velocity()
        return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

    def get_done_status(self, observation, core):
        """Returns whether or not the experiment has to end"""
        return False

    def compute_reward(self, observation, core):
        """Computes the reward"""

        def unit_vector(vector):
            return vector / np.linalg.norm(vector)

        def compute_angle(u, v):
            return -math.atan2(u[0] * v[1] - u[1] * v[0], u[0] * v[0] + u[1] * v[1])

        def find_current_waypoint(map_, hero):
            return map_.get_waypoint(hero.get_location(), project_to_road=False, lane_type=carla.LaneType.Any)

        def inside_lane(waypoint, allowed_types):
            if waypoint is not None:
                return waypoint.lane_type in allowed_types
            return False

        world = core.world
        hero = core.hero
        map_ = core.map

        # Hero-related variables
        hero_location = hero.get_location()
        hero_velocity = self.get_speed(hero)
        hero_heading = hero.get_transform().get_forward_vector()
        hero_heading = [hero_heading.x, hero_heading.y]

        # Initialize last location
        if self.last_location == None:
            self.last_location = hero_location

        # Compute deltas
        delta_distance = float(np.sqrt(np.square(hero_location.x - self.last_location.x) + \
                                       np.square(hero_location.y - self.last_location.y)))
        delta_velocity = hero_velocity - self.last_velocity

        # Update variables
        self.last_location = hero_location
        self.last_velocity = hero_velocity

        # Reward if going forward
        reward = delta_distance

        # Reward if going faster than last step
        if hero_velocity < 20.0:
            reward += 0.05 * delta_velocity

        # La duracion de estas infracciones deberia ser 2 segundos?
        # Penalize if not inside the lane
        closest_waypoint = map_.get_waypoint(
            hero_location,
            project_to_road=False,
            lane_type=carla.LaneType.Any
        )
        if closest_waypoint is None or closest_waypoint.lane_type not in self.allowed_types:
            reward += -0.5
            self.last_heading_deviation = math.pi
        else:
            if not closest_waypoint.is_junction:
                wp_heading = closest_waypoint.transform.get_forward_vector()
                wp_heading = [wp_heading.x, wp_heading.y]
                angle = compute_angle(hero_heading, wp_heading)
                self.last_heading_deviation = abs(angle)

                if np.dot(hero_heading, wp_heading) < 0:
                    # We are going in the wrong direction
                    reward += -0.5

                else:
                    if abs(math.sin(angle)) > 0.4:
                        if self.last_action == None:
                            self.last_action = carla.VehicleControl()

                        if self.last_action.steer * math.sin(angle) >= 0:
                            reward -= 0.05
            else:
                self.last_heading_deviation = 0

        # if self.done_falling:
        #     reward += -40
        # if self.done_time_idle:
        #     print("Done idle")
        #     reward += -100
        # if self.done_time_episode:
        #     print("Done max time")
        #     reward += 100

        return reward