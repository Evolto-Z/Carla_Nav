# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" This module contains a local planner to perform low-level waypoint following based on PID controllers. """

from collections import deque
from rllib_integration.helper import join_dicts
import carla
from carla_agents.controller import VehiclePIDController
from tools.misc import draw_waypoints, get_speed, RoadOption

DEFAULT_CONFIG = {
    "target_speed": 20,  # km/h
    "follow_speed_limits": False,
    "sampling_radius": 2.0,  # meters
    "base_min_distance": 3.0,  # meters
    "max_throttle": 0.75,
    "max_brake": 0.3,
    "max_steering": 0.8,
    "emergency_brake": 1.0,
    "lateral_pid": [1.95, 0.05, 0.2],
    "longitudinal_pid": [1.0, 0.05, 0],
    "offset": 0
}


class LocalPlanner(object):
    """
    LocalPlanner implements the basic behavior of following a
    trajectory of waypoints that is generated on-the-fly.

    The low-level motion of the vehicle is computed by using two PID controllers,
    one is used for the lateral control and the other for the longitudinal control (cruise speed).
    """
    def __init__(self, config={}):
        self._vehicle = None
        self._world = None
        self._map = None

        # Base parameters
        config = join_dicts(config, DEFAULT_CONFIG)
        self._target_speed = config["target_speed"]
        self._follow_speed_limits = config["follow_speed_limits"]
        self._sampling_radius = config["sampling_radius"]
        self._base_min_distance = config["base_min_distance"]
        self._max_throt = config["max_throttle"]
        self._max_brake = config["max_brake"]
        self._max_steer = config["max_steering"]
        self._emergency_brake = config["emergency_brake"]
        pid = config["lateral_pid"]
        self._args_lateral_dict = {'K_P': pid[0], 'K_I': pid[1], 'K_D': pid[2]}
        pid = config["longitudinal_pid"]
        self._args_longitudinal_dict = {'K_P': pid[0], 'K_I': pid[1], 'K_D': pid[2]}
        self._offset = config["offset"]

        # initializing controller
        self._vehicle_controller = VehiclePIDController(args_lateral=self._args_lateral_dict,
                                                        args_longitudinal=self._args_longitudinal_dict,
                                                        offset=self._offset,
                                                        max_throttle=self._max_throt,
                                                        max_brake=self._max_brake,
                                                        max_steering=self._max_steer)

        # Compute the current vehicle waypoint
        self._waypoints_queue = deque()
        self.target_waypoint = None
        self.target_road_option = RoadOption.VOID

    def reset(self, vehicle: carla.Vehicle):
        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        self._map = self._world.get_map()

        self._vehicle_controller.reset(vehicle)
        self._vehicle_controller.set_delta_time(self._world.get_settings().fixed_delta_seconds)

        # Compute the current vehicle waypoint
        self._waypoints_queue.clear()
        current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        self.target_waypoint, self.target_road_option = (current_waypoint, RoadOption.LANEFOLLOW)
        self._waypoints_queue.append((self.target_waypoint, self.target_road_option))

    def set_speed(self, speed: float):
        """
        Changes the target speed

        :param speed: new target speed in Km/h
        :return:
        """
        if self._follow_speed_limits:
            print("WARNING: The max speed is currently set to follow the speed limits. "
                  "Use 'follow_speed_limits' to deactivate this")
        self._target_speed = speed

    def set_follow_speed_limits(self, value=True):
        """
        Activates a flag that makes the max speed dynamically vary according to the spped limits

        :param value: bool
        :return:
        """
        self._follow_speed_limits = value

    def set_global_plan(self, current_plan):
        """
        Adds a new plan to the local planner. A plan must be a list of [carla.Waypoint, RoadOption] pairs

        :param current_plan: list of (carla.Waypoint, RoadOption)
        :return:
        """
        self._waypoints_queue.clear()
        self._waypoints_queue.extend(current_plan)

    def add_emergency_stop(self, control):
        """
        Overwrites the throttle a brake values of a control to perform an emergency stop.
        The steering is kept the same to avoid going out of the lane when stopping during turns

        :param control: (carla.VehicleControl) control to be modified
        :return:
        """
        control.throttle = 0.0
        control.brake = self._emergency_brake
        control.hand_brake = False
        return control

    def run_step(self, debug=False):
        """
        Execute one step of local planning which involves running the longitudinal and lateral PID controllers to
        follow the waypoints trajectory.

        :param debug: boolean flag to activate waypoints debugging
        :return: control to be applied
        """
        if self._follow_speed_limits:
            self._target_speed = self._vehicle.get_speed_limit()

        # Purge the queue of obsolete waypoints
        veh_location = self._vehicle.get_location()
        vehicle_speed = get_speed(self._vehicle) / 3.6
        min_distance = self._base_min_distance + 0.5 * vehicle_speed
        num_waypoint_removed = 0
        for waypoint, _ in self._waypoints_queue:
            if len(self._waypoints_queue) - num_waypoint_removed == 1:
                min_distance = 1  # Don't remove the last waypoint until very close by
            if veh_location.distance(waypoint.transform.location) < min_distance:
                num_waypoint_removed += 1
            else:
                break
        if num_waypoint_removed > 0:
            for _ in range(num_waypoint_removed):
                self._waypoints_queue.popleft()

        # Get the target waypoint and move using the PID controllers. Stop if no target waypoint
        if len(self._waypoints_queue) == 0:
            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 1.0
            control.hand_brake = False
            control.manual_gear_shift = False
        else:
            self.target_waypoint, self.target_road_option = self._waypoints_queue[0]
            control = self._vehicle_controller.run_step(self._target_speed, self.target_waypoint)

        if debug:
            draw_waypoints(self._vehicle.get_world(), [self.target_waypoint], 1.0)

        return control

    def get_incoming_waypoint_and_direction(self, steps=3):
        """
        Returns direction and waypoint at a distance ahead defined by the user.

            :param steps: number of steps to get the incoming waypoint.
        """
        if len(self._waypoints_queue) > steps:
            return self._waypoints_queue[steps]

        else:
            try:
                wpt, direction = self._waypoints_queue[-1]
                return wpt, direction
            except IndexError:
                return None, RoadOption.VOID

    def get_plan(self):
        """Returns the current plan of the local planner"""
        return self._waypoints_queue

    def done(self):
        """
        Returns whether or not the planner has finished

        :return: boolean
        """
        return len(self._waypoints_queue) == 0


def _retrieve_options(list_waypoints, current_waypoint):
    """
    Compute the type of connection between the current active waypoint and the multiple waypoints present in
    list_waypoints. The result is encoded as a list of RoadOption enums.

    :param list_waypoints: list with the possible target waypoints in case of multiple options
    :param current_waypoint: current active waypoint
    :return: list of RoadOption enums representing the type of connection from the active waypoint to each
             candidate in list_waypoints
    """
    options = []
    for next_waypoint in list_waypoints:
        # this is needed because something we are linking to
        # the beggining of an intersection, therefore the
        # variation in angle is small
        next_next_waypoint = next_waypoint.next(3.0)[0]
        link = _compute_connection(current_waypoint, next_next_waypoint)
        options.append(link)

    return options


def _compute_connection(current_waypoint, next_waypoint, threshold=35):
    """
    Compute the type of topological connection between an active waypoint (current_waypoint) and a target waypoint
    (next_waypoint).

    :param current_waypoint: active waypoint
    :param next_waypoint: target waypoint
    :return: the type of topological connection encoded as a RoadOption enum:
             RoadOption.STRAIGHT
             RoadOption.LEFT
             RoadOption.RIGHT
    """
    n = next_waypoint.transform.rotation.yaw
    n = n % 360.0

    c = current_waypoint.transform.rotation.yaw
    c = c % 360.0

    diff_angle = (n - c) % 180.0
    if diff_angle < threshold or diff_angle > (180 - threshold):
        return RoadOption.STRAIGHT
    elif diff_angle > 90.0:
        return RoadOption.LEFT
    else:
        return RoadOption.RIGHT
