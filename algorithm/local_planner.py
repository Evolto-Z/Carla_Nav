from collections import deque
import carla
from tools.misc import draw_waypoints, get_speed, RoadOption
from algorithm.controller import MyController
from rllib_integration.helper import join_dicts

DEFAULT_CONFIG = {
    "target_speed": 20,  # km/h
    "follow_speed_limits": False,
    "sampling_radius": 2.0,  # meters
    "base_min_distance": 3.0,  # meters
    "max_throttle": 0.75,
    "max_brake": 0.3,
    "max_steering": 0.8
}


class LocalPlanner:
    def __init__(self, config={}):
        # Base parameters
        config = join_dicts(config, DEFAULT_CONFIG)
        self._target_speed = config["target_speed"]
        self._follow_speed_limits = config["follow_speed_limits"]
        self._sampling_radius = config["sampling_radius"]
        self._base_min_distance = config["base_min_distance"]
        self._max_throttle = config["max_throttle"]
        self._max_brake = config["max_brake"]
        self._max_steering = config["max_steering"]

        # initializing controller
        self._vehicle = None
        self._world = None
        self._map = None
        self._vehicle_controller = MyController()
        self._training = False

        # Compute the current vehicle waypoint
        self._waypoints_queue = deque()
        self.target_waypoint = None
        self.target_road_option = RoadOption.VOID

    def reset(self, vehicle: carla.Vehicle):
        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        self._map = self._world.get_map()
        self._vehicle_controller.reset(vehicle)

        # Compute the current vehicle waypoint
        self._waypoints_queue.clear()
        current_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        self.target_waypoint, self.target_road_option = (current_waypoint, RoadOption.LANEFOLLOW)
        self._waypoints_queue.append((self.target_waypoint, self.target_road_option))

    def set_training(self, value=True):
        self._training = value
        self._vehicle_controller.set_training(value)

    def set_exploration(self, value=True):
        self._vehicle_controller.set_exploration(value)

    def set_target_speed(self, speed):
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

    def get_model(self):
        """
        Return the deep learning model of the controller.

        :return:
        """
        return self._vehicle_controller.get_model()

    def set_global_plan(self, current_plan):
        """
        Adds a new plan to the local planner. A plan must be a list of [carla.Waypoint, RoadOption] pairs

        :param current_plan: list of (carla.Waypoint, RoadOption)
        :return:
        """
        self._waypoints_queue.clear()
        self._waypoints_queue.extend(current_plan)

    def run_step(self, obs, debug=False):
        """
        Execute one step of local planning.

        :param obs: the image from the camera
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

        # Get the target waypoint and move. Stop if no target waypoint
        if len(self._waypoints_queue) == 0:
            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 1.0
            control.hand_brake = False
            control.manual_gear_shift = False

            action = None
        else:
            self.target_waypoint, self.target_road_option = self._waypoints_queue[0]
            control, action = self._vehicle_controller.run_step(obs, self.target_road_option)

            if not self._training:
                control.throttle = min(control.throttle, self._max_throttle)
                control.brake = min(control.brake, self._max_brake)
                control.steer = max(min(control.steer, self._max_steering), -self._max_steering)

        if debug:
            draw_waypoints(self._vehicle.get_world(), [self.target_waypoint], 1.0)

        return control, action

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
        # the beginning of an intersection, therefore the
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
