import carla

from algorithm.local_planner import LocalPlanner
from carla_agents.global_route_planner import GlobalRoutePlanner
from rllib_integration.helper import join_dicts

DEFAULT_CONFIG = {
    "emergency_brake": 1.0,
    "sampling_resolution": 2.0,  # meters, for global route planning
    "local_planner": {}
}


class MyPolicy:
    def __init__(self, config={}):
        # Base parameters
        config = join_dicts(DEFAULT_CONFIG, config)
        self._emergency_brake = config["emergency_brake"]

        self._vehicle = None
        self._world = None
        self._map = None

        # Initialize the planners
        self._global_planner = GlobalRoutePlanner(config["sampling_resolution"])
        self._local_planner = LocalPlanner(config["local_planner"])

    def reset(self, vehicle: carla.Vehicle):
        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        self._map = self._world.get_map()

        # Reinitialize the planners
        self._global_planner.reset(self._map)
        self._local_planner.reset(self._vehicle)

    def get_model(self):
        return self._local_planner.get_model()

    def set_exploration(self, value=True):
        self._local_planner.set_exploration(value)

    def set_training(self, value=True):
        self._local_planner.set_training(value)

    def set_target_speed(self, speed):
        """
        Changes the target speed of the agent

        :param speed: target speed in Km/h
        :return:
        """
        self._local_planner.set_target_speed(speed)

    def set_follow_speed_limits(self, value=True):
        """
        If active, the agent will dynamically change the target speed according to the speed limits

        :param value: whether or not to activate this behavior
        :return:
        """
        self._local_planner.set_follow_speed_limits(value)

    def get_local_planner(self):
        """Get method for protected member local planner"""
        return self._local_planner

    def get_global_planner(self):
        """Get method for protected member local planner"""
        return self._global_planner

    def set_destination(self, end_location, start_location=None):
        """
        This method creates a list of waypoints between a starting and ending location,
        based on the route returned by the global router, and adds it to the local planner.
        If no starting location is passed, the vehicle local planner's target location is chosen,
        which corresponds (by default), to a location about 5 meters in front of the vehicle.

        :param end_location: (carla.Location) final location of the route
        :param start_location: (carla.Location) starting location of the route
        :return:
        """
        if not start_location:
            start_location = self._local_planner.target_waypoint.transform.location
            clean_queue = True
        else:
            start_location = self._vehicle.get_location()
            clean_queue = False

        start_waypoint = self._map.get_waypoint(start_location)
        end_waypoint = self._map.get_waypoint(end_location)

        route_trace = self.trace_route(start_waypoint, end_waypoint)
        self._local_planner.set_global_plan(route_trace)

    def set_global_plan(self, plan, stop_waypoint_creation=True, clean_queue=True):
        """
        Adds a specific plan to the agent.

        :param plan: list of [carla.Waypoint, RoadOption] representing the route to be followed
        :param stop_waypoint_creation: stops the automatic random creation of waypoints
        :param clean_queue: resets the current agent's plan
        """
        self._local_planner.set_global_plan(
            plan,
            stop_waypoint_creation=stop_waypoint_creation,
            clean_queue=clean_queue
        )

    def trace_route(self, start_waypoint, end_waypoint):
        """
        Calculates the shortest route between a starting and ending waypoint.

        :param start_waypoint: (carla.Waypoint) initial waypoint
        :param end_waypoint: (carla.Waypoint) final waypoint
        :return:
        """
        start_location = start_waypoint.transform.location
        end_location = end_waypoint.transform.location
        return self._global_planner.trace_route(start_location, end_location)

    def add_emergency_stop(self, control):
        """
        Overwrites the throttle a brake values of a control to perform an emergency stop.
        The steering is kept the same to avoid going out of the lane when stopping during turns

        :param control: (carl.VehicleControl) control to be modified
        """
        control.throttle = 0.0
        control.brake = self._emergency_brake
        control.hand_brake = False
        return control

    def run_step(self, obs):
        """Execute one step of navigation."""
        control, action = self._local_planner.run_step(obs)

        return control, action

    def done(self):
        """Check whether the agent has reached its destination."""
        return self._local_planner.done()
