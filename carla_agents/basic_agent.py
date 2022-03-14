"""
This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles. The agent also responds to traffic lights.
It can also make use of the global route planner to follow a specified route
"""

import carla
from shapely.geometry import Polygon
from rllib_integration.helper import join_dicts
from carla_agents.local_planner import LocalPlanner
from carla_agents.global_route_planner import GlobalRoutePlanner
from tools.misc import get_speed, is_within_distance, get_trafficlight_trigger_location, compute_distance

DEFAULT_CONFIG = {
    "base_tlight_threshold": 5.0,  # meters
    "base_vehicle_threshold": 5.0,  # meters
    "emergency_brake": 1.0,
    "ignore_traffic_lights": False,
    "ignore_stop_signs": False,
    "ignore_vehicles": False,
    "sampling_resolution": 2.0,  # meters, for global route planning
    "local_planner": {}
}


class BasicAgent(object):
    """
    BasicAgent implements an agent that navigates the scene.
    This agent respects traffic lights and other vehicles, but ignores stop signs.
    It has several functions available to specify the route that the agent must follow,
    as well as to change its parameters in case a different driving mode is desired.
    """
    def __init__(self, config={}):
        # Base parameters
        config = join_dicts(config, DEFAULT_CONFIG)
        self._base_tlight_threshold = config['base_tlight_threshold']
        self._base_vehicle_threshold = config['base_vehicle_threshold']
        self._emergency_brake = config["emergency_brake"]
        self._ignore_traffic_lights = config["ignore_traffic_lights"]
        self._ignore_stop_signs = config["ignore_stop_signs"]
        self._ignore_vehicles = config["ignore_vehicles"]

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

        self._global_planner.reset(self._map)
        self._local_planner.reset(vehicle)

    def set_target_speed(self, speed):
        """
        Changes the target speed of the agent

        :param speed: (float) target speed in Km/h
        """
        self._local_planner.set_speed(speed)

    def set_follow_speed_limits(self, value=True):
        """
        If active, the agent will dynamically change the target speed according to the speed limits

        :param value: (bool) whether or not to activate this behavior
        """
        self._local_planner.set_follow_speed_limits(value)

    def get_local_planner(self):
        """Get method for protected member local planner"""
        return self._local_planner

    def get_global_planner(self):
        """Get method for protected member local planner"""
        return self._global_planner

    def get_target_road_option(self):
        return self._local_planner.target_road_option

    def get_target_waypoint(self):
        return self._local_planner.target_waypoint

    def set_destination(self, end_location: carla.Location):
        """
        This method creates a list of waypoints between a starting and ending location,
        based on the route returned by the global router, and adds it to the local planner.

        :param end_location: (carla.Location) final location of the route
        """
        start_location = self._vehicle.get_location()
        start_waypoint = self._map.get_waypoint(start_location)
        end_waypoint = self._map.get_waypoint(end_location)

        route_trace = self.trace_route(start_waypoint, end_waypoint)
        self._local_planner.set_global_plan(route_trace)

    def trace_route(self, start_waypoint: carla.Waypoint, end_waypoint: carla.Waypoint):
        """
        Calculates the shortest route between a starting and ending waypoint.

        :param start_waypoint: (carla.Waypoint) initial waypoint
        :param end_waypoint: (carla.Waypoint) final waypoint
        """
        start_location = start_waypoint.transform.location
        end_location = end_waypoint.transform.location
        return self._global_planner.trace_route(start_location, end_location)

    def set_global_plan(self, plan):
        """
        Adds a specific plan to the agent.

        :param plan: list of [carla.Waypoint, RoadOption] representing the route to be followed
        """
        self._local_planner.set_global_plan(plan)

    def run_step(self):
        """Execute one step of navigation."""
        hazard_detected = False

        # Retrieve all relevant actors
        actor_list = self._world.get_actors()
        vehicle_list = actor_list.filter("*vehicle*")
        lights_list = actor_list.filter("*traffic_light*")

        vehicle_speed = get_speed(self._vehicle) / 3.6

        # Check for possible vehicle obstacles
        max_vehicle_distance = self._base_vehicle_threshold + vehicle_speed
        affected_by_vehicle, _, _ = self._vehicle_obstacle_detected(vehicle_list, max_vehicle_distance)
        if affected_by_vehicle:
            hazard_detected = True

        # Check if the vehicle is affected by a red traffic light
        max_tlight_distance = self._base_tlight_threshold + vehicle_speed
        affected_by_tlight, tlight_state = self._affected_by_traffic_light(lights_list, max_tlight_distance)
        if affected_by_tlight and tlight_state.state == carla.TrafficLightState.Red:
            hazard_detected = True

        control = self._local_planner.run_step()

        if hazard_detected:
            control = self.add_emergency_stop(control)

        return control

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

    def done(self):
        """Check whether the agent has reached its destination."""
        return self._local_planner.done()

    def ignore_traffic_lights(self, active=True):
        """(De)activates the checks for traffic lights"""
        self._ignore_traffic_lights = active

    def ignore_stop_signs(self, active=True):
        """(De)activates the checks for stop signs"""
        self._ignore_stop_signs = active

    def ignore_vehicles(self, active=True):
        """(De)activates the checks for stop signs"""
        self._ignore_vehicles = active

    def _affected_by_traffic_light(self, lights_list=None, max_distance=None):
        """
        Method to check if there is a red light affecting the vehicle.

        :param lights_list (list of carla.TrafficLight): list containing TrafficLight objects.
            If None, all traffic lights in the scene are used
        :param max_distance (float): max distance for traffic lights to be considered relevant.
            If None, the base threshold value is used
        """
        if self._ignore_traffic_lights:
            return False, None

        if not lights_list:
            lights_list = self._world.get_actors().filter("*traffic_light*")

        if not max_distance:
            max_distance = self._base_tlight_threshold

        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

        for traffic_light in lights_list:
            object_location = get_trafficlight_trigger_location(traffic_light)
            object_waypoint = self._map.get_waypoint(object_location)

            if object_waypoint.road_id != ego_vehicle_waypoint.road_id:
                continue

            ve_dir = ego_vehicle_waypoint.transform.get_forward_vector()
            wp_dir = object_waypoint.transform.get_forward_vector()
            dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z
            if dot_ve_wp < 0:
                continue

            state = traffic_light.state
            if state == carla.TrafficLightState.Off or state == carla.TrafficLightState.Unknown:
                continue

            if is_within_distance(object_waypoint.transform, self._vehicle.get_transform(), max_distance, [0, 90]):
                return True, state

        return False, None

    def _vehicle_obstacle_detected(self,
                                   vehicle_list=None,
                                   max_distance=None,
                                   up_angle_th=90, low_angle_th=0,
                                   lane_offset=0):
        """
        Method to check if there is a vehicle in front of the agent blocking its path.

        :param vehicle_list (list of carla.Vehicle): list contatining vehicle objects.
            If None, all vehicle in the scene are used
        :param max_distance: max freespace to check for obstacles.
            If None, the base threshold value is used
        :return: (boolean, carla.Vehicle, float)
        """
        if self._ignore_vehicles:
            return False, None, -1

        if not vehicle_list:
            vehicle_list = self._world.get_actors().filter("*vehicle*")

        if not max_distance:
            max_distance = self._base_vehicle_threshold

        ego_transform = self._vehicle.get_transform()
        ego_wpt = self._map.get_waypoint(self._vehicle.get_location())

        # Get the right offset
        if ego_wpt.lane_id < 0 and lane_offset != 0:
            lane_offset *= -1

        # Get the transform of the front of the ego
        ego_forward_vector = ego_transform.get_forward_vector()
        ego_extent = self._vehicle.bounding_box.extent.x
        ego_front_transform = ego_transform
        ego_front_transform.location += ego_extent * ego_forward_vector

        # Get the ego polygon
        route_bb = []
        ego_location = ego_transform.location
        extent_y = self._vehicle.bounding_box.extent.y
        r_vec = ego_transform.get_right_vector()
        p1 = ego_location + extent_y * r_vec
        p2 = ego_location - extent_y * r_vec
        route_bb.append([p1.x, p1.y, p1.z])
        route_bb.append([p2.x, p2.y, p2.z])
        for wp, _ in self._local_planner.get_plan():
            if ego_location.distance(wp.transform.location) > max_distance:
                break
            r_vec = wp.transform.get_right_vector()
            p1 = wp.transform.location + extent_y * r_vec
            p2 = wp.transform.location - extent_y * r_vec
            route_bb.append([p1.x, p1.y, p1.z])
            route_bb.append([p2.x, p2.y, p2.z])
        ego_polygon = Polygon(route_bb) if len(route_bb) >= 3 else None

        for target_vehicle in vehicle_list:
            if target_vehicle.id == self._vehicle.id:
                continue

            target_transform = target_vehicle.get_transform()
            target_wpt = self._map.get_waypoint(target_transform.location, lane_type=carla.LaneType.Any)

            # Simplified version for outside junctions
            if not ego_wpt.is_junction or not target_wpt.is_junction:

                if target_wpt.road_id != ego_wpt.road_id or target_wpt.lane_id != ego_wpt.lane_id + lane_offset:
                    next_wpt = self._local_planner.get_incoming_waypoint_and_direction(steps=3)[0]
                    if not next_wpt:
                        continue
                    if target_wpt.road_id != next_wpt.road_id or target_wpt.lane_id != next_wpt.lane_id + lane_offset:
                        continue

                target_forward_vector = target_transform.get_forward_vector()
                target_extent = target_vehicle.bounding_box.extent.x
                target_rear_transform = target_transform
                target_rear_transform.location -= target_extent * target_forward_vector

                if is_within_distance(target_rear_transform, ego_front_transform,
                                      max_distance,
                                      [low_angle_th, up_angle_th]):
                    return True, target_vehicle, compute_distance(target_transform.location, ego_transform.location)

            # Waypoints aren't reliable, check the proximity of the vehicle to the route
            elif ego_polygon is not None:
                # Compare the two polygons
                if target_vehicle.id == self._vehicle.id:
                    continue
                if ego_location.distance(target_vehicle.get_location()) > max_distance:
                    continue

                target_bb = target_vehicle.bounding_box
                target_vertices = target_bb.get_world_vertices(target_vehicle.get_transform())
                target_list = [[v.x, v.y, v.z] for v in target_vertices]
                target_polygon = Polygon(target_list)

                if ego_polygon.intersects(target_polygon):
                    return True, target_vehicle, compute_distance(target_vehicle.get_location(), ego_location)

        return False, None, -1
