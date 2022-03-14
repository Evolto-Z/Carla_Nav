#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import os
import random
import signal
import subprocess
import time
import psutil
import logging
import platform

import carla

from rllib_integration.sensors.sensor_interface import SensorInterface
from rllib_integration.sensors.factory import SensorFactory
from rllib_integration.helper import join_dicts

BASE_CORE_CONFIG = {
    "host": "localhost",  # Client host
    "port": 2000,
    "timeout": 10.0,  # Timeout of the client
    "timestep": 0.05,  # Time step of the simulation
    "retries_on_error": 10,  # Number of tries to connect to the client
    "resolution_x": 600,  # Width of the server spectator camera
    "resolution_y": 600,  # Height of the server spectator camera
    "quality_level": "Low",  # Quality level of the simulation. Can be 'Low', 'High', 'Epic'
    "enable_map_assets": True,  # enable / disable all town assets except for the road
    "enable_rendering": True  # enable / disable camera images
}

SYSTEM = platform.system()


def is_used(port):
    """Checks whether or not a port is used"""
    return port in [conn.laddr.port for conn in psutil.net_connections()]


def kill_all_servers():
    """Kill all PIDs that start with Carla"""
    processes = [p for p in psutil.process_iter() if "carla" in p.name().lower()]
    for process in processes:
        if SYSTEM == "Windows":
            os.kill(process.pid, signal.SIGINT)
        else:
            os.kill(process.pid, signal.SIGKILL)


class CarlaCore:
    """
    Class responsible of handling all the different CARLA functionalities, such as server-client connecting,
    actor spawning and getting the sensors data.
    """
    def __init__(self, config={}, seed=None):
        """Initialize the server and client"""

        self.config = join_dicts(BASE_CORE_CONFIG, config)
        self.seed = seed
        random.seed(seed)

        self.client = None
        self.traffic_manager = None
        self.world = None
        self.map = None
        self.hero = None

        self.host = self.config["host"]
        self.server_port = self.config["port"]
        while is_used(self.server_port):
            self.server_port += 1
        self.tm_port = self.server_port + 1
        while is_used(self.tm_port):
            self.tm_port += 1

        self.available_maps = []
        self.available_weathers = []

        self.n_vehicles = 0
        self.n_walkers = 0
        self.tm_hybrid_mode = True
        self.tm_actors = []
        self.tm_controllers = []

        self.hero_blueprint = None
        self.sensors_dict = {}
        self.sensor_interface = SensorInterface()

        self.init_server()
        self.connect_client()

    def init_server(self):
        server_command = [
            "{}/CarlaUE4.exe".format(os.environ["CARLA_ROOT"]),
            "-windowed",
            "-ResX={}".format(self.config["resolution_x"]),
            "-ResY={}".format(self.config["resolution_y"]),
            "--carla-rpc-port={}".format(self.server_port),
            "-quality-level={}".format(self.config["quality_level"])
        ]
        server_command_text = " ".join(map(str, server_command))
        print(server_command_text)

        if SYSTEM == "Windows":
            subprocess.Popen(
                server_command_text,
                shell=True,
                stdout=open(os.devnull, "w"),
            )
        else:
            subprocess.Popen(
                server_command_text,
                shell=True,
                preexec_fn=os.setsid,
                stdout=open(os.devnull, "w"),
            )

    def connect_client(self):
        """Connect to the client"""

        for i in range(self.config["retries_on_error"]):
            try:
                self.client = carla.Client(self.host, self.server_port)
                self.client.set_timeout(self.config["timeout"])

                self.world = self.client.get_world()
                settings = self.world.get_settings()
                settings.no_rendering_mode = not self.config["enable_rendering"]
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = self.config["timestep"]
                self.world.apply_settings(settings)
                self.world.tick()

                return  # pushed to the call stack, not return immediately

            except Exception as e:
                print(" Waiting for server to be ready: {}, attempt {} of {}".format(
                    e, i + 1, self.config["retries_on_error"]))
                time.sleep(3)

        raise Exception("Cannot connect to server. "
                        "Try increasing 'timeout' or 'retries_on_error' at the carla configuration")

    def setup(self, experiment_config):
        # Get available maps and weathers
        self.available_maps = self.client.get_available_maps()
        self.available_weathers = experiment_config["weather"]
        if not isinstance(self.available_weathers, list):
            self.available_weathers = [self.available_weathers]

        # Get parameters of the traffic manager
        self.n_vehicles = experiment_config["background_activity"]["n_vehicles"]
        self.n_walkers = experiment_config["background_activity"]["n_walkers"]
        self.tm_hybrid_mode = experiment_config["background_activity"]["tm_hybrid_mode"]

        # Get parameters of the hero
        hero_config = experiment_config["hero"]
        self.hero_blueprint = self.world.get_blueprint_library().find(hero_config['blueprint'])
        self.hero_blueprint.set_attribute("role_name", "hero")
        self.sensors_dict = hero_config["sensors"]

    def reset(self):
        # Manually clean, or there will be errors
        self.sensor_interface.destroy()
        for controller in self.tm_controllers:
            controller.stop()
        for actor in self.tm_actors:
            actor.destroy()
        self.world.tick()

        # Load the world and get the map
        map_name = random.choice(self.available_maps)
        print("Map Name: {}".format(map_name))
        self.world = self.client.load_world(
            map_name=map_name,
            reset_settings=False)
        self.world.tick()  # When changing to a new world, we should get the map after a tick.
        self.map = self.world.get_map()

        # Choose the weather of the simulation
        weather = self.available_weathers
        if isinstance(weather, list):
            weather = random.choice(weather)
        print("Weather: {}".format(weather))
        weather_param = getattr(carla.WeatherParameters, weather)
        self.world.set_weather(weather_param)

        # Set up the traffic manager
        self.traffic_manager = self.client.get_trafficmanager(self.tm_port)
        print("Traffic manager connected to port " + str(self.tm_port))
        self.traffic_manager.set_hybrid_physics_mode(self.tm_hybrid_mode)
        if self.seed is not None:
            self.traffic_manager.set_random_device_seed(self.seed)

        # Spawn the background activity
        n_vehicles = self.n_vehicles
        if isinstance(n_vehicles, list):
            n_vehicles = random.randint(*n_vehicles)
        n_walkers = self.n_walkers
        if isinstance(n_walkers, list):
            n_walkers = random.randint(*n_walkers)
        print("Spawn {} vehicle(s) and {} walker(s).".format(n_vehicles, n_walkers))
        self.spawn_npcs(n_vehicles, n_walkers)

        # Spawn the ego vehicle
        spawn_points = self.map.get_spawn_points()
        random.shuffle(spawn_points, random.random)
        for i in range(len(spawn_points)):
            next_spawn_point = spawn_points[i % len(spawn_points)]
            self.hero = self.world.try_spawn_actor(self.hero_blueprint, next_spawn_point)
            if self.hero is not None:
                print("Hero spawned!")
                break
        assert self.hero is not None

        # Spawn the new sensors
        for name, attributes in self.sensors_dict.items():
            SensorFactory.spawn(name, attributes, self.sensor_interface, self.hero)

    def spawn_npcs(self, n_vehicles, n_walkers):
        """Spawns vehicles and walkers, also setting up the Traffic Manager and its parameters"""
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        # --------------
        # Spawn vehicles
        # --------------
        spawn_points = self.world.get_map().get_spawn_points()
        n_spawn_points = len(spawn_points)

        if n_vehicles < n_spawn_points:
            random.shuffle(spawn_points)
        elif n_vehicles > n_spawn_points:
            logging.warning("{} vehicles were requested, but there were only {} available spawn points"
                            .format(n_vehicles, n_spawn_points))
            n_vehicles = n_spawn_points

        v_batch = []
        v_blueprints = self.world.get_blueprint_library().filter("vehicle.*")

        for n, transform in enumerate(spawn_points):
            if n >= n_vehicles:
                break
            v_blueprint = random.choice(v_blueprints)
            if v_blueprint.has_attribute('color'):
                color = random.choice(v_blueprint.get_attribute('color').recommended_values)
                v_blueprint.set_attribute('color', color)
            if v_blueprint.has_attribute('driver_id'):
                driver_id = random.choice(v_blueprint.get_attribute('driver_id').recommended_values)
                v_blueprint.set_attribute('driver_id', driver_id)
            v_blueprint.set_attribute('role_name', 'autopilot')

            v_batch.append(SpawnActor(v_blueprint, transform)
                           .then(SetAutopilot(FutureActor, True, self.tm_port)))

        results = self.client.apply_batch_sync(v_batch, True)
        if len(results) < n_vehicles:
            logging.warning("{} vehicles were requested but could only spawn {}"
                            .format(n_vehicles, len(results)))
        vehicles_id_list = [r.actor_id for r in results if not r.error]

        # Set automatic vehicle lights update
        all_vehicle_actors = self.world.get_actors(vehicles_id_list)
        for actor in all_vehicle_actors:
            self.traffic_manager.update_vehicle_lights(actor, True)

        # -------------
        # Spawn Walkers
        # -------------
        spawn_locations = [self.world.get_random_location_from_navigation() for i in range(n_walkers)]

        w_batch = []
        w_blueprints = self.world.get_blueprint_library().filter("walker.pedestrian.*")

        for spawn_location in spawn_locations:
            w_blueprint = random.choice(w_blueprints)
            if w_blueprint.has_attribute('is_invincible'):
                w_blueprint.set_attribute('is_invincible', 'false')
            w_batch.append(SpawnActor(w_blueprint, carla.Transform(spawn_location)))

        results = self.client.apply_batch_sync(w_batch, True)
        if len(results) < n_walkers:
            logging.warning("Could only spawn {} out of the {} requested walkers."
                            .format(len(results), n_walkers))
        walkers_id_list = [r.actor_id for r in results if not r.error]

        # Spawn the walker controllers
        wc_batch = []
        wc_blueprint = self.world.get_blueprint_library().find('controller.ai.walker')

        for walker_id in walkers_id_list:
            wc_batch.append(SpawnActor(wc_blueprint, carla.Transform(), walker_id))

        results = self.client.apply_batch_sync(wc_batch, True)
        if len(results) < len(walkers_id_list):
            logging.warning("Only {} out of {} controllers could be created. Some walkers might be stopped"
                            .format(len(results), n_walkers))
        controllers_id_list = [r.actor_id for r in results if not r.error]

        self.world.tick()

        for controller in self.world.get_actors(controllers_id_list):
            controller.start()
            controller.go_to_location(self.world.get_random_location_from_navigation())

        self.world.tick()
        self.tm_actors = self.world.get_actors(vehicles_id_list + walkers_id_list)
        self.tm_controllers = self.world.get_actors(controllers_id_list)

    def tick(self, control):
        """Performs one tick of the simulation, moving all actors, and getting the sensor data"""

        # Move hero vehicle
        if control is not None:
            self.apply_hero_control(control)

        # Tick once the simulation
        self.world.tick()

        # Move the spectator
        if self.config["enable_rendering"]:
            self.set_spectator_camera_view()

        # Return the new sensor data
        return self.get_sensor_data()

    def set_spectator_camera_view(self):
        """This positions the spectator as a 3rd person view of the hero vehicle"""
        transform = self.hero.get_transform()

        # Get the camera position
        server_view_x = transform.location.x - 5 * transform.get_forward_vector().x
        server_view_y = transform.location.y - 5 * transform.get_forward_vector().y
        server_view_z = transform.location.z + 3

        # Get the camera orientation
        server_view_roll = transform.rotation.roll
        server_view_yaw = transform.rotation.yaw
        server_view_pitch = transform.rotation.pitch

        # Get the spectator and place it on the desired position
        spectator = self.world.get_spectator()
        spectator.set_transform(
            carla.Transform(
                carla.Location(x=server_view_x, y=server_view_y, z=server_view_z),
                carla.Rotation(pitch=server_view_pitch,yaw=server_view_yaw,roll=server_view_roll),
            )
        )

    def apply_hero_control(self, control):
        """Applies the control calcualted at the experiment to the hero"""
        self.hero.apply_control(control)

    def get_sensor_data(self):
        """Returns the data sent by the different sensors at this tick"""
        sensor_data = self.sensor_interface.get_data()
        # print("---------")
        # world_frame = self.world.get_snapshot().frame
        # print("World frame: {}".format(world_frame))
        # for name, data in sensor_data.items():
        #     print("{}: {}".format(name, data[0]))
        return sensor_data
