"""
Carla Navigation
"""
from __future__ import print_function

from carla_agents.behavior_agent import BehaviorAgent
from rllib_integration.carla_env import CarlaEnv
from rllib_integration.carla_core import kill_all_servers
from algorithm.policy import MyPolicy
from experiment import MyExperiment
from algorithm.trainer import MyTrainer
from tools.misc import compute_action

import yaml
import random
import torch
import argparse
import os
import shutil
from collections import deque

torch.backends.cudnn.benchmark = True


def run(args):
    try:
        config = args.config
        debug = config["debug"]
        random.seed(config["seed"])

        env = CarlaEnv(config["env_config"], config["seed"])
        policy = MyPolicy(config["policy_config"])
        trainer = MyTrainer(policy.get_model(), config["trainer_config"])

        start_il_epoch = -1
        start_rl_epoch = -1
        if args.restore:
            start_il_epoch = args.il_checkpoint["epoch"]
            start_rl_epoch = args.rl_checkpoint["epoch"]

        # imitation learning phase
        total_epoch = config["total_il_epoch"]
        guider = BehaviorAgent(config["guider_config"])
        max_memory_size = 1024
        memory = deque(maxlen=max_memory_size)
        batch_size = config["il_batch_size"]
        for i in range(start_il_epoch + 1, total_epoch):
            obs = env.reset()
            done = False
            guider.reset(env.hero)
            policy.reset(env.hero)

            # Set the agent destination
            world_map = env.hero.get_world().get_map()
            spawn_points = world_map.get_spawn_points()
            destination = random.choice(spawn_points).location
            guider.set_destination(destination)

            while not done:
                command = guider.get_target_road_option()
                control = guider.run_step(debug=debug)
                action = compute_action(control)
                next_obs, reward, done, info = env.step(control)
                done = done | guider.done()
                next_command = guider.get_target_road_option()
                state = (obs, command)
                next_state = (next_obs, next_command)
                # memory.append((state, action, reward, next_state, done))
                obs = next_obs
            #     if len(memory) == 0:
            #         while len(memory) < MAX_MEMORY_SIZE and not done:
            #             command = guider.get_target_road_option()
            #             control = guider.run_step()
            #             action = compute_action(control)
            #             next_obs, reward, done, info = env.step(control)
            #             done = done | guider.done()
            #             next_command = guider.get_target_road_option()
            #             state = (obs, command)
            #             next_state = (next_obs, next_command)
            #             memory.append((state, action, reward, next_state, done))
            #             obs = next_obs
            # memory.clear()
            print("Done.\n")

        # # reinforcement learning phase
        # total_rl_epoch = config["total_rl_epoch"]
        # for i in range(start_rl_epoch + 1, total_rl_epoch):
        #     pass

    finally:
        kill_all_servers()


def parse_config(args):
    """
    Parses the .yaml configuration file into a readable dictionary
    """
    with open(args.config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config["env"] = CarlaEnv
        config["env_config"]["experiment"]["type"] = MyExperiment
    return config


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument("--config_file",
                           default="config.yaml",
                           help="Configuration file (*.yaml)")
    argparser.add_argument("--exp_id",
                           default=0,
                           help="Experiment ID")
    argparser.add_argument("--restore",
                           action="store_true",
                           default=False,
                           help="Restore the experiment")
    argparser.add_argument("--overwrite",
                           action="store_true",
                           default=False,
                           help="Flag to overwrite a an experiment")
    args = argparser.parse_args()

    # check overwrite and restore
    if args.overwrite and args.restore:
        raise RuntimeError("Both 'overwrite' and 'restore' cannot be True at the same time")
    path = os.path.join("checkpoints", "exp_{}".format(args.exp_id))
    if args.overwrite:
        if os.path.isdir(path):
            shutil.rmtree(path)
            print("Removing all contents inside '" + path + "'")
    elif not args.restore:
        if os.path.isdir(path) and len(os.listdir(path)) != 0:
            raise RuntimeError(
                "The directory where you are trying to train (" +
                path + ") is not empty. "
                "To start a new training instance, make sure this folder is either empty, non-existing "
                "or use the '--overwrite' argument to remove all the contents inside"
            )

    if args.restore:
        args.il_checkpoint = torch.load(os.path.join(path, "il_checkpoint.pth"))
        args.rl_checkpoint = torch.load(os.path.join(path, "rl_checkpoint.pth"))
    else:
        args.il_checkpoint = None
        args.rl_checkpoint = None
    args.tb_logdir = os.path.join(path, "tb_log")
    args.config = parse_config(args)
    run(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\nAll done!')