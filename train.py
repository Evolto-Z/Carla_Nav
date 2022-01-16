from rllib_integration.carla_env import CarlaEnv
from algorithm.policy import MyPolicy
from experiment import MyExperiment
import yaml
import cv2


def parse_config(configuration_file):
    """
    Parses the .yaml configuration file into a readable dictionary
    """
    with open(configuration_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config["env"] = CarlaEnv
        config["env_config"]["experiment"]["type"] = MyExperiment
    return config


configuration_file = "./config.yaml"
config = parse_config(configuration_file)
env = CarlaEnv(config["env_config"])
obs = env.reset()
policy = MyPolicy(env.get_hero())


cnt = 64
while True:
    control = policy.inference()
    env.step(control)
    cnt -= 1
    if cnt == 0:
        cv2.imshow("demo", obs)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
