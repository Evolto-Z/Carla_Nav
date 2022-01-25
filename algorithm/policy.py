from agents.navigation.behavior_agent import BehaviorAgent


class MyPolicy:
    def __init__(self, hero, config={}):
        self.agent = BehaviorAgent(hero, "normal")

    def inference(self):
        control = self.agent.run_step()
        control.manual_gear_shift = False
        return control
