import numpy as np

from tensorforce.agents import RandomAgent, PPOAgent


class SuperHexagonAgent:
    def __init__(self, frame_shape, game_inputs):
        self.frame_shape = frame_shape
        self.game_inputs = game_inputs
        self.game_inputs_mapping = {
            idx: key for idx, key in enumerate(self.game_inputs)
        }

        states = {"type": "float", "shape": self.frame_shape}
        actions = {"type": "int", "num_actions": len(self.game_inputs)}

        # summary = {
        #     "directory": "./tensorboard/",
        #     "seconds": 5,
        #     "labels": [
        #         "configuration",
        #         "gradients_scalar",
        #         "regularization",
        #         "inputs",
        #         "losses",
        #         "variables"
        #     ]
        # }

        network = [
            {"type": "conv2d", "size": 32, "window": 5, "stride": 1},
            {"type": "pool2d", "stride": 2},
            {"type": "conv2d", "size": 64, "window": 4, "stride": 1},
            {"type": "pool2d", "stride": 2},
            {"type": "conv2d", "size": 64, "window": 3, "stride": 1},
            {"type": "flatten"},
            {"type": "dense", "size": 512}
        ]

        self.agent = PPOAgent(
            states=states,
            actions=actions,
            network=network,
            discount=0.99,
            batching_capacity=2000,
            optimization_steps=50,
            actions_exploration={
                "type": "epsilon_anneal",
                "initial_epsilon": 0.5,
                "final_epsilon": 0.0,
                "timesteps": 10000000,
            },
            saver={
                "directory": None,
                "seconds": 600
            },

        )

    def generate_action(self, sequence):
        action = self.agent.act(sequence)
        label = self.game_inputs_mapping[action]
        return self.game_inputs[label]

    def observe(self, reward=0, terminal=False):
        self.agent.observe(reward=reward, terminal=terminal)
