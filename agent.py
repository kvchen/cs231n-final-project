import json

from tensorforce.agents import PPOAgent


def get_agent(environment):
    with open('config/cnn_network.json', 'r') as infile:
        network = json.load(infile)

    return PPOAgent(
        states=environment.states,
        actions=environment.actions,
        network=network,
        actions_exploration={
            "type": "epsilon_anneal",
            "initial_epsilon": 1.0,
            "final_epsilon": 0.01,
            "timesteps": int(1e6),
        },
        saver={
            "directory": "checkpoint/ppo",
            "seconds": 1800,
        },
    )
