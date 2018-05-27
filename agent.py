import glob
import numpy as np
import json
from tqdm import tqdm, trange

from tensorforce.agents import PPOAgent


def load_demos():
    """Combines all the demos into a massive dataset. Returns a dictionary
    with keys "state" and "moves".
    """
    states = []
    moves = []

    for demo_path in tqdm(
        glob.glob('demo/demo_*.npy.npz'),
        desc="Loading demos into memory",
    ):
        data = np.load(demo_path)
        states.append(data['states'])
        moves.append(data['moves'])

    return {
        "states": np.concatenate(states, axis=0),
        "moves": np.concatenate(moves, axis=0),
    }


def bootstrap_agent(agent, batch_size=100):
    demos = load_demos()

    x = demos['states']
    y = demos['moves']

    # We treat the demo moves as the source of truth.
    reward = np.ones((batch_size,), dtype=np.float32)

    for epoch in trange(100, desc="Bootstrapping agent"):
        avg_cost = 0.

        # shuffle indexes
        indices = np.arange(x.shape[0], dtype=np.int32)
        np.random.shuffle(indices)
        total_batch = int(len(x) / batch_size)

        # Loop over all batches
        for i in range(total_batch):
            index_list = indices[i*batch_size:i*batch_size + batch_size]
            batch_x, batch_y = x[index_list], y[index_list]

            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = agent.model.session.run(
                [agent.model.optimize, agent.model.loss],
                feed_dict={
                    agent.model.state['state']: batch_x,
                    agent.model.action['action']: batch_y,
                    agent.model.reward: reward,
                })

            # Compute average loss
            avg_cost += c / total_batch

        # Display logs per epoch step
        print("Epoch: {}, cost: {}".format(epoch, avg_cost))


def get_agent(environment, bootstrap):
    with open('config/cnn_network.json', 'r') as infile:
        network = json.load(infile)

    agent = PPOAgent(
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

    if bootstrap:
        bootstrap_agent(agent)

    return agent
