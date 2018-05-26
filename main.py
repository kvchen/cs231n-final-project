#!/usr/bin/env python3

import click

from environment import SuperHexagonEnvironment

from agent import get_agent
from controller import Controller
from frame import FrameProcessor
from tensorforce.execution import Runner


def start_agent_mode(environment, episodes):
    def episode_finished(runner, runner_id):
        """We just use this function to log some information about each
        episode.
        """
        if runner.episode % 100 == 0:
            print(sum(runner.episode_rewards[-100:]) / 100)

        return True

    agent = get_agent(environment)

    runner = Runner(agent=agent, environment=environment)
    runner.run(episodes=episodes, episode_finished=episode_finished)


def start_record_mode(environment):
    """This allows us to play the game and record demos. Each demo consists of
    an array of (state, action) pairs. We can later play these demos back to
    "train" our agent.
    """
    pass


@click.command()
@click.option('--mode', type=click.Choice(['agent', 'record']),
              default='agent')
@click.option('--episodes', type=int,
              default=int(1e6))
def main(mode, episodes):
    frame_processor = FrameProcessor()
    environment = SuperHexagonEnvironment(
        controller=Controller(),
        frame_processor=frame_processor,
    )

    if mode == 'agent':
        start_agent_mode(environment, episodes=episodes)
    elif mode == 'record':
        start_record_mode(environment)


if __name__ == "__main__":
    main()
