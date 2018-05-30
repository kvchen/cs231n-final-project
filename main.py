#!/usr/bin/env python3

import click

from tensorforce.execution import Runner

from environment import SuperHexagonEnvironment
from controller import Controller
from frame import FrameProcessor

from demo_recorder import DemoRecorder
from agent import get_dqfd_agent, get_ppo_agent


def start_agent_mode(environment, agent, episodes, bootstrap):
    def episode_finished(runner, runner_id):
        """We just use this function to log some information about each
        episode.
        """
        if runner.episode % 100 == 0:
            avg_reward = sum(runner.episode_rewards[-100:]) / 100
            print(
                "Average reward over last 100 episodes: {}".format(avg_reward),
            )

        return True

    if agent == 'ppo':
        tf_agent = get_ppo_agent(environment, bootstrap)
    elif agent == 'dqfd':
        tf_agent = get_dqfd_agent(environment, bootstrap)

    runner = Runner(agent=tf_agent, environment=environment)
    runner.run(episodes=episodes, episode_finished=episode_finished)


def start_record_mode(environment):
    """This allows us to play the game and record demos. Each demo consists of
    an array of (state, action) pairs. We can later play these demos back to
    "train" our agent.

    Since keyboard listens directly to the device file, we need to use sudo
    when invoking this script, i.e.:

    $ sudo su
    $ source .env/bin/activate
    $ python3 main.py --mode record
    """
    recorder = DemoRecorder(environment)
    recorder.start_recording()


@click.command()
@click.option('--mode', type=click.Choice(['agent', 'record']),
              default='agent')
@click.option('--agent', type=click.Choice(['ppo', 'dqfd']),
              default='ppo')
@click.option('--episodes', type=int,
              default=int(1e6))
@click.option('--bootstrap', is_flag=True,
              help="Use recorded demos to bootstrap training agent")
def main(mode, agent, episodes, bootstrap):
    frame_processor = FrameProcessor()
    environment = SuperHexagonEnvironment(
        controller=Controller(),
        frame_processor=frame_processor,
    )

    if mode == 'agent':
        start_agent_mode(environment, agent=agent, episodes=episodes,
                         bootstrap=bootstrap)
    elif mode == 'record':
        start_record_mode(environment)


if __name__ == "__main__":
    main()
