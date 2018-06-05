import click
import os
import tensorflow as tf
import time

from agent.ppo import wrap_env_dqn, wrap_env_ppo, CNNPolicy
from baselines import deepq, logger
from baselines.ppo2 import ppo2
from controller import Controller
from env import SuperHexagonEnv

DEFAULT_GAME_PATH = os.path.join(
    os.environ.get('HOME'),
    'superhexagon',
    'SuperHexagon',
)


def train(agent, env, checkpoint_path="checkpoint"):
    checkpoint_path = os.path.join(checkpoint_path, agent)

    if agent == 'deepq':
        env = wrap_env_dqn(env)
        model = deepq.models.cnn_to_mlp(
            convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
            hiddens=[256],
            dueling=True,
        )
        deepq.learn(
            env,
            q_func=model,
            lr=5e-4,
            max_timesteps=int(1e7),
            buffer_size=50000,
            exploration_fraction=0.1,
            exploration_final_eps=0.01,
            train_freq=4,
            learning_starts=10000,
            target_network_update_freq=1000,
            gamma=0.99,
            prioritized_replay=True,
            prioritized_replay_alpha=0.6,
            checkpoint_freq=int(5e3),
            checkpoint_path="checkpoint/dqn",
        )
    elif agent == 'ppo':
        env = wrap_env_ppo(env)
        with tf.Session().as_default():
            try:
                model = ppo2.learn(
                    policy=CNNPolicy,
                    env=env,
                    nsteps=256,
                    nminibatches=4,
                    noptepochs=4,
                    ent_coef=.01,
                    lr=lambda f: f * 2.5e-4,
                    cliprange=lambda f: f * 0.1,
                    total_timesteps=int(1e7),
                    log_interval=1,
                    save_interval=int(100),
                )
            except Exception:
                model.save(checkpoint_path)
                raise


@click.command()
@click.option('--agent', type=click.Choice(['deepq', 'ppo']),
              default='ppo')
@click.option('--game-path', type=click.Path(exists=True),
              default=DEFAULT_GAME_PATH)
@click.option('--hook-path', type=click.Path(exists=True),
              default=os.path.join('hook', 'libhook.so'))
def main(agent, game_path, hook_path):
    logger.configure(
        dir=os.path.join('log', agent, str(int(time.time()))),
        format_strs=['json'],
    )

    controller = Controller()
    env = SuperHexagonEnv(
        controller=controller,
        game_path=game_path,
        hook_path=hook_path,
    )

    try:
        train(agent, env)
    except Exception as e:
        env.close()
        raise e


if __name__ == "__main__":
    main()
