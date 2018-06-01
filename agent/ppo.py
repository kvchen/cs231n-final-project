import numpy as np
import tensorflow as tf

from baselines import logger
from baselines.a2c.utils import ortho_init
from baselines.bench import Monitor
from baselines.common.atari_wrappers import \
    ClipRewardEnv, WarpFrame, MaxAndSkipEnv
from baselines.common.distributions import make_pdtype
from baselines.common.input import observation_input
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.vec_frame_stack import VecFrameStack


def wrap_env(env):
    env = WarpFrame(env)
    env = ClipRewardEnv(env)
    # env = NoopResetEnv(env, noop_max=8)
    env = MaxAndSkipEnv(env, skip=4)
    env = Monitor(env, logger.get_dir())
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, 4)
    return env


def ppo_cnn_model(state):
    kernel_initializer = ortho_init(np.sqrt(2))
    conv_kwargs = {
        "activation": tf.nn.relu,
        "kernel_initializer": kernel_initializer,
    }

    c1 = tf.layers.conv2d(state, filters=32, kernel_size=8, strides=1,
                          name="c1", **conv_kwargs)
    p1 = tf.layers.max_pooling2d(c1, pool_size=2, strides=1, name="p1")
    c2 = tf.layers.conv2d(p1, filters=64, kernel_size=4, strides=1,
                          name="c2", **conv_kwargs)
    p2 = tf.layers.max_pooling2d(c2, pool_size=2, strides=1, name="p2")
    c3 = tf.layers.conv2d(p2, filters=64, kernel_size=3, strides=1,
                          name="c3", **conv_kwargs)
    p3 = tf.layers.max_pooling2d(c3, pool_size=2, strides=1, name="p3")
    f = tf.layers.flatten(p3)
    return tf.layers.dense(f, units=512, activation=tf.nn.relu,
                           kernel_initializer=kernel_initializer)


class CNNPolicy:
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False):
        self.pdtype = make_pdtype(ac_space)
        X, processed_x = observation_input(ob_space, nbatch)

        with tf.variable_scope("model", reuse=reuse):
            h = ppo_cnn_model(processed_x)
            v = tf.layers.dense(h, 1, name='v')
            vf = tf.squeeze(v, axis=[1])
            self.pd, self.pi = self.pdtype.pdfromlatent(h, init_scale=0.01)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X: ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X: ob})

        self.X = X
        self.vf = vf
        self.step = step
        self.value = value
