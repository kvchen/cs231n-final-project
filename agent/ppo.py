import cv2
import gym
import numpy as np
import tensorflow as tf
import imageio
import time
import random

from baselines import logger
from baselines.a2c.utils import ortho_init
from baselines.bench import Monitor
from baselines.common.atari_wrappers import \
    ClipRewardEnv, MaxAndSkipEnv, FrameStack
from baselines.common.distributions import make_pdtype
from baselines.common.input import observation_input
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from gym import spaces


SHIP_BIT_ARRAY = (255 * np.array(
    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
     [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
     [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
     [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
     [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
     [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
     [0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
)).astype(np.uint8)

SHIP_CONTOUR = cv2.findContours(SHIP_BIT_ARRAY, cv2.RETR_LIST, 1)[1][0]

SHIP_REPLACEMENT_PATTERN = np.zeros((24, 32))
SHIP_REPLACEMENT_PATTERN[1::2, ::2] = 255
SHIP_REPLACEMENT_PATTERN[::2, 1::2] = 255


def get_ship_center(frame):
    center = frame[160:320, 300:468]
    _, contours, _ = cv2.findContours(
        center,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
        offset=(300, 160),
    )

    match_coefficients = [
        cv2.matchShapes(contour, SHIP_CONTOUR, 1, 0.0)
        for contour in contours
    ]
    closest_idx = np.argmin(match_coefficients)
    closest_contour = contours[closest_idx]

    return np.mean(closest_contour, axis=0)[0][::-1].astype(int)


def convert_frame_to_polar(frame):
    center_y, center_x = np.array(frame.shape[:2]) / 2
    im = cv2.linearPolar(
        frame,
        (center_x, center_y),
        center_y + 40,
        cv2.WARP_FILL_OUTLIERS,
    )

    # Copy 120 of the top rows to the bottom so the ship doesn't get cut off
    return np.concatenate((im, im[:120]), axis=0)


class ThresholdResizeFrame(gym.ObservationWrapper):
    def __init__(self, env, width=96, height=96):
        """Thresholds frames to produce a b/w image."""
        gym.ObservationWrapper.__init__(self, env)

        self.width = width
        self.height = height
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.height, self.width, 1),
            dtype=np.uint8,
        )

    def observation(self, frame):
        # should_save = random.random() < 0.01
        # curtime = 'screenshots/' + str(int(time.time()))

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        frame = convert_frame_to_polar(frame)

        # if should_save:
        #     cv2.imwrite(f"{curtime}-gray.png", frame[..., 2])

        frame = ((frame[..., 2] > 128) * 255).astype(np.uint8)

        # if not np.allclose(frame, 255):
        #     # Enlarge the ship
        #     ship_center = get_ship_center(frame)
        #     if ship_center is not None:
        #         y, x = ship_center
        #         frame[y-10:y+10, x-16:x+16] = 255

        # if should_save:
        #     cv2.imwrite(f"{curtime}-clipped.png", frame)

        # if random.random() < 0.01:
        #     imageio.imwrite(
        #         f"screenshots/screenshot_{int(time.time())}.png",
        #         frame,
        #     )

        frame = cv2.resize(
            frame,
            (self.width, self.height),
            interpolation=cv2.INTER_AREA,
        )

        # if should_save:
        #     cv2.imwrite(f"{curtime}-resized.png", frame)

        return frame[:, :, None]


def wrap_env_dqn(env):
    env = ThresholdResizeFrame(env)
    env = ClipRewardEnv(env)
    env = MaxAndSkipEnv(env, skip=4)
    env = Monitor(env, logger.get_dir(), allow_early_resets=True)
    env = FrameStack(env, 4)
    return env


def wrap_env_ppo(env):
    env = ThresholdResizeFrame(env)
    # env = WarpFrame(env)
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
        # "padding": "same",
    }
    pool_kwargs = {
        # "padding": "same",
    }

    c1 = tf.layers.conv2d(state, filters=32, kernel_size=8, strides=1,
                          name="c1", **conv_kwargs)
    p1 = tf.layers.max_pooling2d(c1, pool_size=2, strides=2, name="p1",
                                 **pool_kwargs)
    c2 = tf.layers.conv2d(p1, filters=64, kernel_size=4, strides=1,
                          name="c2", **conv_kwargs)
    p2 = tf.layers.max_pooling2d(c2, pool_size=2, strides=2, name="p2",
                                 **pool_kwargs)
    c3 = tf.layers.conv2d(p2, filters=64, kernel_size=3, strides=1,
                          name="c3", **conv_kwargs)
    p3 = tf.layers.max_pooling2d(c3, pool_size=2, strides=2, name="p3",
                                 **pool_kwargs)
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
