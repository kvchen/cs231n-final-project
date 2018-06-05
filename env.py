#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gym
import logging
import numpy as np
import os
import signal
import subprocess
import zmq
import imageio
import time
import random

from gym import spaces


ACTION_NAMES = ['NOOP', 'LEFT', 'RIGHT']
ACTION_KEYS = [[], ['left'], ['right']]

ACTION_STOP = 0
ACTION_LEFT = 1
ACTION_RIGHT = 2


class SuperHexagonEnv(gym.Env):

    def __init__(self, controller, frame_shape=(480, 768, 3)):
        self.__version__ = "0.0.1"
        logging.info("SuperHexagon - Version {}".format(self.__version__))

        # Do some initialization

        self.controller = controller
        self.frame_shape = frame_shape
        self.frame_len = np.prod(frame_shape)

        # Set the environment attributes
        self.action_space = spaces.Discrete(len(ACTION_NAMES))
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=self.frame_shape,
            dtype=np.float,
        )
        self.num_envs = 1

        # Initialize game process and server
        self.controller.release_keys(sum(ACTION_KEYS, []))
        self.setup_socket()
        self.start_game_process()

        # Navigate into the actual game
        self.goto_game()

    def step(self, action):
        # time.sleep(5)
        self.episode_len += 1

        reward = 1
        terminal = False

        self.controller.handle_keys(ACTION_KEYS[action])
        frame = self.get_next_frame()

        # if self.episode_len % 30 == 0:
        #     reward = 1

        if self.game_over(frame):
            reward = -1
            terminal = True
        # else:
        #     # Testing new reward function
        #     dist_to_wall = get_distance_to_nearest_wall(frame)
        #     if dist_to_wall is not None:
        #         reward = (dist_to_wall - 30) / 380

        return frame, reward, terminal, {}

    def reset(self):
        self.handle_death()
        self.episode_len = 0
        return self.get_next_frame()

    def close(self):
        self.controller.handle_keys([])
        os.killpg(os.getpgid(self.game_process.pid), signal.SIGTERM)
        # self.game_process.kill()

    def get_action_meanings(self):
        return ACTION_NAMES

    # Helper methods

    def game_over(self, frame):
        return np.allclose(frame, 255)

    # Process handling

    def start_game_process(self):
        """Starts the game with our hook loaded. The game will wait until our
        agent server begins accepting frames.
        """
        env = os.environ.copy()
        env['HOME'] = '/home/kevinchen'
        env['TF_CPP_MIN_LOG_LEVEL'] = '3'

        game_path = os.path.join(env.get('HOME'), '.local', 'share', 'Steam',
                                 'steamapps', 'common', 'Super Hexagon',
                                 'SuperHexagon')
        args = ["bash", game_path]

        # Ensure that we're hooking our game process
        hook_path = os.path.join('hook', 'libhook.so')
        env['LD_PRELOAD'] = os.path.abspath(hook_path)

        self.game_process = subprocess.Popen(
            args,
            env=env,
            stdout=subprocess.DEVNULL,
            preexec_fn=os.setsid,
        )

    def goto_game(self):
        """Consumes frames until we're sure we're dead. The next reset will
        trigger a new game.
        """
        self.do_moves({
            60: ['space'],
            62: [],
            90: ['right'],
            92: [],
            100: ['right'],
            102: [],
            110: ['space'],
            112: [],
            600: [],
        })

    def handle_death(self):
        """Consumes frames until we can get back into a playable state."""
        self.do_moves({110: ['space'], 112: [], 150: [], 190: []})

    def do_moves(self, moves):
        """Consumes frames and makes the appropriate moves. Used to setup the
        game for a new episode.
        """
        for i in range(max(moves)):
            self.get_next_frame()
            if i in moves:
                self.controller.handle_keys(moves[i])

    def setup_socket(self):
        """Sets up a ZeroMQ socket that listens for new frames."""
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind('tcp://*:5555')

        self.prev_raw_frame = None
        self.socket = socket

    def get_next_frame(self):
        """Fetches a new frame from our game hook. GLClear is called multiple
        times per game frame, so we want to make sure we only emit one of
        those.
        """
        while True:
            raw_frame = self.socket.recv()
            self.socket.send_string("")

            if (
                len(raw_frame) == self.frame_len and
                raw_frame != self.prev_raw_frame
            ):
                break

        self.prev_raw_frame = raw_frame

        parsed_buff = np.frombuffer(raw_frame, dtype=np.dtype('B'))
        return np.flip(np.reshape(parsed_buff, self.frame_shape), 0)
