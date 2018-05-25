import os
import numpy as np
import random
import subprocess
import zmq

import scipy


class SuperHexagonEnvironment:
    def __init__(self, controller, agent, frame_processor,
                 frame_shape=(480, 768, 3)):
        self.controller = controller
        self.agent = agent
        self.frame_processor = frame_processor
        self.frame_shape = frame_shape

        # This is used to actually get into the game mode.
        self.setup_moves = {
            60: ['space'],
            62: [],
            # 90: ['right'],
            # 92: [],
            # 100: ['right'],
            # 102: [],
            110: ['space'],
            112: [],
        }

    def start_game(self):
        """Starts the game with our hook loaded. The game will wait until our
        agent server begins accepting frames.
        """
        env = os.environ.copy()
        hook_path = os.path.join('hook', 'libhook.so')
        game_path = os.path.join(env.get('HOME'), '.local', 'share', 'Steam',
                                 'steamapps', 'common', 'Super Hexagon',
                                 'SuperHexagon')

        env['LD_PRELOAD'] = os.path.abspath(hook_path)
        args = ["bash", game_path]

        self.controller.handle_keys([])

        self.frame_counter = 0
        self.dead_until = None

        self.game_process = subprocess.Popen(args, env=env)

    def handle_game_loop(self):
        """Here we open up a ZeroMQ server and begin accepting frames.
        We use a server/client model where the game acts as the client and
        sends frames to the Python server. We analyze the frame, update our
        agent, and return a move to the client.
        """
        try:
            context = zmq.Context()
            socket = context.socket(zmq.REP)
            socket.bind('tcp://*:5555')

            frame_size = np.prod(self.frame_shape)

            prev_raw_frame = None

            while True:
                raw_frame = socket.recv()
                if len(raw_frame) < frame_size or raw_frame == prev_raw_frame:
                    socket.send_string("");
                    continue

                self.frame_counter += 1
                prev_raw_frame = raw_frame

                parsed_buff = np.frombuffer(raw_frame, dtype=np.dtype('B'))
                frame = np.flip(np.reshape(parsed_buff, self.frame_shape), 0)

                self.handle_frame(frame)

                socket.send_string("");
        except KeyboardInterrupt:
            self.controller.handle_keys([])
            self.game_process.terminate()

    def handle_frame(self, frame):
        """Here's where our actual game logic takes place."""

        self.frame_processor.push_frame(frame)

        # Take actions to enter the game

        if self.frame_counter <= max(self.setup_moves.keys()):
            self.do_setup()
            return

        # If we're dead, skip through a couple of frames until we can
        # become alive again. The agent does not act during this period.

        if self.dead_until:
            if self.frame_counter == self.dead_until - 20:
                self.controller.handle_keys(['space'])
            elif self.frame_counter == self.dead_until - 18:
                self.controller.handle_keys([])
            elif self.frame_counter > self.dead_until:
                self.dead_until = None
            return

        # Figure out which move to make and observe the appropriate reward

        sequence = self.frame_processor.get_sequence()
        if sequence is None:
            return

        move = self.agent.generate_action(sequence)
        self.controller.handle_keys(move)

        if np.allclose(frame, 255):
            self.agent.observe(0, terminal=True)
            self.dead_until = self.frame_counter + 120
        else:
            self.agent.observe(1, terminal=False)

    def do_setup(self):
        for frame, move in self.setup_moves.items():
            if self.frame_counter == frame:
                self.controller.handle_keys(move)
