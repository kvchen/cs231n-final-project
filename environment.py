import os
import numpy as np
import subprocess
import zmq

# from frame_utils import get_distance_to_nearest_wall
from tensorforce.environments import Environment

ACTION_NAMES = ['stop', 'left', 'right']
ACTION_KEYS = [[], ['left'], ['right']]

ACTION_STOP = 0
ACTION_LEFT = 1
ACTION_RIGHT = 2


class SuperHexagonEnvironment(Environment):
    def __init__(self, controller, frame_processor, frame_shape=(480, 768, 3)):
        self.controller = controller
        self.frame_processor = frame_processor
        self.frame_shape = frame_shape
        self.frame_len = np.prod(frame_shape)

        # Initialize game process and server
        self.setup_socket()
        self.start_game_process()

        # Navigate into the actual game
        self.goto_game()

    def execute(self, actions):
        reward = 1
        terminal = False

        self.controller.handle_keys(ACTION_KEYS[actions])
        frame = self.get_and_store_frame()

        if self.game_over(frame):
            reward = -1
            terminal = True
        # else:
        #     # Testing new reward function
        #     dist_to_wall = get_distance_to_nearest_wall(frame)
        #     if dist_to_wall is not None:
        #         reward = (dist_to_wall - 30) / 380

        return self.state, terminal, reward

    def get_and_store_frame(self):
        frame = self.recv_frame()
        if not self.game_over(frame):
            self.frame_processor.push_frame(frame)

        return frame

    def reset(self):
        # self.do_moves({
        #     0: ['esc'],
        #     2: [],
        # })
        self.handle_death()
        return self.state

    def close(self):
        self.controller.handle_keys([])
        self.game_process.kill()

    def game_over(self, frame):
        return np.allclose(frame, 255)

    @property
    def state(self):
        """Fetches the current sequence from the frame processor and returns
        it as the state.
        """
        return self.frame_processor.get_sequence()

    @property
    def states(self):
        return {
            "shape": self.frame_processor.sequence_shape,
            "type": "float32",
        }

    @property
    def actions(self):
        return {
            "num_actions": len(ACTION_NAMES),
            "type": "int",
        }

    def start_game_process(self):
        """Starts the game with our hook loaded. The game will wait until our
        agent server begins accepting frames.
        """
        env = os.environ.copy()
        env['HOME'] = '/home/kevinchen'
        env['TF_CPP_MIN_LOG_LEVEL'] = '3'

        hook_path = os.path.join('hook', 'libhook.so')
        game_path = os.path.join(env.get('HOME'), '.local', 'share', 'Steam',
                                 'steamapps', 'common', 'Super Hexagon',
                                 'SuperHexagon')

        env['LD_PRELOAD'] = os.path.abspath(hook_path)
        args = ["bash", game_path]

        self.controller.handle_keys([])
        self.game_process = subprocess.Popen(
            args,
            env=env,
            stdout=subprocess.DEVNULL,
        )

    def goto_game(self):
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
        self.do_moves({
            100: ['space'],
            102: [],
            140: [],
        })
        self.do_moves({0: [], 4: []}, push_frames=True)

    def do_moves(self, moves, push_frames=False):
        """Consumes frames and makes the appropriate moves. Used to setup the
        game for a new episode.
        """
        for i in range(max(moves)):
            if i in moves:
                self.controller.handle_keys(moves[i])

            frame = self.recv_frame()
            if push_frames:
                self.frame_processor.push_frame(frame)

    def setup_socket(self):
        """Sets up a ZeroMQ socket that listens for new frames."""
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind('tcp://*:5555')

        self.prev_raw_frame = None
        self.socket = socket

    def recv_frame(self):
        """Fetches a new frame from our game hook."""
        # Spin until we get a valid frame
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
        frame = np.flip(np.reshape(parsed_buff, self.frame_shape), 0)
        return frame
