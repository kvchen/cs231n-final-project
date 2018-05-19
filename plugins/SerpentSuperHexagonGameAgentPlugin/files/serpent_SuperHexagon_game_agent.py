import random
import time
import numpy as np

from serpent.game_agent import GameAgent
from serpent.input_controller import KeyboardKey


class SerpentSuperHexagonGameAgent(GameAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.frame_handlers["PLAY"] = self.handle_play
        self.frame_handler_setups["PLAY"] = self.setup_play

    def setup_play(self):
        self.frames_seen = 0
        self.moves = {
            "NONE": [],
            "LEFT": [KeyboardKey.KEY_LEFT],
            "RIGHT": [KeyboardKey.KEY_RIGHT],
        }

        # Advance from the menu screen to actual gameplay.
        # Menu -> Hexagon -> Hexagoner -> Hexagonest -> Gameplay
        self.input_controller.tap_key(KeyboardKey.KEY_SPACE)
        self.input_controller.tap_key(KeyboardKey.KEY_RIGHT)
        self.input_controller.tap_key(KeyboardKey.KEY_SPACE)

    def handle_play(self, game_frame):
        self.frames_seen += 1

        # Check to see if we've hit a game over.
        # We do this by checking if the time in the current and previous frames
        # is the same.
        if len(self.game_frame_buffer.frames) > 2:
            current_frame, prev_frame = self.game_frame_buffer.frames[:2]
            current_frame_time = current_frame.frame[:64, -96:]
            prev_frame_time = prev_frame.frame[:64, -96:]

            if np.allclose(current_frame_time, prev_frame_time):
                print('Died!')
                self.input_controller.tap_key(KeyboardKey.KEY_SPACE)

        move = random.choice(list(self.moves.values()))
        self.input_controller.handle_keys(move)
