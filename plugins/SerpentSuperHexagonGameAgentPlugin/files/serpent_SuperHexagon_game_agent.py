import random
import time

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
        print(self.frames_seen)

        move = random.choice(list(self.moves.values()))
        self.input_controller.handle_keys(move)
