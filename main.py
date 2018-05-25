#!/usr/bin/env python3

import os
import numpy as np
import scipy.misc

from environment import SuperHexagonEnvironment

from agent import SuperHexagonAgent
from controller import Controller
from frame import FrameProcessor

moves = {
    'LEFT': ['left'],
    'RIGHT': ['right'],
    'NONE': []
}

def main():
    frame_processor = FrameProcessor()
    environment = SuperHexagonEnvironment(
        controller=Controller(),
        agent=SuperHexagonAgent(
            [*frame_processor.output_shape, frame_processor.buffer_size],
            moves,
        ),
        frame_processor=frame_processor,
    )

    environment.start_game()
    environment.handle_game_loop()


if __name__ == "__main__":
    main()
