import keyboard
import numpy as np
import os
import time

from environment import ACTION_STOP, ACTION_LEFT, ACTION_RIGHT


class DemoRecorder:
    def __init__(self, environment):
        self.environment = environment

        self.states = []
        self.moves = []

    def start_recording(self):
        self.environment.reset()

        try:
            while True:
                frame = self.environment.get_and_store_frame()
                self.states.append(self.environment.state)
                # self.states.append(frame)
                self.moves.append(self.get_action())

                if self.environment.game_over(frame):
                    self.flush_episode_demo()
                    self.environment.reset()
        except KeyboardInterrupt:
            self.environment.close()

    def flush_episode_demo(self):
        if len(self.states) < 300:
            return

        filename = f"demo_{int(time.time())}.npy"
        states = np.stack(self.states, axis=0)
        moves = np.stack(self.moves, axis=0)

        np.savez_compressed(
            os.path.join('demo', filename),
            states=states,
            moves=moves,
        )

        self.states = []
        self.moves = []

    def get_action(self):
        if keyboard.is_pressed('left'):
            return ACTION_LEFT
        elif keyboard.is_pressed('right'):
            return ACTION_RIGHT
        else:
            return ACTION_STOP
