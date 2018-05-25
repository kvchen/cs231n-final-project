import pyautogui

pyautogui.PAUSE = 0

class Controller:
    def __init__(self):
        self.previous_keys = set()

    def handle_keys(self, keys):
        current_keys = set(keys)

        if current_keys == self.previous_keys:
            return

        pressed = current_keys - self.previous_keys
        released = self.previous_keys - current_keys

        for key in pressed:
            pyautogui.keyDown(key)

        for key in released:
            pyautogui.keyUp(key)

        self.previous_keys = current_keys
