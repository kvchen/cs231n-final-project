import numpy as np
import skimage.transform
import skimage.color
import sklearn.preprocessing


class FrameProcessor:
    def __init__(self, buffer_size=5, output_shape=(100, 100)):
        self.buffer_size = buffer_size
        self.buffer = []

        self.output_shape = output_shape
        self.processing_pipeline = [
            lambda frame: skimage.transform.resize(frame, output_shape),
            skimage.color.rgb2gray,
        ]

    def push_frame(self, frame):
        if len(self.buffer) >= self.buffer_size:
            self.buffer = self.buffer[1:]

        processed_frame = self.preprocess_frame(frame)
        self.buffer.append(processed_frame)

    def preprocess_frame(self, frame):
        for step in self.processing_pipeline:
            frame = step(frame)

        return frame

    def get_sequence(self):
        if len(self.buffer) < self.buffer_size:
            return None
        else:
            return np.stack(self.buffer, axis=2)
