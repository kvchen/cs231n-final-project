import numpy as np
import skimage.transform
import skimage.color
import skimage.filters


def rescale_intensity(frame):
    cropped_frame = frame[70:]
    imin, imax = np.min(cropped_frame), np.max(cropped_frame)
    return np.clip((frame - imin) / (imax - imin), 0, 1)


def remove_times(frame):
    frame[:30, :190] = 0
    frame[:48, 560:] = 0
    return frame


class FrameProcessor:
    def __init__(self, buffer_size=2, output_shape=(96, 96)):
        self.buffer_size = buffer_size
        self.buffer = []

        self.output_shape = output_shape
        self.sequence_shape = output_shape + (buffer_size,)

        self.processing_pipeline = [
            skimage.color.rgb2gray,
            rescale_intensity,
            lambda frame: frame > 0.5,
            remove_times,
            lambda frame: skimage.transform.resize(frame, output_shape),
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
            return np.zeros(self.sequence_shape)
        else:
            return np.stack(self.buffer, axis=2)
