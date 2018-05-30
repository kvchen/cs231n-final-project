import numpy as np
import math
import cv2
import skimage.transform
import skimage.color
import skimage.filters
from skimage import img_as_ubyte
import itertools


def rescale_intensity(frame):
    cropped_frame = frame[70:]
    imin, imax = np.min(cropped_frame), np.max(cropped_frame)
    return np.clip((frame - imin) / (imax - imin), 0, 1)


def preprocess_frame(frame):
    frame = skimage.color.rgb2gray(frame)
    return rescale_intensity(frame) > 0.5


def get_frame_center(frame):
    return frame[160:320, 300:468]


def find_ship_contour(frame):
    center = get_frame_center(frame).copy()
    _, contours, _ = cv2.findContours(
        center,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
        offset=(300, 160),
    )

    for contour in contours:
        approx = cv2.approxPolyDP(
            contour,
            0.07 * cv2.arcLength(contour, True),
            True,
        )

        if len(approx) == 3:
            return contour


def get_distance_to_nearest_wall(frame):
    frame = preprocess_frame(frame)
    center = np.array([480, 768]) / 2
    ship_contour = find_ship_contour(img_as_ubyte(frame))

    if ship_contour is None:
        return None

    ship_coords = np.mean(ship_contour, axis=0)[0][::-1]
    ship_angle = math.atan2(
        ship_coords[0] - center[0],
        ship_coords[1] - center[1],
    )

    ray = []
    for i in range(380):
        y = int(math.sin(ship_angle) * i + ship_coords[0])
        x = int(math.cos(ship_angle) * i + ship_coords[1])

        if y < 0 or y >= 480 or x < 0 or x >= 768:
            break

        ray.append(frame[y, x])

    chunks = [list(g) for k, g in itertools.groupby(ray)]

    if len(chunks) >= 1:
        # This means we've crashed or are about to crash.
        if len(chunks[0]) > 8:
            return 0

        return len(chunks[1])
