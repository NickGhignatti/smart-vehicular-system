"""Synthetic scenes for offline tests: a car is a 3D box standing on the road."""

import itertools

import numpy as np

from svs.perception.camera_geometry import PinholeCamera
from svs.perception.detector import Detection

# The project camera: 640x360, FOV 60, Lab4 mount (1.4 m high, pitch -6).
CAMERA = PinholeCamera(width=640, height=360, fov_deg=60.0, mount_height_m=1.4, pitch_deg=-6.0)


def car_box(camera, distance_m, lateral_m=0.0, width_m=1.85, height_m=1.5, length_m=4.5, clip=True):
    """Pixel box of a car whose rear face is distance_m ahead of the camera (horizontally).

    The box sits flat on the road, so the bottom edge of its image is exactly the rear face:
    tests use bottom_to_rear_m=0 to match.
    """
    xs = (distance_m, distance_m + length_m)
    ys = (lateral_m - width_m / 2, lateral_m + width_m / 2)
    zs = (-camera.mount_height_m, -camera.mount_height_m + height_m)
    corners = np.array(list(itertools.product(xs, ys, zs)))
    uv = camera.project(corners)
    x1, y1 = uv.min(axis=0)
    x2, y2 = uv.max(axis=0)
    if clip:
        x1, x2 = np.clip([x1, x2], 0, camera.width)
        y1, y2 = np.clip([y1, y2], 0, camera.height)
    return (float(x1), float(y1), float(x2), float(y2))


def car_detection(camera, distance_m, lateral_m=0.0, noise_px=0.0, rng=None, **kwargs):
    box = np.array(car_box(camera, distance_m, lateral_m, **kwargs))
    if noise_px > 0:
        box = box + rng.normal(0.0, noise_px, size=4)
    return Detection(tuple(float(c) for c in box), 1.0, "car")
