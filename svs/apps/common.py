"""Helpers shared by the CLI apps: ego speed control, HUD drawing, video, mount geometry."""

import logging
import math

import carla
import cv2
import numpy as np

log = logging.getLogger(__name__)


def speed_control(speed_mps, target_kmh):
    """Tiny P controller (Lab3 cruise logic) to drive at a constant speed; target 0 = hold brake."""
    if target_kmh <= 0.0:
        return carla.VehicleControl(throttle=0.0, brake=1.0)
    error = target_kmh / 3.6 - speed_mps
    if error > 0:
        return carla.VehicleControl(throttle=float(np.clip(0.3 + 0.15 * error, 0.0, 0.6)), brake=0.0)
    return carla.VehicleControl(throttle=0.0, brake=float(np.clip(-0.1 * error, 0.0, 0.3)))


def fmt(value, spec):
    return "--" if value is None or not math.isfinite(value) else format(value, spec)


def draw_hud(frame_bgr, lines):
    """Text block in the top-left corner, outlined so it stays readable on any background (Lab4 HUD)."""
    for i, text in enumerate(lines):
        y = 26 + 24 * i
        cv2.putText(frame_bgr, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(frame_bgr, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)


def open_video_writer(path, fps, width, height):
    path.parent.mkdir(parents=True, exist_ok=True)
    log.info("Recording to %s", path)
    return cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))


def camera_height_above_road(world, camera):
    """Measured camera height over the road surface (call after at least one tick)."""
    location = camera.get_transform().location
    waypoint = world.get_map().get_waypoint(location, project_to_road=True)
    return location.z - waypoint.transform.location.z


def bumper_offset(ego, sensor_x):
    """How far the ego front bumper is ahead of a sensor mounted at x=sensor_x."""
    bb = ego.bounding_box
    return bb.location.x + bb.extent.x - sensor_x
