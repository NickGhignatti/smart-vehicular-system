"""Front camera and radar: spawning, raw-data conversion and frame-synchronized reads."""

import logging
import math
import queue

import carla
import numpy as np

log = logging.getLogger(__name__)

# Column layout of radar_to_array().
DEPTH, AZIMUTH, ALTITUDE, VELOCITY = range(4)


def _set_attributes(bp, **attributes):
    for key, value in attributes.items():
        if bp.has_attribute(key):
            bp.set_attribute(key, str(value))
        else:
            log.warning("Blueprint %s has no attribute %s", bp.id, key)


def spawn_camera(world, parent, cfg, role_name="svs_front_camera"):
    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    _set_attributes(bp, image_size_x=cfg.width, image_size_y=cfg.height, fov=cfg.fov,
                    sensor_tick=cfg.sensor_tick, role_name=role_name)
    transform = carla.Transform(carla.Location(x=cfg.x, z=cfg.z), carla.Rotation(pitch=cfg.pitch))
    return world.spawn_actor(bp, transform, attach_to=parent)


def spawn_radar(world, parent, cfg, role_name="svs_front_radar"):
    bp = world.get_blueprint_library().find("sensor.other.radar")
    _set_attributes(bp, horizontal_fov=cfg.horizontal_fov, vertical_fov=cfg.vertical_fov,
                    points_per_second=cfg.points_per_second, range=cfg.range,
                    sensor_tick=cfg.sensor_tick, role_name=role_name)
    transform = carla.Transform(carla.Location(x=cfg.x, z=cfg.z))
    return world.spawn_actor(bp, transform, attach_to=parent)


def camera_intrinsics(cfg):
    """Pinhole matrix K from image size and horizontal FOV (CARLA cameras have square pixels)."""
    focal = cfg.width / (2.0 * math.tan(math.radians(cfg.fov) / 2.0))
    return np.array([
        [focal, 0.0, cfg.width / 2.0],
        [0.0, focal, cfg.height / 2.0],
        [0.0, 0.0, 1.0],
    ])


def image_to_bgr(image):
    """carla.Image (BGRA bytes) -> HxWx3 uint8 BGR array for OpenCV."""
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
    return arr[:, :, :3].copy()


def radar_to_array(measurement):
    """carla.RadarMeasurement -> (N, 4) float array [depth, azimuth, altitude, velocity].

    Built from the detection attributes rather than raw_data so the column order is explicit.
    Angles are in radians, depth in meters, velocity is the radial relative speed in m/s.
    """
    return np.array(
        [(d.depth, d.azimuth, d.altitude, d.velocity) for d in measurement],
        dtype=np.float64,
    ).reshape(-1, 4)


class SensorSync:
    """Pairs sensor messages with the world frame that produced them.

    The labs keep the "latest" message from each callback, which is fine locally but
    over a remote connection messages arrive late and out of step with the simulation.
    Here every callback pushes into its own queue, and after each world.tick() we read
    until the message stamped with that exact frame shows up.
    """

    def __init__(self, timeout):
        self.timeout = timeout
        self._queues = {}
        self._pending = {}  # message from a future frame, held back for a later get()

    def add(self, name, sensor):
        q = queue.Queue()
        self._queues[name] = q
        self._pending[name] = None
        sensor.listen(q.put)

    def get(self, frame):
        """Return {name: data or None} for `frame`; None means it did not arrive in time."""
        return {name: self._get_one(name, frame) for name in self._queues}

    def _get_one(self, name, frame):
        while True:
            data, self._pending[name] = self._pending[name], None
            if data is None:
                try:
                    data = self._queues[name].get(timeout=self.timeout)
                except queue.Empty:
                    log.warning("Frame %d: no %s data within %.1fs", frame, name, self.timeout)
                    return None
            if data.frame == frame:
                return data
            if data.frame > frame:
                self._pending[name] = data
                return None
            # Older than the requested frame: stale, drop it.
