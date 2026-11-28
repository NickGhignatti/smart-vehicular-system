"""Pinhole model of the front camera: projection and monocular distance.

Pure numpy, no CARLA import, so every formula here is unit-tested offline.

Frames follow CARLA (Lab2/Lab4): x forward, y right, z up, angles in degrees with
negative pitch = looking down. Points are expressed relative to the camera position,
with axes aligned to the vehicle heading (the camera is only pitched, not yawed/rolled).
"""

import math
from dataclasses import dataclass

import numpy as np


def focal_length_px(width_px, fov_deg):
    """Focal length in pixels from image width and horizontal FOV (CARLA cameras have square pixels)."""
    return width_px / (2.0 * math.tan(math.radians(fov_deg) / 2.0))


@dataclass(frozen=True)
class PinholeCamera:
    width: int
    height: int
    fov_deg: float
    mount_height_m: float  # camera height above the road
    pitch_deg: float = 0.0

    @classmethod
    def from_config(cls, cfg, mount_height_m=None):
        """Build from CameraConfig; pass the measured height above the road if known."""
        height = cfg.z if mount_height_m is None else mount_height_m
        return cls(cfg.width, cfg.height, cfg.fov, height, cfg.pitch)

    @property
    def f(self):
        return focal_length_px(self.width, self.fov_deg)

    @property
    def cx(self):
        return self.width / 2.0

    @property
    def cy(self):
        return self.height / 2.0

    @property
    def pitch(self):
        return math.radians(self.pitch_deg)

    @property
    def K(self):
        return np.array([[self.f, 0.0, self.cx], [0.0, self.f, self.cy], [0.0, 0.0, 1.0]])

    def project(self, points):
        """(N, 3) points [x fwd, y right, z up] relative to the camera -> (N, 2) pixels [u, v].

        Points behind the camera come back as NaN.
        """
        p = np.atleast_2d(np.asarray(points, dtype=float))
        c, s = math.cos(self.pitch), math.sin(self.pitch)
        depth = p[:, 0] * c + p[:, 2] * s     # along the optical axis
        up = -p[:, 0] * s + p[:, 2] * c       # camera "up" axis
        with np.errstate(divide="ignore", invalid="ignore"):
            u = self.cx + self.f * p[:, 1] / depth
            v = self.cy - self.f * up / depth
        behind = depth <= 1e-6
        u[behind] = np.nan
        v[behind] = np.nan
        return np.stack([u, v], axis=1)

    # --- ground-plane method: the bottom of the box touches the road -------------------

    def angle_below_horizon(self, v):
        return math.atan((v - self.cy) / self.f) - self.pitch

    def ground_distance(self, v):
        """Horizontal distance to the road point seen at image row v (None above the horizon)."""
        alpha = self.angle_below_horizon(v)
        if alpha <= 1e-3:
            return None
        return self.mount_height_m / math.tan(alpha)

    def ground_distance_sigma(self, v, sigma_px):
        """1-sigma distance error caused by sigma_px pixels of error on row v.

        d = h / tan(alpha)  ->  |dd/dv| = h / sin^2(alpha) * dalpha/dv, which grows ~ d^2.
        """
        alpha = self.angle_below_horizon(v)
        if alpha <= 1e-3:
            return None
        t = (v - self.cy) / self.f
        dalpha_dv = 1.0 / (self.f * (1.0 + t * t))
        return self.mount_height_m / math.sin(alpha) ** 2 * dalpha_dv * sigma_px

    def min_ground_distance(self):
        """Closest road point still inside the image (bottom pixel row)."""
        return self.ground_distance(self.height - 0.5)

    # --- known-width method: a car is ~1.85 m wide -------------------------------------

    def width_distance(self, width_px, object_width_m):
        """Distance from apparent width. Exact along the optical axis; the small pitch is corrected."""
        if width_px <= 0:
            return None
        return self.f * object_width_m / width_px / math.cos(self.pitch)

    def width_distance_sigma(self, distance_m, width_px, sigma_px, sigma_width_m, object_width_m):
        """Relative errors add in quadrature: pixel error on both edges + unknown true width."""
        rel_px = math.sqrt(2.0) * sigma_px / width_px
        rel_w = sigma_width_m / object_width_m
        return distance_m * math.sqrt(rel_px ** 2 + rel_w ** 2)

    # --- lateral geometry ----------------------------------------------------------------

    def bearing(self, u):
        """Azimuth of image column u in radians, positive to the right (same sign as CARLA radar azimuth)."""
        return math.atan((u - self.cx) / self.f)

    def lateral_offset(self, u, distance_m):
        """Lateral position in meters of something seen at column u, distance_m ahead."""
        return distance_m * (u - self.cx) / self.f
