"""Realistic degradation of CARLA radar detections.

CARLA's radar returns exact depth and Doppler velocity for every ray hit. Real automotive
radars are noisier, lose frames, and report false targets. Without this, "radar-only" would
be a perfect sensor and the fusion study would be meaningless. All effects are seeded and
configured in RadarPerceptionConfig, and applied identically in live runs and replays.

Velocity sign: points use CARLA's convention (positive = moving towards the sensor), the
same one svs.perception.radar.extract_radar_measurements assumes.
"""

import numpy as np

from svs.sim.sensors import ALTITUDE, AZIMUTH, DEPTH, VELOCITY


class RadarDegrader:
    def __init__(self, cfg, rng=None):
        self.cfg = cfg
        self.rng = rng if rng is not None else np.random.default_rng()
        self._ghost = None  # (x_m, y_m, z_m, remaining_s) in the radar frame

    def apply(self, points, ego_speed_mps, dt):
        """Return a degraded copy of `points` ((N, 4) [depth, azimuth, altitude, velocity])."""
        cfg = self.cfg
        self._advance_ghost(ego_speed_mps, dt)

        if self.rng.random() < cfg.frame_dropout:
            return np.empty((0, 4))

        out = np.array(points, dtype=float).reshape(-1, 4)
        if len(out):
            out[:, DEPTH] = np.maximum(0.1, out[:, DEPTH] + self.rng.normal(0, cfg.point_depth_noise_m, len(out)))
            out[:, VELOCITY] += self.rng.normal(0, cfg.point_velocity_noise_mps, len(out))

        if self._ghost is not None:
            out = np.vstack([out, self._ghost_points(ego_speed_mps)])
        return out

    def _advance_ghost(self, ego_speed_mps, dt):
        cfg = self.cfg
        if self._ghost is not None:
            x, y, z, remaining = self._ghost
            x -= ego_speed_mps * dt  # stationary object: it comes towards us at our speed
            remaining -= dt
            self._ghost = (x, y, z, remaining) if remaining > 0 and x > 1.0 else None
        elif cfg.ghosts_per_minute > 0 and self.rng.random() < cfg.ghosts_per_minute / 60.0 * dt:
            # Somewhere in the lane ahead, near the road surface.
            x = self.rng.uniform(8.0, 40.0)
            y = self.rng.uniform(-1.0, 1.0)
            z = self.rng.uniform(-0.6, 0.0)
            self._ghost = (x, y, z, cfg.ghost_duration_s)

    def _ghost_points(self, ego_speed_mps):
        x, y, z, _ = self._ghost
        n = int(self.rng.integers(2, 5))
        xs = x + self.rng.normal(0, 0.1, n)
        ys = y + self.rng.normal(0, 0.1, n)
        zs = np.full(n, z)
        depth = np.sqrt(xs ** 2 + ys ** 2 + zs ** 2)
        azimuth = np.arctan2(ys, xs)
        altitude = np.arcsin(zs / depth)
        # Stationary object: radial speed towards the sensor = ego speed projected on the ray.
        velocity = ego_speed_mps * np.cos(azimuth) * np.cos(altitude)
        velocity += self.rng.normal(0, self.cfg.point_velocity_noise_mps, n)
        ghost = np.empty((n, 4))
        ghost[:, DEPTH], ghost[:, AZIMUTH], ghost[:, ALTITUDE], ghost[:, VELOCITY] = depth, azimuth, altitude, velocity
        return ghost
