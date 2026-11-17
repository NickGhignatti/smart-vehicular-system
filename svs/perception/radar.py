"""
Utilities for radar perception.
"""

from dataclasses import dataclass

import numpy as np

from svs.sim.sensors import ALTITUDE, AZIMUTH, DEPTH, VELOCITY

# Default gating limits, in meters, in the sensor frame (the radar is mounted ~1 m above ground).
MIN_RANGE_M = 1.0
CORRIDOR_HALF_WIDTH_M = 1.5
MIN_HEIGHT_M = -0.8  # below this the detection is most likely the road surface
DEFAULT_PERCENTILE = 10
CLUSTER_DEPTH_M = 3.0


@dataclass
class RadarMeasurement:
    """
    One reading of the lead vehicle, in the radar frame (x forward, y right, metres).

    Attributes:
        range_m: distance along x to the nearest return. Not the ray length: a lead at 5 deg
            off-axis is cos(5 deg) closer along x than along the ray.
        range_rate_mps: change of range_m over time. Negative = closing in, positive = opening.
            The sign is the opposite of CARLA's radar velocity (positive towards the sensor).
        lateral_m: y of the lead, positive to the right of the radar axis. Radar angular
            resolution is coarse and the returns are not the vehicle centre, so expect it noisy.
        n_points: detections used for the reading. More points, more trust.
    """

    range_m: float
    range_rate_mps: float
    lateral_m: float
    n_points: int

def extract_radar_measurements(points, percentile=DEFAULT_PERCENTILE) -> RadarMeasurement | None:
    """
    Summarises the vehicle in front of the radar as one RadarMeasurement.

    Args:
        points: (N, 4) array [depth, azimuth, altitude, velocity], as radar_to_array() gives.
        percentile: which percentile of x counts as the range. Low values ignore ghosts closer
            than the vehicle but follow its nearest edge.

    Returns:
        None when no detection is inside the front corridor, so the caller can skip the update.

    Steps: keep the detections inside the front corridor, keep the ones within CLUSTER_DEPTH_M
    of the nearest return (the lead only), then take the median velocity and y.

    Limits:
        - Doppler gives the speed along the ray. Dividing by cos(azimuth) * cos(altitude) gives
          the speed along x only if the relative velocity points along x. A lead that moves
          sideways (cut-in) is read wrong.
        - If most corridor points belong to a far vehicle, the percentile lands on it and the
          far vehicle is read instead of the lead.
    """

    if len(points) == 0:
        return None

    cartesian: np.ndarray = _to_cartesian(points)
    mask: np.ndarray = _front_corridor_mask(cartesian)
    front_xyz: np.ndarray = cartesian[mask]
    front_points: np.ndarray = points[mask]

    if len(front_points) == 0:
        return None

    vehicle_mask: np.ndarray = _nearest_vehicle_mask(front_xyz, percentile)
    front_xyz = front_xyz[vehicle_mask]
    front_points = front_points[vehicle_mask]

    # Radial velocity -> longitudinal one.
    longitudinal_velocity = front_points[:, VELOCITY] / (
        np.cos(front_points[:, ALTITUDE]) * np.cos(front_points[:, AZIMUTH]))

    return RadarMeasurement(
        range_m=float(np.percentile(front_xyz[:, 0], percentile)),
        range_rate_mps=float(-np.median(longitudinal_velocity)),
        lateral_m=float(np.median(front_xyz[:, 1])),
        n_points=len(front_points),
    )

def _to_cartesian(points) -> np.ndarray:
    """
    Converts radar detections from polar to cartesian coordinates.
    """
    depth = points[:, DEPTH]
    azimuth = points[:, AZIMUTH]
    altitude = points[:, ALTITUDE]

    horizontal = depth * np.cos(altitude)
    return np.column_stack((
        horizontal * np.cos(azimuth),
        horizontal * np.sin(azimuth),
        depth * np.sin(altitude),
    ))


def _front_corridor_mask(xyz, min_range_m=MIN_RANGE_M, half_width_m=CORRIDOR_HALF_WIDTH_M,
                         min_height_m=MIN_HEIGHT_M) -> np.ndarray:
    """
    Boolean mask of the detections inside the corridor in front of the ego.
    """
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    return (x > min_range_m) & (np.abs(y) < half_width_m) & (z > min_height_m)


def _nearest_vehicle_mask(xyz, percentile, depth_m=CLUSTER_DEPTH_M) -> np.ndarray:
    """
    Boolean mask of the detections within depth_m behind the nearest return, i.e. the lead only.
    """
    x = xyz[:, 0]
    return x < np.percentile(x, percentile) + depth_m
