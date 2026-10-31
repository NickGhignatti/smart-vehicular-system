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


@dataclass
class RadarMeasurement:
    range_m: float
    range_rate_mps: float
    lateral_m: float
    n_points: int

def extract_radar_measurements(points, percentile=DEFAULT_PERCENTILE) -> RadarMeasurement | None:
    """
    Extracts radar measurements from a set of points.
    """

    if len(points) == 0:
        return None

    cartesian: np.ndarray = _to_cartesian(points)
    mask: np.ndarray = _front_corridor_mask(cartesian)
    front_xyz: np.ndarray = cartesian[mask]
    front_points: np.ndarray = points[mask]
    num_points: int = len(front_points)

    if num_points == 0:
        return None

    # Radial velocity -> longitudinal one.
    longitudinal_velocity = front_points[:, VELOCITY] / (
        np.cos(front_points[:, ALTITUDE]) * np.cos(front_points[:, AZIMUTH]))

    return RadarMeasurement(
        range_m=float(np.percentile(front_xyz[:, 0], percentile)),
        range_rate_mps=float(-np.median(longitudinal_velocity)),
        lateral_m=float(np.median(front_xyz[:, 1])),
        n_points=num_points,
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
