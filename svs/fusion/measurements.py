"""Camera and radar readings expressed in one frame, ready for the Kalman filter.

Common reference: the gap is bumper to bumper (ego front bumper -> lead rear), as the
ground truth. The radar measures from its own mount (RadarMeasurement.range_m), so its
range is shifted by the distance between the radar and the ego front bumper. Lateral
positions are relative to the ego centerline, positive to the right, for both sensors
(both are mounted on the centerline).
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class GapMeasurement:
    source: str                        # "camera" or "radar"
    gap_m: float
    sigma_gap_m: float
    lateral_m: float
    rate_mps: Optional[float] = None   # radar only: d(gap)/dt, negative = closing in
    sigma_rate_mps: Optional[float] = None


def from_camera(m, sigma_scale=1.0):
    """CameraMeasurement -> GapMeasurement (already bumper-to-bumper)."""
    if m is None:
        return None
    return GapMeasurement("camera", m.gap_m, m.sigma_m * sigma_scale, m.lateral_m)


def from_radar(m, bumper_offset_m, cfg, sigma_scale=1.0):
    """RadarMeasurement -> GapMeasurement.

    bumper_offset_m: how far the ego front bumper is ahead of the radar.
    cfg: RadarPerceptionConfig (noise model and minimum number of points).
    """
    if m is None or m.n_points < cfg.min_points:
        return None
    sigma_gap = (cfg.sigma_range_m + cfg.sigma_range_per_m * m.range_m) * sigma_scale
    return GapMeasurement(
        "radar",
        gap_m=m.range_m - bumper_offset_m - cfg.range_bias_m,
        sigma_gap_m=sigma_gap,
        lateral_m=m.lateral_m,
        rate_mps=m.range_rate_mps,
        sigma_rate_mps=cfg.sigma_rate_mps * sigma_scale,
    )
