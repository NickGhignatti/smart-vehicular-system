"""Front camera -> lead-vehicle measurement (gap, uncertainty, bearing).

Camera counterpart of Lab5's FrontRadarTracker: call `update()` once per frame and read
the result from `measurement`. Unlike Lab4/Lab5 there is no exponential smoothing here:
the tracker reports raw per-frame measurements with a 1-sigma uncertainty, and the
Kalman filter (svs.fusion) does the smoothing using that uncertainty.
"""

import math
from dataclasses import dataclass

from svs.perception.lead_selector import LeadSelector


@dataclass(frozen=True)
class CameraMeasurement:
    frame: int
    timestamp: float
    box: tuple
    label: str
    score: float
    gap_m: float          # bumper-to-bumper, like svs.sim.ground_truth.LeadTruth.gap_m
    sigma_m: float        # 1-sigma uncertainty of gap_m, used as Kalman measurement noise
    azimuth_rad: float    # bearing of the box center, for association with the radar
    method: str           # "ground", "width" or "ground+width"


def estimate_distance(det, camera, cfg):
    """Monocular distance from the camera to the lead's rear, with 1-sigma uncertainty.

    Two independent estimates, combined by inverse-variance weighting:
    - ground plane: needs the bottom edge of the box (the rear tires) inside the image;
    - known width: needs both side edges inside the image (not cut off by the border).
    Returns (distance_m, sigma_m, method) or None when neither is usable.
    """
    estimates = []

    if det.box[3] < camera.height - cfg.border_px:
        d = camera.ground_distance(det.bottom_v)
        if d is not None:
            sigma = camera.ground_distance_sigma(det.bottom_v, cfg.sigma_px)
            # The bottom of the box is the rear tires; the rear bumper is closer by the overhang.
            estimates.append(("ground", d - cfg.bottom_to_rear_m, sigma))

    if det.box[0] > cfg.border_px and det.box[2] < camera.width - cfg.border_px:
        d = camera.width_distance(det.width, cfg.car_width_m)
        if d is not None:
            sigma = camera.width_distance_sigma(d, det.width, cfg.sigma_px, cfg.sigma_car_width_m, cfg.car_width_m)
            estimates.append(("width", d, sigma))

    if not estimates:
        return None

    weights = [1.0 / s ** 2 for _, _, s in estimates]
    distance = sum(w * d for w, (_, d, _) in zip(weights, estimates)) / sum(weights)
    sigma = math.sqrt(1.0 / sum(weights))
    method = "+".join(name for name, _, _ in estimates)
    return distance, sigma, method


class FrontCameraTracker:
    def __init__(self, camera, detector, cfg, bumper_offset_m):
        """
        camera: PinholeCamera; detector: see svs.perception.detector; cfg: CameraPerceptionConfig;
        bumper_offset_m: how far the ego front bumper is ahead of the camera.
        """
        self.camera = camera
        self.detector = detector
        self.cfg = cfg
        self.bumper_offset_m = bumper_offset_m
        self.selector = LeadSelector(
            camera, cfg.lane_half_width_m, cfg.lane_margin_m, cfg.max_misses,
            cfg.car_width_m, cfg.border_px, cfg.min_box_px,
        )

        self.detections = []
        self.lead = None
        self.measurement = None

    def update(self, frame, timestamp, images):
        self.detections = self.detector.detect(images)
        self.lead = self.selector.update(self.detections)
        self.measurement = None
        if self.lead is None:
            return None

        estimate = estimate_distance(self.lead, self.camera, self.cfg)
        if estimate is None:
            return None
        distance, sigma, method = estimate

        self.measurement = CameraMeasurement(
            frame=frame,
            timestamp=timestamp,
            box=self.lead.box,
            label=self.lead.label,
            score=self.lead.score,
            gap_m=distance - self.bumper_offset_m,
            sigma_m=sigma,
            azimuth_rad=self.camera.bearing(self.lead.center_u),
            method=method,
        )
        return self.measurement
