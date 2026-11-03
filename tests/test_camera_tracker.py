import dataclasses

import numpy as np
import pytest

from svs.config import CameraPerceptionConfig
from svs.perception.camera_tracker import FrontCameraTracker, estimate_distance
from svs.perception.detector import Detection
from tests.synthetic import CAMERA, car_detection

# Synthetic cars are boxes flat on the road: their bottom edge is the rear face.
CFG = CameraPerceptionConfig(bottom_to_rear_m=0.0)


class FakeDetector:
    required_images = ("rgb",)

    def __init__(self, detections):
        self.detections = detections

    def detect(self, images):
        return self.detections


@pytest.mark.parametrize("distance", [5.0, 15.0, 30.0])
def test_clean_box_uses_both_methods_and_is_accurate(distance):
    d, sigma, method = estimate_distance(car_detection(CAMERA, distance), CAMERA, CFG)
    assert method == "ground+width"
    assert abs(d - distance) < 0.02 * distance
    assert sigma > 0


def test_combined_sigma_is_smaller_than_each_method():
    det = car_detection(CAMERA, 20.0)
    _, both, _ = estimate_distance(det, CAMERA, CFG)
    # Push the box onto the left border -> width unusable -> ground only.
    _, ground_only, method = estimate_distance(
        Detection((0.0,) + det.box[1:], 1.0, "car"), CAMERA, CFG)
    assert method == "ground"
    assert both < ground_only


def test_bottom_cut_off_falls_back_to_width():
    # At 2.5 m the rear wheels are below the image (the road is visible from ~3.2 m),
    # but the car (1.85 m) still fits in the 60 deg width (+-1.44 m at 2.5 m).
    assert CAMERA.min_ground_distance() > 2.5
    det = car_detection(CAMERA, 2.5)
    assert det.box[3] == CAMERA.height and 0 < det.box[0] and det.box[2] < CAMERA.width
    d, _, method = estimate_distance(det, CAMERA, CFG)
    assert method == "width"
    assert d == pytest.approx(2.5, rel=0.05)


def test_box_cut_on_side_and_bottom_gives_no_estimate():
    det = Detection((0.0, 200.0, 640.0, 360.0), 1.0, "car")
    assert estimate_distance(det, CAMERA, CFG) is None


def test_bottom_to_rear_correction_shifts_ground_estimate():
    det = Detection((0.0,) + car_detection(CAMERA, 20.0).box[1:], 1.0, "car")  # ground only
    d0, _, _ = estimate_distance(det, CAMERA, CFG)
    d1, _, _ = estimate_distance(det, CAMERA, dataclasses.replace(CFG, bottom_to_rear_m=0.9))
    assert d0 - d1 == pytest.approx(0.9)


@pytest.mark.parametrize("distance", [10.0, 25.0])
def test_reported_sigma_matches_actual_spread(distance):
    """The Kalman filter trusts sigma_m, so it must describe the real error spread."""
    rng = np.random.default_rng(1)
    cfg = dataclasses.replace(CFG, sigma_px=1.5, sigma_car_width_m=0.0)
    results = [estimate_distance(car_detection(CAMERA, distance, noise_px=1.5, rng=rng), CAMERA, cfg)
               for _ in range(3000)]
    estimates = np.array([r[0] for r in results])
    reported = np.median([r[1] for r in results])
    assert abs(estimates.mean() - distance) < 0.05 * distance
    assert 0.7 < estimates.std() / reported < 1.4


def test_tracker_outputs_bumper_gap_and_bearing():
    lead = car_detection(CAMERA, 20.0, lateral_m=0.5)
    tracker = FrontCameraTracker(CAMERA, FakeDetector([lead]), CFG, bumper_offset_m=0.6)
    m = tracker.update(frame=7, timestamp=1.25, images={})
    assert (m.frame, m.timestamp, m.box) == (7, 1.25, lead.box)
    assert m.gap_m == pytest.approx(20.0 - 0.6, rel=0.02)
    assert m.azimuth_rad > 0   # to the right
    assert tracker.measurement is m and tracker.lead == lead


def test_tracker_without_lead_returns_none():
    tracker = FrontCameraTracker(CAMERA, FakeDetector([]), CFG, bumper_offset_m=0.6)
    assert tracker.update(0, 0.0, {}) is None
    assert tracker.measurement is None and tracker.lead is None
