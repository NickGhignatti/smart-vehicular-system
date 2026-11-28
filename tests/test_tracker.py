import dataclasses
import math

import numpy as np
import pytest

from svs.config import FusionConfig, RadarPerceptionConfig
from svs.fusion.measurements import GapMeasurement, from_camera, from_radar
from svs.fusion.tracker import LeadTracker, RawEstimator, make_estimator
from svs.perception.camera_tracker import CameraMeasurement
from svs.perception.radar import RadarMeasurement

DT = 0.05
CFG = FusionConfig()


def cam(gap, sigma=1.0, lateral=0.0):
    return GapMeasurement("camera", gap, sigma, lateral)


def radar(gap, rate, sigma=0.3, sigma_rate=0.2, lateral=0.0):
    return GapMeasurement("radar", gap, sigma, lateral, rate, sigma_rate)


# --- measurement conversion ------------------------------------------------------------

def test_from_radar_shifts_to_bumper_and_models_noise():
    cfg = RadarPerceptionConfig(sigma_range_m=0.1, sigma_range_per_m=0.01, sigma_rate_mps=0.2, min_points=2)
    m = from_radar(RadarMeasurement(range_m=20.4, range_rate_mps=-1.0, lateral_m=0.3, n_points=5), 0.4, cfg)
    assert (m.source, m.gap_m, m.rate_mps, m.lateral_m) == ("radar", pytest.approx(20.0), -1.0, 0.3)
    assert m.sigma_gap_m == pytest.approx(0.1 + 0.01 * 20.4)
    assert from_radar(RadarMeasurement(20.0, -1.0, 0.0, n_points=1), 0.4, cfg) is None
    assert from_radar(None, 0.4, cfg) is None


def test_from_camera_scales_sigma():
    m = CameraMeasurement(1, 0.0, (0, 0, 1, 1), "car", 1.0, gap_m=12.0, sigma_m=0.5,
                          azimuth_rad=0.0, lateral_m=-0.2, method="ground")
    g = from_camera(m, sigma_scale=2.0)
    assert (g.source, g.gap_m, g.sigma_gap_m, g.lateral_m, g.rate_mps) == ("camera", 12.0, 1.0, -0.2, None)


# --- track life cycle --------------------------------------------------------------------

def test_track_is_tentative_then_confirmed():
    t = LeadTracker(CFG, "radar")
    assert t.step(DT, radar=radar(30, -1)).status == "tentative"
    assert t.step(DT, radar=radar(30, -1)).status == "tentative"
    est = t.step(DT, radar=radar(30, -1))
    assert est.status == "confirmed" and est.confirmed


def test_coasting_then_dropped():
    t = LeadTracker(CFG, "radar")
    for _ in range(5):
        t.step(DT, radar=radar(30, -2))
    est = t.step(DT)
    assert est.status == "coasting"
    assert est.gap_m < 30                       # still predicting the approach
    n = int(CFG.max_coast_s / DT)
    for _ in range(n):
        est = t.step(DT)
    assert est.status == "none" and not est.valid


def test_ghost_is_gated_out():
    t = LeadTracker(CFG, "radar")
    for _ in range(10):
        t.step(DT, radar=radar(30, 0))
    est = t.step(DT, radar=radar(12, -8))       # stationary false target much closer
    assert est.rejected == ("radar",)
    assert est.gap_m == pytest.approx(30, abs=0.2)


def test_reinitializes_when_the_lead_changes():
    t = LeadTracker(CFG, "radar")
    for _ in range(10):
        t.step(DT, radar=radar(30, 0))
    for _ in range(CFG.reinit_after + 4):       # a car cut in at 12 m and stays
        est = t.step(DT, radar=radar(12, 0))
    assert est.gap_m == pytest.approx(12, abs=0.3)


def test_mode_ignores_other_sensor():
    t = LeadTracker(CFG, "camera")
    assert t.step(DT, radar=radar(30, 0)).status == "none"
    assert t.step(DT, camera=cam(25), radar=radar(30, 0)).updated_by == ("camera",)


def test_fused_needs_the_camera_to_confirm():
    t = LeadTracker(CFG, "fused")
    for _ in range(10):
        est = t.step(DT, radar=radar(15, -8))   # radar-only object (e.g. ghost or clutter)
    assert est.status == "tentative" and not est.confirmed
    est = t.step(DT, camera=cam(est.gap_m - 8 * DT), radar=radar(est.gap_m - 8 * DT, -8))
    assert est.status == "confirmed"


def test_fused_continues_on_radar_during_camera_occlusion():
    t = LeadTracker(CFG, "fused")
    for _ in range(5):
        t.step(DT, camera=cam(20), radar=radar(20, 0))
    for _ in range(40):                         # 2 s without camera
        est = t.step(DT, radar=radar(20, 0))
    assert est.status == "confirmed" and est.updated_by == ("radar",)


def test_fused_init_trusts_camera_when_sensors_disagree():
    t = LeadTracker(CFG, "fused")
    est = t.step(DT, camera=cam(25, lateral=0.0), radar=radar(10, -5, lateral=0.2))
    assert est.updated_by == ("camera",) and est.gap_m == pytest.approx(25)


def test_fused_beats_each_single_sensor():
    """Lead braking and accelerating (stop-and-go); fused gap RMSE <= best single sensor."""
    rng = np.random.default_rng(3)
    trackers = {mode: LeadTracker(CFG, mode) for mode in ("camera", "radar", "fused")}
    sq = {mode: [] for mode in trackers}
    gap, rate = 25.0, 0.0
    for k in range(1200):
        accel = 1.5 * math.sin(2 * math.pi * k * DT / 12.0)
        rate += accel * DT
        gap += rate * DT
        c = cam(gap + rng.normal(0, 1.0), sigma=1.0)
        r = radar(gap + rng.normal(0, 0.3), rate + rng.normal(0, 0.2)) if rng.random() > 0.1 else None
        for mode, t in trackers.items():
            est = t.step(DT, camera=c, radar=r)
            if k > 100 and est.valid:
                sq[mode].append((est.gap_m - gap) ** 2)
    rmse = {mode: math.sqrt(np.mean(v)) for mode, v in sq.items()}
    assert rmse["fused"] <= rmse["radar"] * 1.02
    assert rmse["fused"] < rmse["camera"]


# --- raw baselines -------------------------------------------------------------------------

def test_raw_camera_differentiates_gap():
    r = RawEstimator("camera")
    assert r.step(DT, camera=cam(20.0)).gap_rate_mps is None
    est = r.step(DT, camera=cam(19.9))
    assert est.gap_rate_mps == pytest.approx(-2.0)
    r.step(DT)                                   # one missing frame
    assert r.step(DT, camera=cam(19.7)).gap_rate_mps == pytest.approx(-2.0)


def test_raw_radar_passes_measurement_through():
    est = make_estimator("radar_raw", CFG).step(DT, radar=radar(18.0, -1.5))
    assert (est.gap_m, est.gap_rate_mps, est.confirmed) == (18.0, -1.5, True)
    assert not make_estimator("radar_raw", CFG).step(DT).valid


def test_invalid_mode():
    with pytest.raises(ValueError):
        LeadTracker(CFG, "lidar")
    assert isinstance(make_estimator("fused", dataclasses.replace(CFG)), LeadTracker)
