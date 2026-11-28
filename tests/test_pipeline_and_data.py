"""Recording format, noise models, scenarios and the shared pipeline."""

import dataclasses
import math

import numpy as np
import pytest

from svs.calibration import fit_linear_sigma, robust_stats, sigma_scale, sign_agreement
from svs.config import RadarPerceptionConfig, load_config, with_overrides
from svs.control.speed import AccelToSpeed, SpeedController
from svs.data.recording import FrameData, Recording, RecordingMeta
from svs.perception.detector import Detection, ReplayDetector, perturb_detections
from svs.perception.occlusion import apply_camera_events
from svs.perception.radar import extract_radar_measurements
from svs.perception.radar_noise import RadarDegrader
from svs.pipeline import Pipeline, replay
from svs.scenarios import DriverModel, approach_accel, make_scenario
from svs.sim.ground_truth import LeadTruth
from svs.synth import generate

CFG = load_config()


# --- recordings ---------------------------------------------------------------------------

def make_meta(**kw):
    base = dict(source="synthetic", scenario="test", seed=1, dt=0.05,
                camera=dict(width=640, height=360, fov=60.0, pitch=-6.0, x=1.8, z=1.4),
                camera_mount_height_m=1.4, camera_bumper_offset_m=0.6, radar_bumper_offset_m=0.4)
    base.update(kw)
    return RecordingMeta(**base)


def test_recording_roundtrip(tmp_path):
    rec = Recording(make_meta(events=[[1.0, 2.0, "camera_occlusion", 0.0]]))
    rec.append(FrameData(1, 0.0, 5.0, [Detection((1, 2, 3, 4), 0.9, "car")],
                         np.array([[10.0, 0.1, 0.0, -1.0]]), LeadTruth(9.0, 0.1, -1.0, 5.0, 4.0, 9.0)))
    rec.append(FrameData(2, 0.05, 5.0, [], np.empty((0, 4)), None))
    loaded = Recording.load(rec.save(tmp_path / "r.npz"))
    assert loaded.meta == rec.meta
    a, b = loaded.frames
    assert a.detections == [Detection((1.0, 2.0, 3.0, 4.0), pytest.approx(0.9), "car")]
    assert a.radar_points.tolist() == [[10.0, pytest.approx(0.1), 0.0, -1.0]]
    assert a.truth.gap_m == 9.0 and a.truth.ttc_s == pytest.approx(9.0)
    assert b.truth is None and b.detections == [] and b.radar_points.shape == (0, 4)


# --- noise models ---------------------------------------------------------------------------

def test_perturb_keeps_truncated_edges_on_the_border():
    det = Detection((0.0, 100.0, 200.0, 360.0), 1.0, "car")     # cut on the left and bottom
    rng = np.random.default_rng(0)
    for _ in range(50):
        (out,) = perturb_detections([det], 640, 360, noise_px=3.0, miss_rate=0.0, rng=rng)
        assert out.box[0] == 0.0 and out.box[3] == 360.0
        assert out.box[1] != 100.0


def test_replay_detector_is_deterministic_per_seed():
    dets = [Detection((100.0, 100.0, 200.0, 200.0), 1.0, "car")]
    a = ReplayDetector(640, 360, 1.5, 0.1, np.random.default_rng(5))
    b = ReplayDetector(640, 360, 1.5, 0.1, np.random.default_rng(5))
    assert [a.detect({"detections": dets}) for _ in range(20)] == [b.detect({"detections": dets}) for _ in range(20)]


def lead_points(depth=20.0, n=10, closing=2.0):
    az = np.linspace(-0.03, 0.03, n)
    return np.column_stack([np.full(n, depth), az, np.zeros(n), np.full(n, closing)])


def test_radar_degrader_noise_dropout_and_ghosts():
    quiet = RadarPerceptionConfig(point_depth_noise_m=0.0, point_velocity_noise_mps=0.0, frame_dropout=0.0,
                                  ghosts_per_minute=0.0)
    pts = lead_points()
    assert np.array_equal(RadarDegrader(quiet).apply(pts, 5.0, 0.05), pts)
    always_drop = dataclasses.replace(quiet, frame_dropout=1.0)
    assert len(RadarDegrader(always_drop).apply(pts, 5.0, 0.05)) == 0

    ghosts = dataclasses.replace(quiet, ghosts_per_minute=6000.0, ghost_duration_s=0.5)
    deg = RadarDegrader(ghosts, np.random.default_rng(1))
    deg.apply(pts, 8.0, 0.05)                                  # ghost spawns
    first = deg._ghost[0]
    out = deg.apply(pts, 8.0, 0.05)
    assert len(out) > len(pts)
    assert deg._ghost[0] == pytest.approx(first - 8.0 * 0.05)  # stationary: approaches at ego speed
    ghost = out[len(pts):]
    assert np.all(ghost[:, 3] < -7.0)                           # CARLA sign: negative when closing in


def test_radar_ghost_hijacks_nearest_cluster():
    """Why radar-only false warnings happen: a closer stationary ghost wins the extraction."""
    ghosts = RadarPerceptionConfig(point_depth_noise_m=0.0, point_velocity_noise_mps=0.0, frame_dropout=0.0,
                                   ghosts_per_minute=6000.0)
    deg = RadarDegrader(ghosts, np.random.default_rng(3))
    for _ in range(3):
        out = deg.apply(lead_points(depth=45.0, closing=0.0), 8.0, 0.05)
    m = extract_radar_measurements(out)
    assert m.range_m < 41 and m.range_rate_mps < -7


def test_camera_occlusion_events():
    dets = [Detection((300.0, 150.0, 340.0, 190.0), 1.0, "car"), Detection((10.0, 150.0, 60.0, 190.0), 1.0, "car")]
    events = [[1.0, 2.0, "camera_occlusion", 0.0], [3.0, 4.0, "partial_occlusion", 0.5]]
    assert apply_camera_events(dets, events, 1.5, 640) == []
    partial = apply_camera_events(dets, events, 3.5, 640)
    assert partial[0].box == (300.0, 150.0, 320.0, 190.0) and partial[1] == dets[1]   # only the central box
    assert apply_camera_events(dets, events, 5.0, 640) == dets


# --- scenarios and drivers --------------------------------------------------------------------

def test_scenarios_are_deterministic_and_plausible():
    a, b = make_scenario("stop_and_go", 7), make_scenario("stop_and_go", 7)
    assert np.array_equal(a.lead_speed, b.lead_speed) and a.events == b.events
    assert a.lead_speed.min() == 0.0 and 6.0 < a.lead_speed.max() <= 8.3
    assert np.max(np.abs(np.diff(a.lead_speed))) / a.dt <= 3.5 + 1e-9      # bounded accelerations
    assert a.events_of("distraction")
    assert make_scenario("occlusion", 7).events_of("camera_occlusion") or make_scenario("occlusion", 7).events_of("partial_occlusion")
    assert make_scenario("cut_in", 7).events_of("cut_in")
    assert not make_scenario("approach", 7).lead_speed.any()
    with pytest.raises(ValueError):
        make_scenario("highway", 1)


def test_driver_keeps_distance_and_distraction_delays_reaction():
    plan = make_scenario("stop_and_go", 2)
    attentive = DriverModel(dataclasses.replace(plan, events=[]))
    distracted = DriverModel(dataclasses.replace(plan, events=[e for e in plan.events if e.kind == "distraction"]))
    t = plan.events_of("distraction")[0].t_start + 0.5
    # Lead stopped 15 m ahead, ego at 8 m/s: attentive brakes, distracted does not (TTC ~1.9 s > wake-up).
    for _ in range(20):
        a_att = attentive.accel(t, 8.0, 15.0, 0.0)
        a_dis = distracted.accel(t, 8.0, 15.0, 0.0)
    assert a_att < -2.0 and a_dis >= -0.1
    for _ in range(20):
        a_dis = distracted.accel(t, 8.0, 6.0, 0.0)           # TTC 0.75 s: wakes up
    assert a_dis < -5.0


def test_approach_driver_stops_at_the_mark():
    gap, v = 50.0, 0.0
    for _ in range(int(60 / 0.05)):
        a = approach_accel(gap, v)
        v = max(0.0, v + a * 0.05)
        gap -= v * 0.05
    assert v < 0.05 and 4.0 < gap < 6.0


def test_speed_controller_and_reference():
    ctrl = SpeedController()
    throttle, brake = ctrl.step(0.05, 2.0, 8.0)
    assert throttle > 0.3 and brake == 0.0
    throttle, brake = ctrl.step(0.05, 8.0, 2.0)
    assert throttle == 0.0 and brake > 0.0
    assert SpeedController().step(0.05, 0.0, 0.0) == (0.0, 0.8)    # hold at standstill
    ref = AccelToSpeed(5.0)
    assert ref.step(0.05, -2.0, 5.0) == pytest.approx(4.9)
    assert ref.step(1.0, 5.0, 0.0) <= 2.0                         # cannot run away from reality


# --- config and calibration helpers -----------------------------------------------------------

def test_with_overrides_casts_and_validates():
    c = with_overrides(CFG, {"fusion.sigma_accel_mps2": "2.5", "warning.on_frames": "4",
                             "fusion.fused_confirm_requires_camera": "false"})
    assert (c.fusion.sigma_accel_mps2, c.warning.on_frames, c.fusion.fused_confirm_requires_camera) == (2.5, 4, False)
    with pytest.raises(KeyError):
        with_overrides(CFG, {"fusion.nope": 1})


def test_calibration_statistics():
    rng = np.random.default_rng(0)
    x = rng.normal(0.3, 0.2, 5000)
    x[:50] = 40.0                                   # outliers do not move robust stats
    med, sig, n = robust_stats(x)
    assert med == pytest.approx(0.3, abs=0.02) and sig == pytest.approx(0.2, rel=0.08) and n == 5000
    rows = [(d, 0.0, 0.1 + 0.01 * d, 50) for d in (5, 15, 25, 35)]
    a, b = fit_linear_sigma(rows)
    assert a == pytest.approx(0.1) and b == pytest.approx(0.01)
    assert sign_agreement(np.array([1.0, -2.0, 3.0]), np.array([2.0, -1.0, -3.0])) == pytest.approx(2 / 3)

    distance = rng.uniform(10, 40, 3000)
    sigma = 0.05 * distance
    error = rng.normal(0, 1, 3000) * sigma * 2.0     # sensor twice as noisy as reported
    scale, _ = sigma_scale(distance, error, sigma, np.arange(0, 60, 5.0))
    assert scale == pytest.approx(2.0, rel=0.1)


# --- synthetic world + pipeline ------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_stop_and_go():
    return generate("stop_and_go", 4, CFG, duration_s=40.0)


def test_synthetic_world_is_consistent(synthetic_stop_and_go):
    rec = synthetic_stop_and_go
    gt = rec.truth_array("gap_m")
    assert np.isfinite(gt).all() and gt.min() > 0
    assert np.mean([len(f.detections) > 0 for f in rec]) > 0.9
    assert np.mean([len(f.radar_points) > 0 for f in rec]) > 0.9
    # The clean camera boxes and radar points measure the right distance.
    res = replay(rec, with_overrides(CFG, {"radar_perception.ghosts_per_minute": 0.0}), "radar_raw")
    err = np.array([r.radar_gap.gap_m - g for r, g in zip(res, gt) if r.radar_gap])
    assert abs(np.median(err)) < 0.1


def test_all_modes_see_identical_measurements(synthetic_stop_and_go):
    rec = synthetic_stop_and_go
    runs = {mode: replay(rec, CFG, mode) for mode in ("camera", "radar", "fused")}
    for a, b in zip(runs["camera"], runs["fused"]):
        assert a.camera_gap == b.camera_gap and a.radar_gap == b.radar_gap
    assert replay(rec, CFG, "fused")[-1].estimate == runs["fused"][-1].estimate   # deterministic


def test_pipeline_with_control_outputs_commands(synthetic_stop_and_go):
    p = Pipeline(CFG, synthetic_stop_and_go.meta, "fused", seed=1, with_control=True)
    results = [p.step(f) for f in synthetic_stop_and_go.frames[:100]]
    assert all(r.command is not None for r in results)
    assert any(r.estimate.confirmed for r in results)


def test_hud_draws_on_any_step(synthetic_stop_and_go):
    from svs.hud import draw_warning_hud
    p = Pipeline(CFG, synthetic_stop_and_go.meta, "fused", seed=1, with_control=True)
    for f in synthetic_stop_and_go.frames[:30]:
        img = np.zeros((360, 640, 3), np.uint8)
        draw_warning_hud(img, p.step(f), "fused", CFG.warning, f.truth)
    assert img.any()
    assert not math.isnan(float(img.mean()))
