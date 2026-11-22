"""Smoke tests of the simulator apps against tests/fake_carla.py (no CARLA server).

They catch crashes and wiring mistakes before a real (and scarce) simulator session.
"""

import dataclasses

import carla
import numpy as np
import pytest

from svs.config import load_config
from svs.data.recording import Recording
from tests import fake_carla


@pytest.fixture
def cfg(monkeypatch, tmp_path):
    monkeypatch.setattr(carla, "Client", fake_carla.FakeClient)
    fake_carla.FakeClient.world = None
    return dataclasses.replace(load_config(), output_dir=tmp_path, sensor_timeout=0.5)


def test_check_app(cfg):
    from svs.apps import check
    check.run(cfg)


def test_sensors_app(cfg):
    from svs.apps import sensor_check
    sensor_check.run(cfg, seconds=2.0, display=False)
    assert fake_carla.FakeClient.world.actors == []                      # everything destroyed
    assert fake_carla.FakeClient.world.settings.synchronous_mode is False   # settings restored


def test_camera_app_logs_csv(cfg):
    from svs.apps import camera_check
    camera_check.run(cfg, seconds=3.0, display=False)
    (csv_file,) = (cfg.output_dir / "logs").glob("camera_check_*.csv")
    rows = csv_file.read_text().splitlines()
    assert len(rows) > 10 and "cam_gap_m" in rows[0]


@pytest.mark.parametrize("scenario,control", [("stop_and_go", "warn"), ("approach", "warn"),
                                              ("cut_in", "warn"), ("occlusion", "acc")])
def test_run_app_saves_a_usable_recording(cfg, scenario, control):
    from svs.apps import run_scenario
    path = run_scenario.run(cfg, scenario=scenario, seed=3, seconds=8.0 if scenario != "cut_in" else 45.0,
                            mode="fused", control=control, display=False, video=True)
    rec = Recording.load(path)
    assert rec.meta.source == "carla" and rec.meta.replay_events
    assert len(rec) >= int(8.0 / rec.meta.dt) - 1
    assert sum(len(f.detections) for f in rec) > 0                        # the instance camera saw the lead
    assert sum(len(f.radar_points) for f in rec) > 0
    gt = rec.truth_array("gap_m")
    assert np.isfinite(gt).mean() > 0.9
    assert path.with_suffix(".mp4").exists()
    assert fake_carla.FakeClient.world.actors == []
    if scenario == "cut_in":
        assert any(a for a in rec.meta.events if a[2] == "cut_in")


def test_evaluate_and_calibrate_on_fake_recording(cfg):
    from svs.apps import evaluate, run_scenario
    path = run_scenario.run(cfg, scenario="approach", seed=1, seconds=20.0, display=False)
    summary = evaluate.run(cfg, [path], cfg.output_dir / "report", plots=True)
    fused = next(r for r in summary if r["mode"] == "fused")
    assert fused["availability"] > 0.8 and fused["gap_rmse_m"] < 1.5
    assert (cfg.output_dir / "report" / "summary.md").exists()
