"""CARLA-side helpers, tested with fake sensors/actors (carla value types work without a server)."""

import math
from types import SimpleNamespace as NS

import carla
import pytest

from svs.apps.common import bumper_offset
from svs.apps.sensor_check import front_radar_summary
from svs.sim.ground_truth import lead_truth, time_to_collision
from svs.sim.sensors import SensorSync, radar_to_array


class FakeSensor:
    def listen(self, callback):
        self.callback = callback


def make_sync():
    cam, radar = FakeSensor(), FakeSensor()
    sync = SensorSync(timeout=0.05)
    sync.add("camera", cam)
    sync.add("radar", radar)
    return sync, cam.callback, radar.callback


def test_sensor_sync_drops_stale_frames():
    sync, cam, radar = make_sync()
    cam(NS(frame=9))
    cam(NS(frame=10))
    radar(NS(frame=10))
    data = sync.get(10)
    assert data["camera"].frame == 10 and data["radar"].frame == 10


def test_sensor_sync_holds_future_frames_and_reports_missing():
    sync, cam, radar = make_sync()
    cam(NS(frame=11))
    radar(NS(frame=12))            # radar skipped frame 11
    data = sync.get(11)
    assert data["camera"].frame == 11 and data["radar"] is None
    cam(NS(frame=12))
    assert sync.get(12)["radar"].frame == 12   # the held-back message is delivered


def test_radar_to_array_and_front_summary():
    dets = [NS(depth=20.0, azimuth=0.01, altitude=0.0, velocity=-3.0),
            NS(depth=50.0, azimuth=0.5, altitude=0.0, velocity=-8.0)]   # 0.5 rad: outside the corridor
    points = radar_to_array(dets)
    assert points.shape == (2, 4)
    assert front_radar_summary(points) == (20.0, -3.0, 1)
    assert radar_to_array([]).shape == (0, 4)


def test_time_to_collision():
    assert time_to_collision(20.0, -4.0) == 5.0
    assert time_to_collision(20.0, 1.0) == math.inf     # opening: no collision
    assert time_to_collision(20.0, -0.05) == math.inf   # tiny closing speed treated as none


def fake_vehicle(x, speed, half_length=2.0):
    return NS(
        get_transform=lambda: carla.Transform(carla.Location(x=x), carla.Rotation(yaw=0.0)),
        get_location=lambda: carla.Location(x=x),
        get_velocity=lambda: carla.Vector3D(x=speed),
        bounding_box=NS(location=carla.Location(), extent=carla.Vector3D(x=half_length)),
    )


def test_lead_truth_gap_rate_and_ttc():
    ego, lead = fake_vehicle(0.0, speed=10.0), fake_vehicle(30.0, speed=6.0)
    truth = lead_truth(ego, lead)
    assert truth.gap_m == pytest.approx(30.0 - 2.0 - 2.0)
    assert truth.range_rate_mps == pytest.approx(-4.0)
    assert truth.ttc_s == pytest.approx(26.0 / 4.0)


def test_bumper_offset():
    assert bumper_offset(fake_vehicle(0.0, 0.0, half_length=2.4), sensor_x=1.8) == pytest.approx(0.6)
