import dataclasses

import numpy as np
import pytest

from svs.config import ControlConfig
from svs.control.acc import GapController, accel_to_pedals
from svs.fusion.tracker import TrackEstimate

DT = 0.05
CFG = ControlConfig()


def run(lead_speed_fn, seconds=60.0, gap0=30.0, v0=8.0, lag_s=0.3):
    """Closed loop on a point-mass ego whose acceleration follows the command with a lag."""
    ctrl = GapController(CFG)
    gap, v, a = gap0, v0, 0.0
    log = []
    for k in range(int(seconds / DT)):
        v_lead = lead_speed_fn(k * DT)
        est = TrackEstimate("confirmed", gap_m=gap, gap_rate_mps=v_lead - v)
        cmd = ctrl.step(DT, est, v)
        a += (cmd.accel_mps2 - a) * DT / lag_s
        if v < 0.3 and cmd.reason == "hold":
            a = min(a, 0.0)
        v = max(0.0, v + a * DT)
        gap += (v_lead - v) * DT
        log.append((gap, v, cmd.accel_mps2, cmd.reason))
    return log


def test_converges_to_time_gap_behind_steady_lead():
    log = run(lambda t: 7.0)
    gap, v, _, _ = log[-1]
    assert v == pytest.approx(7.0, abs=0.2)
    assert gap == pytest.approx(CFG.standstill_gap_m + CFG.time_gap_s * 7.0, abs=0.5)


def test_never_exceeds_cruise_speed_behind_fast_lead():
    _, v, _, reason = run(lambda t: 15.0)[-1]
    assert v == pytest.approx(CFG.cruise_speed_mps, abs=0.2) and reason == "cruise"


def test_stop_and_go_without_collision_and_gently():
    # Lead: 8 m/s, brakes at 2 m/s^2 to a stop, waits, goes again.
    def lead(t):
        t = t % 20.0
        if t < 5:
            return 8.0
        if t < 9:
            return max(0.0, 8.0 - 2.0 * (t - 5))
        if t < 13:
            return 0.0
        return min(8.0, 2.0 * (t - 13))
    log = run(lead, seconds=80.0, gap0=16.0)
    gaps = np.array([g for g, *_ in log])
    accels = np.array([a for _, _, a, _ in log])
    assert gaps.min() > 2.0                          # never closer than 2 m
    assert accels.min() >= -CFG.comfort_decel_mps2 - 1e-9   # gentle: no emergency needed
    assert np.max(np.abs(np.diff(accels))) <= CFG.max_jerk_mps3 * DT + 1e-9


def test_emergency_braking_only_when_ttc_is_small():
    ctrl = GapController(CFG)
    cmd = ctrl.step(DT, TrackEstimate("confirmed", gap_m=5.0, gap_rate_mps=-8.0), 10.0)  # TTC 0.6 s
    assert cmd.reason == "emergency" and cmd.accel_mps2 == -CFG.emergency_decel_mps2
    ctrl = GapController(CFG)
    cmd = ctrl.step(DT, TrackEstimate("confirmed", gap_m=12.0, gap_rate_mps=-2.0), 10.0)
    assert cmd.reason == "follow" and cmd.accel_mps2 >= -CFG.max_jerk_mps3 * DT - 1e-9


def test_cruise_without_lead_and_hold_at_standstill():
    ctrl = GapController(CFG)
    for _ in range(100):
        cmd = ctrl.step(DT, None, 3.0)
    assert cmd.reason == "cruise" and cmd.throttle > 0 and cmd.brake == 0
    hold = GapController(CFG).step(DT, TrackEstimate("confirmed", gap_m=3.0, gap_rate_mps=0.0), 0.0)
    assert hold.reason == "hold" and hold.brake > 0


def test_unconfirmed_track_is_ignored():
    ctrl = GapController(dataclasses.replace(CFG, max_jerk_mps3=100.0))
    cmd = ctrl.step(DT, TrackEstimate("tentative", gap_m=3.0, gap_rate_mps=-5.0), 5.0)
    assert cmd.reason == "cruise"


def test_pedal_mapping():
    assert accel_to_pedals(0.0, CFG) == (CFG.throttle_bias, 0.0)
    assert accel_to_pedals(-0.1, CFG) == (0.0, 0.0)
    throttle, brake = accel_to_pedals(-2.0, CFG)
    assert throttle == 0.0 and brake == pytest.approx(2.0 * CFG.brake_per_mps2)
    assert accel_to_pedals(100.0, CFG) == (1.0, 0.0)
