import dataclasses
import math

import pytest

from svs.config import WarningConfig
from svs.decision.warning import ForwardCollisionWarning, time_headway, time_to_collision
from svs.fusion.tracker import TrackEstimate

DT = 0.05
CFG = WarningConfig(ttc_on_s=2.5, ttc_off_s=3.2, on_frames=3, min_on_s=1.0, min_closing_mps=0.5)


def est(gap, rate, status="confirmed"):
    return TrackEstimate(status, gap_m=gap, gap_rate_mps=rate)


def test_ttc_and_headway():
    assert time_to_collision(20.0, 4.0) == 5.0
    assert time_to_collision(20.0, 0.3) == math.inf          # barely closing
    assert time_to_collision(20.0, -2.0) == math.inf         # opening
    assert time_to_collision(None, 2.0) == math.inf
    assert time_headway(15.0, 10.0) == 1.5
    assert time_headway(15.0, 0.0) == math.inf


def test_needs_persistence_to_raise():
    w = ForwardCollisionWarning(CFG)
    states = [w.step(DT, est(8.0, -4.0), 10.0) for _ in range(3)]   # TTC 2.0
    assert [s.active for s in states] == [False, False, True]
    assert states[2].raised


def test_single_noisy_frame_does_not_warn():
    w = ForwardCollisionWarning(CFG)
    seq = [est(20, -2), est(20, -2), est(4, -4), est(20, -2), est(20, -2)]
    assert not any(w.step(DT, e, 10.0).active for e in seq)


def test_hysteresis_and_minimum_duration():
    w = ForwardCollisionWarning(CFG)
    for _ in range(3):
        w.step(DT, est(8.0, -4.0), 10.0)                 # TTC 2.0 -> raised
    # TTC 2.8: between on and off thresholds -> stays on.
    for _ in range(40):
        assert w.step(DT, est(11.2, -4.0), 10.0).active
    # TTC far above off: clears (min_on_s already elapsed).
    assert w.step(DT, est(30.0, -1.0), 10.0).cleared


def test_minimum_on_time():
    w = ForwardCollisionWarning(CFG)
    for _ in range(3):
        w.step(DT, est(8.0, -4.0), 10.0)
    states = [w.step(DT, est(50.0, 1.0), 10.0) for _ in range(int(CFG.min_on_s / DT) + 2)]
    assert states[0].active and not states[-1].active


def test_unconfirmed_track_never_warns():
    w = ForwardCollisionWarning(CFG)
    assert not any(w.step(DT, est(5.0, -5.0, "tentative"), 10.0).active for _ in range(10))
    assert not w.step(DT, None, 10.0).active


def test_config_is_respected():
    w = ForwardCollisionWarning(dataclasses.replace(CFG, on_frames=1))
    assert w.step(DT, est(8.0, -4.0), 10.0).active
    assert w.step(DT, est(8.0, -4.0), 10.0).ttc_s == pytest.approx(2.0)
