import math

import numpy as np
import pytest

from svs.config import WarningConfig
from svs.evaluation import estimate_metrics, segments, summarize, warning_metrics

DT = 0.05
CFG = WarningConfig(ttc_on_s=2.5, label_margin_s=0.5, label_tolerance_s=0.5)


def timeline(seconds=20.0):
    return np.arange(0, seconds, DT)


def test_segments():
    t = timeline(1.0)
    mask = np.zeros(len(t), bool)
    mask[2:5] = True
    mask[10:12] = True
    assert segments(mask, t) == [pytest.approx((0.1, 0.25)), pytest.approx((0.5, 0.6))]
    assert segments(np.zeros(len(t), bool), t) == []


def test_true_warning_and_latency():
    t = timeline()
    gt_ttc = np.full(len(t), math.inf)
    gt_ttc[(t >= 5) & (t < 7)] = 2.0          # critical episode
    active = (t >= 5.3) & (t < 8)             # warning a bit late
    m = warning_metrics(active, gt_ttc, t, CFG)
    assert (m["warnings"], m["false_warnings"], m["critical_episodes"], m["missed"]) == (1, 0, 1, 0)
    assert m["latency_s"] == pytest.approx(0.3)


def test_false_and_missed_warnings():
    t = timeline()
    gt_ttc = np.full(len(t), math.inf)
    gt_ttc[(t >= 5) & (t < 7)] = 2.0          # episode nobody warns about
    active = (t >= 12) & (t < 13)             # warning with nothing critical around
    m = warning_metrics(active, gt_ttc, t, CFG)
    assert (m["false_warnings"], m["missed"]) == (1, 1)
    assert m["false_per_min"] == pytest.approx(1 / (len(t) * DT / 60))


def test_borderline_warning_is_not_false():
    t = timeline()
    gt_ttc = np.full(len(t), math.inf)
    gt_ttc[(t >= 5) & (t < 6)] = 2.8          # above ttc_on, below ttc_on + margin
    active = (t >= 5.2) & (t < 6.5)
    assert warning_metrics(active, gt_ttc, t, CFG)["false_warnings"] == 0


def test_estimate_metrics():
    gt = np.array([10.0, 10.0, 10.0, 10.0, np.nan, 80.0])
    est = np.array([10.5, 10.5, 10.5, np.nan, 3.0, 70.0])   # constant bias, one gap
    m = estimate_metrics(est, np.zeros(6), gt, np.zeros(6))
    assert m["availability"] == pytest.approx(3 / 4)          # 80 m is out of range, NaN truth ignored
    assert m["gap_rmse_m"] == pytest.approx(0.5)
    assert m["gap_noise_m"] == pytest.approx(0.0)             # a constant bias is not noise


def test_summarize_sums_counts():
    rows = [dict(mode="fused", duration_s=60, availability=1.0, gap_rmse_m=0.2, rate_rmse_mps=0.1,
                 gap_noise_m=0.01, rate_noise_mps=0.02, latency_s=0.1, warnings=3, false_warnings=1,
                 critical_episodes=2, missed=0)] * 2
    (agg,) = summarize(rows)
    assert (agg["false_warnings"], agg["warnings"], agg["minutes"]) == (2, 6, 2.0)
    assert agg["false_per_min"] == pytest.approx(1.0)
