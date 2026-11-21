"""Metrics for the report: estimate accuracy / noise and warning quality vs ground truth.

Estimate metrics (frames where a lead exists within MAX_RANGE_M):
- availability:  fraction of those frames with a confirmed estimate;
- gap / rate RMSE: error of the confirmed estimate;
- gap / rate noise: RMS frame-to-frame change of the error. It ignores a constant bias and
  measures what the driver would see as jitter ("the fused estimate is less noisy").

Warning metrics, event based:
- a *warning event* is a contiguous run of frames with the warning on;
- a *critical episode* is a contiguous run of frames with ground-truth TTC < ttc_on;
- a warning event is justified if ground-truth TTC < ttc_on + label_margin somewhere
  within [start - tolerance, end + tolerance]; otherwise it is a FALSE warning;
- a critical episode with no warning overlapping it (+- tolerance) is MISSED;
- latency = warning start - episode start for detected episodes (negative = early).
"""

import math

import numpy as np

from svs.fusion.tracker import MODES
from svs.pipeline import replay

MAX_RANGE_M = 50.0
MIN_EPISODE_S = 0.2


def segments(mask, t):
    """Contiguous True runs of a boolean array -> list of (t_start, t_end) (t_end exclusive)."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    dt = t[1] - t[0] if len(t) > 1 else 0.05
    edges = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    return [(float(t[s]), float(t[e - 1] + dt)) for s, e in zip(starts, ends)]


def warning_metrics(active, gt_ttc, t, cfg):
    """cfg: WarningConfig. Returns a dict of counts and rates."""
    duration_min = (t[-1] - t[0] + (t[1] - t[0])) / 60.0 if len(t) > 1 else 0.0
    events = segments(active, t)
    justified_mask = gt_ttc < cfg.ttc_on_s + cfg.label_margin_s
    episodes = [e for e in segments(gt_ttc < cfg.ttc_on_s, t) if e[1] - e[0] >= MIN_EPISODE_S]
    tol = cfg.label_tolerance_s

    false_events = 0
    for start, end in events:
        window = (t >= start - tol) & (t < end + tol)
        if not justified_mask[window].any():
            false_events += 1

    missed, latencies = 0, []
    for start, end in episodes:
        overlapping = [s for s, e in events if s < end + tol and e > start - tol]
        if overlapping:
            latencies.append(min(overlapping) - start)
        else:
            missed += 1

    return {
        "warnings": len(events),
        "false_warnings": false_events,
        "critical_episodes": len(episodes),
        "missed": missed,
        "latency_s": float(np.mean(latencies)) if latencies else math.nan,
        "false_per_min": false_events / duration_min if duration_min else math.nan,
        "warning_time_frac": float(np.mean(active)) if len(active) else 0.0,
    }


def _rms(x):
    x = x[~np.isnan(x)]
    return float(np.sqrt(np.mean(x ** 2))) if len(x) else math.nan


def estimate_metrics(est_gap, est_rate, gt_gap, gt_rate):
    relevant = ~np.isnan(gt_gap) & (gt_gap <= MAX_RANGE_M)
    have = relevant & ~np.isnan(est_gap)
    gap_err = np.where(have, est_gap - gt_gap, np.nan)
    rate_err = np.where(have & ~np.isnan(est_rate), est_rate - gt_rate, np.nan)
    return {
        "availability": float(have.sum() / relevant.sum()) if relevant.any() else math.nan,
        "gap_rmse_m": _rms(gap_err),
        "rate_rmse_mps": _rms(rate_err),
        "gap_noise_m": _rms(np.diff(gap_err)),
        "rate_noise_mps": _rms(np.diff(rate_err)),
    }


def _nan_to_inf(x):
    # Not np.nan_to_num: it would also turn inf into the largest float.
    return np.where(np.isnan(x), math.inf, x)


def series(recording, results):
    """Per-frame arrays of one replay, for metrics and plots."""
    def arr(fn):
        return np.array([fn(r) for r in results], dtype=float)

    def confirmed(fn):
        return arr(lambda r: fn(r.estimate) if r.estimate.confirmed and fn(r.estimate) is not None else np.nan)

    return {
        "t": recording.timestamps(),
        "gt_gap": recording.truth_array("gap_m"),
        "gt_rate": recording.truth_array("range_rate_mps"),
        "gt_ttc": _nan_to_inf(recording.truth_array("ttc_s")),
        "ego_speed": np.array([f.ego_speed_mps for f in recording]),
        "camera_gap": arr(lambda r: r.camera_gap.gap_m if r.camera_gap else np.nan),
        "radar_gap": arr(lambda r: r.radar_gap.gap_m if r.radar_gap else np.nan),
        "radar_rate": arr(lambda r: r.radar_gap.rate_mps if r.radar_gap else np.nan),
        "est_gap": confirmed(lambda e: e.gap_m),
        "est_rate": confirmed(lambda e: e.gap_rate_mps),
        "est_sigma_gap": confirmed(lambda e: e.sigma_gap_m),
        "ttc": arr(lambda r: r.warning.ttc_s),
        "warning": np.array([r.warning.active for r in results], dtype=bool),
        "nis_camera": arr(lambda r: r.estimate.nis.get("camera", np.nan)),
        "nis_radar": arr(lambda r: r.estimate.nis.get("radar", np.nan)),
    }


def evaluate_recording(recording, cfg, modes=MODES, seed=None):
    """Replay `recording` in every mode. Returns (rows, series_by_mode)."""
    rows, all_series = [], {}
    for mode in modes:
        s = series(recording, replay(recording, cfg, mode, seed))
        all_series[mode] = s
        row = {"recording": f"{recording.meta.scenario}_{recording.meta.seed}",
               "scenario": recording.meta.scenario, "source": recording.meta.source, "mode": mode,
               "duration_s": recording.duration_s}
        row.update(estimate_metrics(s["est_gap"], s["est_rate"], s["gt_gap"], s["gt_rate"]))
        row.update(warning_metrics(s["warning"], s["gt_ttc"], s["t"], cfg.warning))
        rows.append(row)
    return rows, all_series


def summarize(rows):
    """Aggregate rows over recordings, per mode (sums for counts, means for the rest)."""
    out = []
    for mode in MODES:
        sub = [r for r in rows if r["mode"] == mode]
        if not sub:
            continue
        minutes = sum(r["duration_s"] for r in sub) / 60.0

        def mean(key):
            vals = [r[key] for r in sub if not math.isnan(r[key])]
            return float(np.mean(vals)) if vals else math.nan

        agg = {"mode": mode, "recordings": len(sub), "minutes": minutes}
        for key in ("availability", "gap_rmse_m", "rate_rmse_mps", "gap_noise_m", "rate_noise_mps", "latency_s"):
            agg[key] = mean(key)
        for key in ("warnings", "false_warnings", "critical_episodes", "missed"):
            agg[key] = int(sum(r[key] for r in sub))
        agg["false_per_min"] = agg["false_warnings"] / minutes if minutes else math.nan
        out.append(agg)
    return out
