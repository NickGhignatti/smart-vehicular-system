"""`python -m svs tune`: grid search of filter / warning parameters on recordings.

Use a TUNING set of recordings and keep separate TEST recordings for the final
`evaluate` (tuning and reporting on the same runs overstates the results).

Only fusion.sigma_accel_mps2 changes the estimates; for each value every recording is
replayed once and the warning parameters are evaluated on the cached estimates.

Score (lower is better), weights in COST_WEIGHTS:
    w_false * false warnings per minute + w_missed * fraction of critical episodes missed
    + w_latency * mean latency (if late) + w_rmse * gap RMSE
The labels (what is critical) stay fixed: warning.label_ttc_s is never tuned.

Writes tuning.csv (every combination), tuning.md (top 15 + the current config) and
best.json (the winning overrides, for --config-file).
"""

import dataclasses
import itertools
import json
import logging
import math

import numpy as np

from svs.apps.evaluate import write_csv
from svs.config import with_overrides
from svs.data.recording import Recording
from svs.decision.warning import ForwardCollisionWarning
from svs.evaluation import estimate_metrics, series, warning_metrics
from svs.pipeline import replay

log = logging.getLogger(__name__)

GRID = {
    "fusion.sigma_accel_mps2": [0.8, 1.5, 2.5, 4.0],
    "warning.ttc_on_s": [2.0, 2.5, 3.0],
    "ttc_hysteresis_s": [0.4, 0.8],          # ttc_off_s = ttc_on_s + this
    "warning.on_frames": [1, 3, 5],
}
QUICK_GRID = {
    "fusion.sigma_accel_mps2": [1.5, 3.0],
    "warning.ttc_on_s": [2.5, 3.0],
    "ttc_hysteresis_s": [0.7],
    "warning.on_frames": [2, 4],
}
COST_WEIGHTS = {"false_per_min": 2.0, "missed_frac": 4.0, "latency_s": 1.0, "gap_rmse_m": 0.5}


def _warnings(results, ego_speed, dt, wcfg):
    fcw = ForwardCollisionWarning(wcfg)
    return np.array([fcw.step(dt, r.estimate, v).active for r, v in zip(results, ego_speed)])


def _score(m):
    latency = m["latency_s"] if math.isfinite(m["latency_s"]) else 0.0
    w = COST_WEIGHTS
    return (w["false_per_min"] * m["false_per_min"] + w["missed_frac"] * m["missed_frac"]
            + w["latency_s"] * max(0.0, latency) + w["gap_rmse_m"] * m["gap_rmse_m"])


def _evaluate(cached, wcfg):
    false = missed = episodes = 0
    minutes = 0.0
    latencies, rmse = [], []
    for rec, results, s in cached:
        active = _warnings(results, s["ego_speed"], rec.meta.dt, wcfg)
        m = warning_metrics(active, s["gt_ttc"], s["t"], wcfg)
        false += m["false_warnings"]
        missed += m["missed"]
        episodes += m["critical_episodes"]
        minutes += rec.duration_s / 60.0
        if math.isfinite(m["latency_s"]):
            latencies.append(m["latency_s"])
        e = estimate_metrics(s["est_gap"], s["est_rate"], s["gt_gap"], s["gt_rate"])
        if math.isfinite(e["gap_rmse_m"]):
            rmse.append(e["gap_rmse_m"])
    out = {"false_warnings": false, "false_per_min": false / minutes if minutes else math.nan,
           "missed": missed, "critical_episodes": episodes,
           "missed_frac": missed / episodes if episodes else 0.0,
           "latency_s": float(np.mean(latencies)) if latencies else math.nan,
           "gap_rmse_m": float(np.mean(rmse)) if rmse else math.nan}
    out["score"] = _score(out)
    return out


def run(cfg, paths, out_dir, quick=False, mode="fused"):
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = QUICK_GRID if quick else GRID
    recordings = [Recording.load(p) for p in paths]
    sigma_values = sorted(set(grid["fusion.sigma_accel_mps2"]) | {cfg.fusion.sigma_accel_mps2})
    warning_combos = list(itertools.product(grid["warning.ttc_on_s"], grid["ttc_hysteresis_s"], grid["warning.on_frames"]))
    current = (cfg.fusion.sigma_accel_mps2, cfg.warning.ttc_on_s,
               round(cfg.warning.ttc_off_s - cfg.warning.ttc_on_s, 6), cfg.warning.on_frames)
    log.info("Tuning %s on %d recordings: %d x %d combinations", mode, len(recordings), len(sigma_values),
             len(warning_combos))

    rows = []
    for sigma in sigma_values:
        cfg_s = with_overrides(cfg, {"fusion.sigma_accel_mps2": sigma})
        cached = []
        for rec in recordings:
            results = replay(rec, cfg_s, mode)
            cached.append((rec, results, series(rec, results)))
        combos = warning_combos + ([current[1:]] if sigma == current[0] else [])
        for on, hysteresis, frames in dict.fromkeys(combos):
            wcfg = dataclasses.replace(cfg.warning, ttc_on_s=on, ttc_off_s=on + hysteresis, on_frames=frames)
            row = {"fusion.sigma_accel_mps2": sigma, "warning.ttc_on_s": on, "warning.ttc_off_s": round(on + hysteresis, 3),
                   "warning.on_frames": frames, "is_current": (sigma, on, round(hysteresis, 6), frames) == current}
            row.update(_evaluate(cached, wcfg))
            rows.append(row)
        log.info("sigma_accel %.2f done (best so far %.3f)", sigma, min(r["score"] for r in rows))

    rows.sort(key=lambda r: r["score"])
    write_csv(rows, out_dir / "tuning.csv")
    best = rows[0]
    keys = ("fusion.sigma_accel_mps2", "warning.ttc_on_s", "warning.ttc_off_s", "warning.on_frames")
    best_overrides = {k: best[k] for k in keys}
    (out_dir / "best.json").write_text(json.dumps(
        {**best_overrides, "_mode": mode, "_score": round(best["score"], 4), "_recordings": [p.name for p in paths],
         "_note": "From `python -m svs tune`. Verify on separate TEST recordings with evaluate --config-file."},
        indent=2), encoding="utf-8")

    columns = list(keys) + ["false_per_min", "missed", "critical_episodes", "latency_s", "gap_rmse_m", "score"]
    header = "| " + " | ".join(c.split(".")[-1] for c in columns) + " |"
    lines = [header, "|" + "---|" * len(columns)]
    shown = rows[:15] + [r for r in rows[15:] if r["is_current"]]
    for r in shown:
        cells = [f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in columns]
        lines.append("| " + " | ".join(cells) + " |" + (" <- current config" if r["is_current"] else ""))
    md = "\n".join([f"# Tuning ({mode})", "", f"Recordings: {', '.join(p.stem for p in paths)}", "",
                    f"Score weights: {COST_WEIGHTS}", "", *lines, "",
                    f"Best overrides (best.json): {best_overrides}"])
    (out_dir / "tuning.md").write_text(md, encoding="utf-8")
    print(md)
    return best_overrides
