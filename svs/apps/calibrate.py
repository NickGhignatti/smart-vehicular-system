"""`python -m svs calibrate`: sensor bias / noise from recordings -> suggested config values.

Best input: a few `python -m svs run --scenario approach` recordings (parked lead, the ego
drives from ~50 m to 5 m, so every distance is covered). Stop-and-go recordings work too.

Writes to --out:
- calibration.json  suggested overrides -> use with `--config-file calibration.json`
- calibration.md    the report (what was measured, before/after values, checks)
- error_vs_distance.png

Iterate: apply the file, run calibrate again, check that biases are ~0 and sigma ratios ~1.
Radar ghosts and frame dropouts are disabled while calibrating (they model false targets,
not the sensor's accuracy); per-point radar noise and detector noise stay on, since the
Kalman filter sees them.
"""

import dataclasses
import json
import logging
import math

import numpy as np

from svs.calibration import binned, fit_linear_sigma, robust_stats, sigma_scale, sign_agreement
from svs.data.recording import Recording
from svs.pipeline import Pipeline
from svs.plots import plot_error_vs_distance

log = logging.getLogger(__name__)

EDGES = np.arange(0.0, 60.0, 5.0)


def _collect(cfg, paths):
    """Per-frame arrays over all recordings."""
    cal_cfg = dataclasses.replace(cfg, radar_perception=dataclasses.replace(
        cfg.radar_perception, ghosts_per_minute=0.0, frame_dropout=0.0))
    keys = ("gt_gap", "gt_rate", "gt_lateral", "cam_gap", "cam_sigma", "cam_lateral", "ground_overhang",
            "implied_width", "radar_gap", "radar_rate", "radar_lateral", "radar_sigma", "radar_range",
            "nis_camera", "nis_radar")
    data = {k: [] for k in keys}
    for path in paths:
        rec = Recording.load(path)
        pipeline = Pipeline(cal_cfg, rec.meta, "fused", seed=rec.meta.seed)
        full = Pipeline(cfg, rec.meta, "fused", seed=rec.meta.seed)      # NIS with the real config
        cam = pipeline.camera_model
        border = cfg.camera_perception.border_px
        for frame in rec:
            r = pipeline.step(frame)
            nis = full.step(frame).estimate.nis
            truth = frame.truth
            row = dict.fromkeys(keys, math.nan)
            row["nis_camera"] = nis.get("camera", math.nan)
            row["nis_radar"] = nis.get("radar", math.nan)
            if truth is not None:
                row.update(gt_gap=truth.gap_m, gt_rate=truth.range_rate_mps, gt_lateral=truth.lateral_m)
                true_cam_distance = truth.gap_m + rec.meta.camera_bumper_offset_m
                if r.camera is not None:
                    row.update(cam_gap=r.camera.gap_m, cam_sigma=r.camera.sigma_m, cam_lateral=r.camera.lateral_m)
                    x1, _, x2, y2 = r.camera.box
                    if y2 < cam.height - border:
                        ground = cam.ground_distance(y2)
                        if ground is not None:
                            row["ground_overhang"] = ground - true_cam_distance
                    if x1 > border and x2 < cam.width - border:
                        row["implied_width"] = true_cam_distance / cam.width_distance(x2 - x1, 1.0)
                if r.radar_gap is not None:
                    row.update(radar_gap=r.radar_gap.gap_m, radar_rate=r.radar_gap.rate_mps,
                               radar_lateral=r.radar_gap.lateral_m, radar_sigma=r.radar_gap.sigma_gap_m,
                               radar_range=r.radar.range_m)
            for k in keys:
                data[k].append(row[k])
    return {k: np.array(v, dtype=float) for k, v in data.items()}


def run(cfg, paths, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    d = _collect(cfg, paths)
    pc, rc, fc = cfg.camera_perception, cfg.radar_perception, cfg.fusion
    report, suggest, checks = [], {}, []

    # --- camera ---
    cam_err = d["cam_gap"] - d["gt_gap"]
    bias, spread, n_cam = robust_stats(cam_err)
    overhang, _, n_ground = robust_stats(d["ground_overhang"])
    width, _, n_width = robust_stats(d["implied_width"])
    sigma_ratio, n_ratio = sigma_scale(d["gt_gap"], cam_err, d["cam_sigma"], EDGES)
    if n_ground:
        suggest["camera_perception.bottom_to_rear_m"] = round(overhang, 3)
    if n_width:
        suggest["camera_perception.car_width_m"] = round(width, 3)
    if math.isfinite(sigma_ratio):
        suggest["fusion.camera_sigma_scale"] = round(fc.camera_sigma_scale * sigma_ratio, 3)
    report += [
        "## Camera", "",
        f"- frames with a camera measurement: {n_cam}",
        f"- gap error (camera - truth): median **{bias:+.3f} m**, robust sigma {spread:.3f} m",
        f"- actual spread / reported sigma beyond 8 m: **{sigma_ratio:.2f}** (1.0 = honest; > 1 = overconfident; "
        f"{n_ratio} frames)",
        f"- rear-tire overhang seen by the ground-plane method: **{overhang:.3f} m** "
        f"(config bottom_to_rear_m = {pc.bottom_to_rear_m}, {n_ground} frames)",
        f"- effective car width seen by the width method: **{width:.3f} m** "
        f"(config car_width_m = {pc.car_width_m}, {n_width} frames)", "",
        "| distance bin [m] | median error [m] | robust sigma [m] | frames |", "|---|---|---|---|",
    ] + [f"| {c:.0f} | {m:+.3f} | {s:.3f} | {n} |" for c, m, s, n in binned(d["gt_gap"], cam_err, EDGES)]
    if abs(bias) > 0.3:
        checks.append(f"camera bias {bias:+.2f} m > 0.3 m: apply the suggested overhang / width and re-run")

    # --- radar ---
    radar_err = d["radar_gap"] - d["gt_gap"]
    rbias, rspread, n_radar = robust_stats(radar_err)
    rate_bias, rate_sigma, _ = robust_stats(d["radar_rate"] - d["gt_rate"])
    agree = sign_agreement(d["radar_rate"], d["gt_rate"])
    a, b = fit_linear_sigma(binned(d["radar_range"], radar_err - (0 if math.isnan(rbias) else rbias), EDGES))
    if n_radar:
        suggest["radar_perception.range_bias_m"] = round(rc.range_bias_m + rbias, 3)
        suggest["radar_perception.sigma_range_m"] = round(a, 3)
        suggest["radar_perception.sigma_range_per_m"] = round(b, 4)
        suggest["radar_perception.sigma_rate_mps"] = round(max(rate_sigma, 0.02), 3)
    report += [
        "", "## Radar", "",
        f"- frames with a radar measurement: {n_radar}",
        f"- gap error (radar - truth): median **{rbias:+.3f} m**, robust sigma {rspread:.3f} m; "
        f"sigma model a + b*range: a = {a:.3f} m, b = {b:.4f}",
        f"- range-rate error: median {rate_bias:+.3f} m/s, robust sigma **{rate_sigma:.3f} m/s**",
        f"- **range-rate sign agreement with truth: {agree:.1%}** (must be ~100%)", "",
        "| distance bin [m] | median error [m] | robust sigma [m] | frames |", "|---|---|---|---|",
    ] + [f"| {c:.0f} | {m:+.3f} | {s:.3f} | {n} |" for c, m, s, n in binned(d["gt_gap"], radar_err, EDGES)]
    if math.isfinite(agree) and agree < 0.9:
        checks.append(f"RADAR VELOCITY SIGN LOOKS FLIPPED ({agree:.0%} agreement): CARLA's velocity convention is "
                      "the opposite of what svs.perception.radar assumes; flip the sign in extract_radar_measurements")

    # --- lateral + filter consistency ---
    lat_cam = robust_stats(d["cam_lateral"] - d["gt_lateral"])
    lat_radar = robust_stats(d["radar_lateral"] - d["gt_lateral"])
    nis_c, nis_r = np.nanmean(d["nis_camera"]), np.nanmean(d["nis_radar"])
    report += [
        "", "## Lateral position and filter consistency", "",
        f"- lateral error camera: median {lat_cam[0]:+.2f} m, sigma {lat_cam[1]:.2f} m; "
        f"radar: median {lat_radar[0]:+.2f} m, sigma {lat_radar[1]:.2f} m",
        f"- mean NIS of accepted updates (current config): camera {nis_c:.2f} (ideal ~1), radar {nis_r:.2f} (ideal ~2)",
    ]
    if math.isfinite(nis_c) and not 0.5 < nis_c < 2.0:
        checks.append(f"camera NIS {nis_c:.2f} far from 1: apply fusion.camera_sigma_scale")

    suggest_lines = [f"- `{k}`: {v}" for k, v in suggest.items()]
    md = "\n".join(
        ["# Calibration", "", f"Recordings: {', '.join(p.stem for p in paths)}", "",
         "## Checks", ""] + ([f"- **{c}**" for c in checks] or ["- all good"]) +
        ["", "## Suggested values (calibration.json)", ""] + suggest_lines + [""] + report)
    (out_dir / "calibration.md").write_text(md, encoding="utf-8")
    payload = dict(suggest)
    payload["_note"] = "Suggested by `python -m svs calibrate`; use with --config-file. Re-run calibrate to verify."
    payload["_recordings"] = [p.name for p in paths]
    (out_dir / "calibration.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    plot_error_vs_distance(d["gt_gap"], {"camera": cam_err, "radar": radar_err},
                           {"camera": d["cam_sigma"], "radar": d["radar_sigma"]},
                           out_dir / "error_vs_distance.png", "Measurement error vs distance")
    print(md)
    log.info("Calibration written to %s", out_dir)
    return suggest, checks
