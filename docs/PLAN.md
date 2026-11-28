# Project plan and status

Goal (from the assignment): track the lead vehicle smoothly in stop-and-go traffic by fusing the
CARLA front camera, radar and ego speed with a constant-velocity Kalman filter; compute TTC; warn
the driver (stable warning) or slow down gently; show that the fused estimate is less noisy and
gives fewer false warnings than camera-only and radar-only, in stop-and-go traffic and mild
occlusion.

Legend: ✅ done and tested offline · 🟡 implemented, needs the simulator to validate/tune · ⬜ to do

## Status by phase

| Phase | What | Status |
|---|---|---|
| 0 | Consolidate branches (camera + radar on one branch), CARLA version pin | ✅ |
| 1 | Simulator access | ⬜ waiting for the session (docs/SIMULATOR_SESSION.md) |
| 2 | Sensor validation + calibration (`sensors`, `camera`, `calibrate`) | 🟡 tools ready; run in the session |
| 3 | Scenarios + recorder (`run`: stop_and_go, approach, occlusion, cut_in) | 🟡 smoke-tested on a fake CARLA |
| 4 | Fusion: association, CV Kalman filter, track management, 3 modes + raw baselines | ✅ (tuning: 🟡) |
| 5 | TTC + stable warning (hysteresis, persistence, min duration, confirmed track) | ✅ (thresholds: 🟡) |
| 6 | Gentle slow-down controller (ACC gap policy, comfort/jerk limits, emergency fallback) | ✅ logic; pedal map 🟡 |
| 7 | Live HUD warning widget + demo videos (`run --video`) | 🟡 |
| 8 | Evaluation + report figures (`evaluate`), tuning (`tune`) | ✅ tools; results need CARLA recordings |
| — | Report text | ⬜ (docs/REPORT_OUTLINE.md) |

## Deliverables

| Deliverable | Where |
|---|---|
| CARLA code | `svs/` (see README "Layout"); entrypoint `python -m svs` |
| Kalman-fusion module | `svs/fusion/` (`kalman.py`, `tracker.py`, `measurements.py`), tests in `tests/test_kalman.py`, `tests/test_tracker.py` |
| Report with distance/TTC plots | figures from `python -m svs evaluate` (timeline per run + summary); outline in docs/REPORT_OUTLINE.md |
| Demo videos | `python -m svs run ... --video` (docs/SIMULATOR_SESSION.md step 7) |

## Stop-and-test / tune gates

| Gate | When | Pass criterion | Tool |
|---|---|---|---|
| G1 connection | session start | client = server version; sensors deliver every frame | `check`, `sensors` |
| G2 radar sign | session | radar range-rate negative while closing in (sign agreement ~100%) | `sensors`, `calibrate` |
| G3 calibration | after approach runs | camera and radar \|bias\| < 0.3 m; sigma ratio 0.7-1.4; camera NIS ~1 | `calibrate` (twice) |
| G4 dataset sanity | after recordings | no safety brakes; cut-in / overtake spawned; critical episodes present | `run` logs, `evaluate` |
| G5 filter tuning | offline, tuning set | fused gap RMSE <= best single sensor; NIS ~ chi-square mean | `tune`, `evaluate` |
| G6 warning tuning | offline, tuning set | fewest false warnings with no extra misses; latency <= ~0.3 s | `tune` |
| G7 final claim | offline, TEST set | fused: lower gap/closing-speed noise and fewer false warnings than both single sensors | `evaluate` |
| G8 controller | session | no safety brake; pedal ratios 0.8-1.2; gentle decel (<= comfort) | `run --control acc` |

## Tuning knobs

All in `svs/config.py`, marked `TUNE:`; change them with `--set section.field=value` or a JSON
file with `--config-file`.

- Sensor models (from `calibrate`): `camera_perception.bottom_to_rear_m`, `car_width_m`,
  `sigma_px`, `fusion.camera_sigma_scale`, `radar_perception.range_bias_m`, `sigma_range_m`,
  `sigma_range_per_m`, `sigma_rate_mps`.
- Filter (from `tune`): `fusion.sigma_accel_mps2` (laggy on braking if too small, noisy if too
  large). Gates (`gate_chi2_*`) and track management (`confirm_hits`, `max_coast_s`,
  `reinit_after`) rarely need changes.
- Warning (from `tune`): `warning.ttc_on_s`, `ttc_off_s`, `on_frames`, `min_on_s`. The label
  definition `warning.label_ttc_s` is fixed: it defines what "critical" means.
- Controller (from the pedal check in `run --control acc`): `control.throttle_per_mps2`,
  `brake_per_mps2`, `throttle_bias`; behaviour: `time_gap_s`, `standstill_gap_m`, `comfort_decel_mps2`.
- Study definition (keep fixed across modes, state them in the report):
  `radar_perception.point_*_noise`, `frame_dropout`, `ghosts_per_minute`, `ghost_duration_s`,
  `camera_perception.instance_noise_px`, `instance_miss_rate`.

## Design decisions (for the report)

- **Record once, replay every mode**: the comparison is fair because camera-only, radar-only and
  fused see identical (seeded) measurements; only the estimator differs.
- **Sensor degradation models**: CARLA's radar has no noise model and the instance-segmentation
  camera gives perfect boxes; without degradation radar-only would be perfect. Per-point radar
  noise, frame dropouts and stationary ghost targets, and detector edge noise / misses, are applied
  at replay with fixed parameters for all modes.
- **Camera detector**: CARLA's instance-segmentation camera (aligned with the RGB camera) gives the
  visible pixels of each vehicle, so occlusion is exact; YOLO is optional (`--detector yolo`).
- **Fused confirmation needs the camera**: a radar-only object (ghost) cannot raise a warning; once
  confirmed, the track continues on radar alone through camera occlusions (object persistence).
- **Labels independent of the system**: "critical" = ground-truth TTC < `label_ttc_s`; the warning
  threshold is tuned separately, so tuning cannot move the goalposts.
- **Tuning vs test split**: seeds 1-3 tune, seeds 11-13 report.

## Known limitations / open points

- Not yet validated on CARLA: instance-segmentation decoding and vehicle tags for 0.9.16, the
  radar velocity sign (assumed in `svs/perception/radar.py`), lane keeping of the scripted cars
  on curves, pedal mapping. The session checklist covers each.
- Synthetic recordings (`python -m svs synth`) are for development only; report numbers must come
  from CARLA recordings.
- One target (the lead); a multi-object tracker is out of scope.
- Ego speed is taken from the simulator (exact), as the assignment's "ego-vehicle speed" input.
