# Report outline

Each section lists the material that produces its figures/tables. Numbers come from CARLA
recordings of the TEST set (seeds 11-13), evaluated with the tuned config.

1. **Introduction** — task, stop-and-go motivation, why fusion (camera classifies and confirms,
   radar measures range and range-rate directly).

2. **Setup** — CARLA version and map, sensors (camera 640x360, FOV 60 deg, Lab4 mount; radar
   30x10 deg, 6000 pts/s; ego speed), synchronous mode at 20 Hz, frame-matched sensor reads.
   Table of sensor parameters from `svs/config.py`.

3. **Perception**
   - Camera: instance-segmentation detector, lead selection (lane corridor + hysteresis),
     monocular distance (ground plane + known width, inverse-variance fusion), sigma model.
     Figure: `outputs/calibration/error_vs_distance.png` (camera panel). FOV choice: README table.
   - Radar: corridor, nearest cluster, signed Doppler range-rate (fix of Lab5's `abs()`).
     Figure: same file (radar panel); sign-agreement number from `calibration.md`.
   - Degradation models (study definition): parameters and why they are needed.

4. **Fusion** — constant-velocity KF (state, F, Q, H/R per sensor, Joseph update), gating
   (chi-square), track management (tentative/confirmed/coasting, re-initialization), fused
   confirmation rule. Consistency: NIS means from `calibration.md`.

5. **Decision and control** — TTC / headway, warning state machine (hysteresis, persistence,
   minimum duration); ACC gap policy and limits; pedal-mapping check.

6. **Scenarios and method** — stop-and-go profile, IDM driver with distraction windows (why:
   genuine critical events vs safe following), occlusion and cut-in events, record-once /
   replay-all-modes, tuning/test split, label definition, metrics definitions
   (from `svs/evaluation.py` docstring).

7. **Results**
   - Table: `outputs/report/summary.md` (availability, gap RMSE/noise, closing-speed RMSE/noise,
     warnings, false, false/min, critical episodes, missed, latency) for camera raw, radar raw,
     camera KF, radar KF, fused KF.
   - Figure: `outputs/report/summary.png`.
   - Figures: one stop-and-go timeline and one occlusion timeline (`outputs/report/<recording>.png`):
     gap, closing speed, TTC, warning strips. Point at a ghost (radar false warning) and at an
     occlusion (camera dropout, fused continues on radar).
   - Tuning: `outputs/tuning/tuning.md` (top configurations, chosen values).
   - Controller: closed-loop run summary (min gap, max decel, pedal check).

8. **Discussion** — when fusion helps (ghosts, camera noise at range, occlusion), limitations
   (single target, simulated degradation, ground-truth ego speed), possible extensions (YOLO,
   multi-target tracking, uncertainty-aware TTC).

9. **Conclusion**

Demo videos: `run --video` outputs (same seed, `--mode fused/camera/radar`, and `--control acc`).
