## Goal

Build a small sensor-fusion project that tracks the lead vehicle smoothly in stop-and-go traffic and triggers a stable warning or a gentle slow-down action.

### Tools
- CARLA front RGB camera, radar and ego-vehicle speed.
- A simple constant-velocity Kalman filter to fuse distance, relative speed and object persistence over time
- A warning widget or a basic longitudinal controller

### Implementation
- [ ] Detect the lead vehicle in the camera image and extract radar range/range-rate for the same target.
- [ ] Associate the two measurements and fuse them with a Kalman filter to obtain a smoother estimate of gap and closing speed.
- [ ] Compute a simple time-to-collision indicator and use it to warn the driver or reduce speed.
- [ ] Compare camera-only, radar-only, and fused behavior in stop-and-go traffic and mild occlusion.

## Validation and Deliverables
- Show that the fused estimate is less noisy and produce fewer false warnings than the single-sensor baseline.
- Deliverables:
    - [ ] CARLA code
    - [ ] Kalman-fusion module
    - [ ] Report with distance/TTC plots
    - [ ] Demo video

## Setup

The CARLA 0.9.15 Python client only exists for **Python 3.7–3.10**, and it must match the server version.
With [uv](https://docs.astral.sh/uv/):

```bash
uv venv --python 3.10 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt   # Linux/macOS: .venv/bin/python
cp .env.example .env    # then set CARLA_HOST / CARLA_PORT of the remote simulator
```

For notebooks, register the venv as a Jupyter kernel and select it in VS Code:

```bash
.venv/Scripts/python.exe -m ipykernel install --user --name svs-carla --display-name "Python 3.10 (SVS CARLA)"
```

## Usage

Activate the venv (`.venv\Scriptsctivate` on Windows), then:

```bash
python -m svs check                     # read-only: versions, latency, server settings, maps
python -m svs sensors                   # ego + parked lead, camera/radar vs ground truth HUD
python -m svs sensors --record          # also save the HUD video to outputs/videos/
python -m svs camera                    # camera lead detection + gap vs ground truth, CSV in outputs/logs/
python -m svs camera --detector yolo    # same with a real detector (pip install -r requirements-yolo.txt)
python -m svs --host 10.0.0.5 --town Town04 sensors --seconds 60 --no-display
```

`--host`, `--port` and `--town` override the values in `.env`.

Every run uses synchronous mode (`FIXED_DELTA_SECONDS`, default 0.05 s), reads sensor data matched
to the simulation frame, and on exit destroys its actors and restores the server settings,
so a crashed run does not leave a shared simulator frozen.

## Tests

The perception math runs without a simulator, on synthetic scenes (a car is a 3D box
projected through the camera model):

```bash
python -m pytest            # all tests
python -m pyflakes svs tests
```

## Camera pipeline

`svs/perception/`, one step per module, called once per frame by `FrontCameraTracker.update()`
(the camera counterpart of Lab5's `FrontRadarTracker`):

1. **Detection** (`detector.py`): boxes around every vehicle.
   - `instance` (default): CARLA's instance-segmentation camera, aligned pixel-for-pixel with the
     RGB camera. It handles occlusion exactly; pixel noise and random misses are added so it behaves
     like a real detector.
   - `yolo` (optional): pretrained YOLOv8n on the RGB image.
2. **Lead selection** (`lead_selector.py`): nearest box inside our lane, the Lab4/Lab5 "front
   corridor" applied to boxes, with a little hysteresis so the lead does not flicker.
3. **Distance** (`camera_tracker.py` + `camera_geometry.py`), two monocular estimates fused by
   inverse-variance weighting:
   - ground plane: `d = h / tan(angle below horizon of the box bottom)`;
   - known width: `d = f * 1.85 m / box width`.
   Each comes with a 1-sigma uncertainty (it grows with d^2), which becomes the Kalman measurement
   noise. No smoothing happens here; that is the Kalman filter's job.

### Camera field of view

A narrower FOV sees *farther*; a wider one sees *wider and closer*. With 640x360 and the Lab4 mount
(1.4 m high, pitch -6 deg):

| FOV | max range (car >= 20 px) | gap error per pixel at 20 m | closest gap with wheels in view |
|---|---|---|---|
| **60 deg** (used) | ~51 m | 0.52 m | 1.7 m |
| 90 deg | ~30 m | 0.90 m | 0.5 m |

Below 1.7 m the width method still works, and the radar is very precise at short range,
so one 60 deg camera covers the whole stop-and-go range.

## Layout

```
svs/
  __main__.py          CLI entrypoint (python -m svs ...)
  config.py            .env values (machine) + sensor/experiment parameters (code)
  sim/
    session.py         CarlaSession: connect, sync mode, actor tracking, cleanup
    spawn.py           straight-lane search, ego/lead spawning, spectator
    sensors.py         camera/radar spawn, raw-data conversion, frame-synced reads
    ground_truth.py    true gap, range-rate, TTC (evaluation only)
  perception/
    camera_geometry.py pinhole model, projection, monocular distance + uncertainty
    detector.py        Detection, instance-segmentation and YOLO detectors
    lead_selector.py   lead vehicle in our lane, with hysteresis
    camera_tracker.py  FrontCameraTracker -> CameraMeasurement (gap, sigma, bearing)
  apps/                check.py, sensor_check.py, camera_check.py, common.py
  fusion/              association + Kalman filter + track manager   (todo)
  decision/            TTC + stable warning                          (todo)
  control/             gentle slow-down controller                   (todo)
tests/                 offline unit tests (pytest)
outputs/               logs, plots, videos (git-ignored)
```
