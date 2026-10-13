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

Activate the venv (`.venv\Scripts\activate` on Windows), then:

```bash
python -m svs check                     # read-only: versions, latency, server settings, maps
python -m svs sensors                   # ego + parked lead, camera/radar vs ground truth HUD
python -m svs sensors --record          # also save the HUD video to outputs/videos/
python -m svs --host 10.0.0.5 --town Town04 sensors --seconds 60 --no-display
```

`--host`, `--port` and `--town` override the values in `.env`.

Every run uses synchronous mode (`FIXED_DELTA_SECONDS`, default 0.05 s), reads sensor data matched
to the simulation frame, and on exit destroys its actors and restores the server settings,
so a crashed run does not leave a shared simulator frozen.

## Layout

```
svs/
  __main__.py        CLI entrypoint (python -m svs ...)
  config.py          .env values (machine) + sensor/experiment parameters (code)
  sim/
    session.py       CarlaSession: connect, sync mode, actor tracking, cleanup
    spawn.py         straight-lane search, ego/lead spawning, spectator
    sensors.py       camera/radar spawn, raw-data conversion, frame-synced reads
    ground_truth.py  true gap, range-rate, TTC (evaluation only)
  apps/              check.py, sensor_check.py
  perception/        camera detector, radar target extraction      (todo)
  fusion/            association + Kalman filter + track manager   (todo)
  decision/          TTC + stable warning                          (todo)
  control/           gentle slow-down controller                   (todo)
outputs/             logs, plots, videos (git-ignored)
```
