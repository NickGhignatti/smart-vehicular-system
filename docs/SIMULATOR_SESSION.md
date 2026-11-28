# Simulator session checklist

Everything that needs CARLA, in order, with what to look for at each step. The goal of the
session is to **collect recordings**; calibration, tuning, evaluation and plots run offline
afterwards on those recordings (no simulator needed).

Estimated time: ~1.5 h with a working setup. Steps marked **GATE** must pass before moving on.

---

## 0. Setup (10 min)

On the machine that runs the client (it can be a different PC than the server):

```bash
git clone <repo> && cd smart-vehicular-system && git checkout feat/fusion
uv venv --python 3.10 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt   # Linux/macOS: .venv/bin/python
cp .env.example .env          # set CARLA_HOST / CARLA_PORT; CARLA_TOWN=Town06 recommended
.venv\Scripts\activate        # Windows (Linux/macOS: source .venv/bin/activate)
python -m pytest -q           # all green before touching the simulator
```

- The client version must match the server exactly. The repo pins `carla==0.9.16`; for a 0.9.15
  server run `uv pip install --python .venv/Scripts/python.exe carla==0.9.15`.
- **Map**: the scenarios need a long road with a second lane in the same direction (overtake and
  cut-in events). `Town06` (long straight multi-lane roads) or `Town04` (highway). Set
  `CARLA_TOWN=Town06` in `.env` or pass `--town Town06`. On a shared server, loading a map
  changes it for everybody.

## 1. Connection (2 min) — GATE

```bash
python -m svs check
```

Look for: `client 0.9.x, server 0.9.x` identical; RPC round-trip; `synchronous=False` (if True,
someone else is running or a previous run crashed: wait or ask). **GATE**: no version mismatch.

## 2. Sensors smoke test (3 min) — GATE

```bash
python -m svs sensors --seconds 30
```

A window shows the camera with ground truth (GT) and radar lines. Look for:
- `Missing messages ... camera=0 radar=0` at the end (a few are OK on a remote server);
- while the ego approaches the parked car, **GT rate is negative and RADAR vel is negative too**.
  (Verified on CARLA 0.9.16: the radar velocity is negative when closing in, same as the GT rate,
  and `svs/perception/radar.py` passes it through unchanged. If a different version disagrees,
  negate `range_rate_mps` there.)
- RADAR rng within ~0.5 m of GT gap (it is measured from the radar, the GT is bumper to bumper).

Optional: `python -m svs sensors --lateral 1.0` (lead shifted 1 m right): RADAR lat ~ +1.

## 3. Camera check (3 min) — GATE

```bash
python -m svs camera --seconds 45
```

Look for: a green **LEAD** box on the car ahead, gray boxes on other vehicles, `CAM gap` close to
`GT gap`. At the end it prints the camera RMSE and writes `outputs/logs/camera_check_*.csv`.
**GATE**: the lead box is there for most frames. If no boxes at all: the instance-segmentation
decoding (`svs/perception/detector.py`, `decode_instance_image` / `VEHICLE_TAGS`) does not match
this CARLA version; save one frame and check the tag values.

## 4. Calibration recordings (10 min)

```bash
python -m svs run --scenario approach --seed 1 --seconds 45
python -m svs run --scenario approach --seed 2 --seconds 45
python -m svs run --scenario approach --seed 3 --seconds 45
```

The ego drives slowly from ~50 m towards a parked car and stops at ~5 m. Recordings go to
`outputs/recordings/carla/approach_<seed>_warn_<time>.npz`.

Then (offline, can be done right away):

```bash
python -m svs calibrate "outputs/recordings/carla/approach_*.npz"
```

Read `outputs/calibration/calibration.md`. **GATE**:
- radar range-rate sign agreement ~100% (otherwise see step 2);
- camera: suggested `bottom_to_rear_m` and `car_width_m` are plausible (0.5-1.2 m, 1.7-2.1 m).

Apply and verify (both should now show |bias| < 0.3 m and sigma ratio 0.7-1.4):

```bash
copy outputs\calibration\calibration.json config\calibration.json      # cp on Linux/macOS
python -m svs --config-file config/calibration.json calibrate "outputs/recordings/carla/approach_*.npz" --out outputs/calibration_check
```

## 5. Dataset recordings (25-40 min)

Seeds 1-3 are the **tuning** set, seeds 11-13 the **test** set (never tune on the test set).
`--no-display` is faster; the HUD is not needed here.

PowerShell:

```powershell
foreach ($s in 1,2,3,11,12,13) { python -m svs --config-file config/calibration.json run --scenario stop_and_go --seed $s --no-display }
foreach ($s in 1,2,11,12)      { python -m svs --config-file config/calibration.json run --scenario occlusion   --seed $s --no-display }
foreach ($s in 1,11)           { python -m svs --config-file config/calibration.json run --scenario cut_in      --seed $s --no-display }
```

bash: `for s in 1 2 3 11 12 13; do python -m svs --config-file config/calibration.json run --scenario stop_and_go --seed $s --no-display; done` (same for the others).

Look for in the log: no `Safety brake used` warning (if it appears, the scripted driver crashed
into the lead: that run's warnings are still valid, but report it); `cut_in car spawned` /
`overtake car spawned` (if "No adjacent lane", change map). Optional variety: add
`--weather HardRainNoon` to a couple of test runs.

**Bring home `outputs/recordings/carla/`**: everything below runs offline.

## 6. Controller (10 min) — GATE

```bash
python -m svs --config-file config/calibration.json run --scenario stop_and_go --seed 1 --control acc
```

The gap controller now drives the ego with the fused estimate. At the end the log prints a
**pedal mapping check** (`actual/commanded` for accelerating and braking, with suggested
`--set control.throttle_per_mps2=...` / `brake_per_mps2=...`). Apply the suggestions, run again
until the ratios are ~0.8-1.2. **GATE**: no safety brake, smooth following in stop-and-go.

## 7. Demo videos (10 min)

Same seed, different estimators shown on the HUD (`--video` saves an .mp4 next to the recording):

```bash
python -m svs --config-file config/calibration.json run --scenario occlusion --seed 11 --mode fused  --video
python -m svs --config-file config/calibration.json run --scenario occlusion --seed 11 --mode camera --video
python -m svs --config-file config/calibration.json run --scenario occlusion --seed 11 --mode radar  --video
python -m svs --config-file config/calibration.json run --scenario stop_and_go --seed 2 --control acc --video
```

---

## After the session (offline)

```bash
# 1. Tune on the tuning set (seeds 1-3) -> outputs/tuning/best.json
python -m svs --config-file config/calibration.json tune "outputs/recordings/carla/stop_and_go_[123]_warn_*.npz" "outputs/recordings/carla/occlusion_[12]_warn_*.npz" "outputs/recordings/carla/cut_in_1_warn_*.npz"
copy outputs\tuning\best.json config\best.json

# 2. Report on the test set (seeds 11-13) with both files -> outputs/report/
python -m svs --config-file config/calibration.json --config-file config/best.json evaluate "outputs/recordings/carla/stop_and_go_1[123]_warn_*.npz" "outputs/recordings/carla/occlusion_1[12]_warn_*.npz" "outputs/recordings/carla/cut_in_11_warn_*.npz"
```

`outputs/report/summary.md` is the comparison table; `summary.png` and one timeline figure per
recording are the report plots. **GATE (the project's claim)**: fused has a lower gap noise and
fewer false warnings than camera-only and radar-only, without more missed critical episodes.
If not, see "Tuning knobs" in docs/PLAN.md.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `time-out ... while waiting for the simulator` | Server not reachable: host/port in `.env`, firewall (ports 2000-2001), VPN |
| Version mismatch warning | Install the matching `carla==0.9.x` in the venv |
| `synchronous=True` in `check` before running | Someone else is running, or a crashed run: wait, or restart the server |
| Very slow ticks | Remote server + two cameras at 20 Hz: use `--no-display`; it does not change results (synchronous mode) |
| `No straight lane of ... found` | Map has no long straight road: `--town Town06` |
| `No adjacent lane for the cut_in event` | Map/road has a single lane: `--town Town06` |
| Camera shows no boxes | See step 3 (instance-segmentation tags) |
| Radar rate sign wrong | See step 2 |
