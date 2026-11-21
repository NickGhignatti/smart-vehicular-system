"""Project configuration.

Two sources, on purpose:
- `.env` holds machine-specific values (server address, ports, output folder), so each
  teammate can point at the remote simulator without touching the code.
- The dataclass defaults below hold experiment parameters (sensor mounts, rates), which
  must be identical for everyone and are versioned with the code.
"""

import dataclasses
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def _env(name, default, cast=str):
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return cast(value.strip())


@dataclass(frozen=True)
class CarlaConfig:
    host: str = _env("CARLA_HOST", "localhost")
    port: int = _env("CARLA_PORT", 2000, int)
    timeout: float = _env("CARLA_TIMEOUT", 20.0, float)
    tm_port: int = _env("CARLA_TM_PORT", 8000, int)
    town: str = _env("CARLA_TOWN", "")
    fixed_delta_seconds: float = _env("FIXED_DELTA_SECONDS", 0.05, float)


@dataclass(frozen=True)
class CameraConfig:
    # 640x360 keeps a BGRA frame under 1 MB, important when streaming from a remote server.
    width: int = 640
    height: int = 360
    # 60 deg instead of 90: cars stay detectable to ~50 m (vs ~30 m) and the monocular
    # distance error per pixel is roughly halved. See README "Camera field of view".
    fov: float = 60.0
    # Same as the world step, so in synchronous mode every tick has an image.
    sensor_tick: float = 0.05
    # Windshield mount from Lab4; the slight downward pitch keeps the lead's wheels in
    # view down to a ~1.7 m gap.
    x: float = 1.8
    z: float = 1.4
    pitch: float = -6.0


@dataclass(frozen=True)
class CameraPerceptionConfig:
    # "instance": CARLA instance-segmentation camera (+ synthetic noise); "yolo": optional real detector.
    detector: str = "instance"
    # Lead selection: our lane is ~3.5 m wide; a box already tracked gets a wider gate (hysteresis).
    lane_half_width_m: float = 1.8
    lane_margin_m: float = 0.5
    max_misses: int = 5
    # Distance estimation.
    car_width_m: float = 1.85
    sigma_car_width_m: float = 0.15
    sigma_px: float = 1.5              # TUNE: box-edge pixel noise of the detector in use
    # The lowest visible point of the lead is its rear tires, ahead of its rear bumper.
    # TUNE: `python -m svs calibrate` reports the camera bias on an approach recording.
    bottom_to_rear_m: float = 0.9
    min_box_px: int = 12
    border_px: int = 2
    # Instance-segmentation detector: perfect boxes, so add noise to behave like a real detector.
    instance_min_pixels: int = 40
    instance_noise_px: float = 1.5
    instance_miss_rate: float = 0.05
    # YOLO detector (optional dependency, see requirements-yolo.txt).
    yolo_model: str = "yolov8n.pt"
    yolo_confidence: float = 0.4


@dataclass(frozen=True)
class RadarConfig:
    horizontal_fov: float = 30.0
    vertical_fov: float = 10.0
    points_per_second: int = 6000
    range: float = 80.0
    sensor_tick: float = 0.05
    # Front bumper mount (as in Lab5).
    x: float = 2.0
    z: float = 1.0


@dataclass(frozen=True)
class RadarPerceptionConfig:
    # 1-sigma noise of RadarMeasurement, used as Kalman measurement noise.
    # TUNE: fit from `python -m svs calibrate` on an approach recording.
    sigma_range_m: float = 0.15
    sigma_range_per_m: float = 0.005   # extra range sigma per meter of range
    sigma_rate_mps: float = 0.15
    # Degradation applied on top of the CARLA detections (CARLA's radar has no noise model).
    # Real automotive radars: ~0.1 m / ~0.1 m/s per detection, occasional ghost targets.
    # TUNE: keep these fixed across modes; they define the "realistic radar" of the study.
    point_depth_noise_m: float = 0.10
    point_velocity_noise_mps: float = 0.10
    frame_dropout: float = 0.03        # probability that a whole radar frame is lost
    # Ghost targets: stationary false returns in the corridor (manhole covers, bridges, debris).
    # Each lives ghost_duration_s and moves towards the ego at the ego speed, like a real
    # stationary object, so a radar-only system sees a target closing in fast.
    ghosts_per_minute: float = 3.0
    ghost_duration_s: float = 0.8
    min_points: int = 2                # fewer corridor points than this = no measurement


@dataclass(frozen=True)
class FusionConfig:
    # Constant-velocity model: white-noise acceleration of the gap (stop-and-go: lead brakes/accelerates).
    # TUNE: too small = laggy estimate on braking; too large = noisy closing speed.
    sigma_accel_mps2: float = 1.5
    # Gating: chi-square 99% quantiles for 1 and 2 degrees of freedom.
    gate_chi2_1d: float = 6.63
    gate_chi2_2d: float = 9.21
    # Camera measurement sigma multiplier (the pixel-error model may be optimistic in CARLA).
    # TUNE: set so the camera NIS mean is ~1 (calibrate reports it).
    camera_sigma_scale: float = 1.0
    radar_sigma_scale: float = 1.0
    # Track management ("object persistence").
    confirm_hits: int = 3              # M of N updates to confirm a track
    confirm_window: int = 5
    max_coast_s: float = 1.5           # predict-only time before the track is dropped
    max_gap_sigma_m: float = 4.0       # drop the track if the gap uncertainty grows beyond this
    reinit_after: int = 6              # consecutive gated-out frames before re-initializing (lead changed)
    init_rate_sigma_mps: float = 3.0   # closing-speed uncertainty of a track started from the camera only
    # Fused initialization: camera and radar must agree laterally to start a track from both.
    max_lateral_disagreement_m: float = 1.5
    # Fused mode: a track is confirmed only after the camera has seen it at least once, so a
    # radar-only false target (ghost) can never raise a warning. Once confirmed, the track
    # keeps running on radar alone through camera occlusions.
    fused_confirm_requires_camera: bool = True


@dataclass(frozen=True)
class WarningConfig:
    # Forward-collision warning on TTC with hysteresis + persistence (stable, no flicker).
    # TUNE: on the tuning recordings; report results on the test recordings.
    ttc_on_s: float = 2.5
    ttc_off_s: float = 3.2
    on_frames: int = 3                 # consecutive frames below ttc_on to raise the warning
    min_on_s: float = 1.0              # once raised, keep it at least this long
    min_closing_mps: float = 0.5       # ignore TTC when barely closing in
    # Evaluation labels: a warning is justified if ground-truth TTC drops below ttc_on + margin
    # within +-tolerance of the warning event.
    label_margin_s: float = 0.5
    label_tolerance_s: float = 0.5


@dataclass(frozen=True)
class ControlConfig:
    # ACC-style gap keeping: desired gap = standstill_gap + time_gap * ego speed.
    standstill_gap_m: float = 4.0
    time_gap_s: float = 1.5
    k_gap: float = 0.25                # 1/s^2
    k_rate: float = 0.7                # 1/s
    cruise_speed_mps: float = 30 / 3.6
    k_speed: float = 0.5               # 1/s, cruise speed tracking when there is no lead
    max_accel_mps2: float = 1.5
    comfort_decel_mps2: float = 3.0    # "gentle slow-down"
    emergency_decel_mps2: float = 6.0  # only when TTC < emergency_ttc_s
    emergency_ttc_s: float = 1.2
    max_jerk_mps3: float = 2.5
    # Acceleration -> pedals mapping for CARLA vehicles.
    # TUNE: `python -m svs run --control acc` and compare commanded vs actual acceleration.
    throttle_per_mps2: float = 0.25
    throttle_bias: float = 0.15        # throttle needed to hold speed (rolling resistance)
    brake_per_mps2: float = 0.12


@dataclass(frozen=True)
class AppConfig:
    carla: CarlaConfig = field(default_factory=CarlaConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    camera_perception: CameraPerceptionConfig = field(default_factory=CameraPerceptionConfig)
    radar: RadarConfig = field(default_factory=RadarConfig)
    radar_perception: RadarPerceptionConfig = field(default_factory=RadarPerceptionConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    warning: WarningConfig = field(default_factory=WarningConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    seed: int = _env("SEED", 42, int)
    output_dir: Path = PROJECT_ROOT / _env("OUTPUT_DIR", "outputs")
    log_level: str = _env("LOG_LEVEL", "INFO")
    # Max wait for a sensor message after a tick before treating it as missing.
    sensor_timeout: float = 2.0


def load_config():
    return AppConfig()


def with_overrides(cfg, overrides):
    """Return cfg with "section.field" -> value overrides applied (strings are cast to the field type).

    Example: {"fusion.sigma_accel_mps2": "2.0", "warning.ttc_on_s": 2.2}
    Used by `--set` / `--config-file` on the command line and by the tuning tools.
    """
    sections = {}
    for key, value in overrides.items():
        section, _, name = key.partition(".")
        if not name or not hasattr(cfg, section) or not hasattr(getattr(cfg, section), name):
            raise KeyError(f"Unknown config key {key!r} (expected section.field, e.g. fusion.sigma_accel_mps2)")
        current = getattr(getattr(cfg, section), name)
        if isinstance(value, str) and not isinstance(current, str):
            if isinstance(current, bool):
                value = value.strip().lower() in ("1", "true", "yes", "on")
            else:
                value = type(current)(value)
        sections.setdefault(section, {})[name] = value
    for section, values in sections.items():
        cfg = dataclasses.replace(cfg, **{section: dataclasses.replace(getattr(cfg, section), **values)})
    return cfg


def load_overrides_file(path):
    """JSON file {"section.field": value, ...} (written by `calibrate` / `tune`)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith("_")}
