"""Project configuration.

Two sources, on purpose:
- `.env` holds machine-specific values (server address, ports, output folder), so each
  teammate can point at the remote simulator without touching the code.
- The dataclass defaults below hold experiment parameters (sensor mounts, rates), which
  must be identical for everyone and are versioned with the code.
"""

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
    sigma_px: float = 1.5
    # The lowest visible point of the lead is its rear tires, ahead of its rear bumper.
    # Calibrate against ground truth once the simulator is available.
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
class AppConfig:
    carla: CarlaConfig = field(default_factory=CarlaConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    camera_perception: CameraPerceptionConfig = field(default_factory=CameraPerceptionConfig)
    radar: RadarConfig = field(default_factory=RadarConfig)
    seed: int = _env("SEED", 42, int)
    output_dir: Path = PROJECT_ROOT / _env("OUTPUT_DIR", "outputs")
    log_level: str = _env("LOG_LEVEL", "INFO")
    # Max wait for a sensor message after a tick before treating it as missing.
    sensor_timeout: float = 2.0


def load_config():
    return AppConfig()
