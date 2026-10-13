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
    fov: float = 90.0
    # Same as the world step, so in synchronous mode every tick has an image.
    sensor_tick: float = 0.05
    # Windshield mount; zero pitch keeps the ground-plane distance geometry simple.
    x: float = 1.5
    z: float = 1.4
    pitch: float = 0.0


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
    radar: RadarConfig = field(default_factory=RadarConfig)
    seed: int = _env("SEED", 42, int)
    output_dir: Path = PROJECT_ROOT / _env("OUTPUT_DIR", "outputs")
    log_level: str = _env("LOG_LEVEL", "INFO")
    # Max wait for a sensor message after a tick before treating it as missing.
    sensor_timeout: float = 2.0


def load_config():
    return AppConfig()
