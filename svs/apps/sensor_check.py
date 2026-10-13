"""`python -m svs sensors`: end-to-end smoke test of the simulation setup.

Spawns ego + a parked lead vehicle on a straight lane, mounts the front camera and radar,
and creeps the ego towards the lead in synchronous mode. Each frame it shows the camera
image with the radar reading next to the ground truth.

This is also the first experiment of the project: CARLA's radar velocity sign
convention. As the ego closes in, the ground-truth range-rate is negative; check whether
the radar velocity has the same sign (Lab5 used abs() and lost it).
"""

import logging
import math
import random

import carla
import cv2
import numpy as np

from svs.sim.ground_truth import lead_truth
from svs.sim.sensors import (ALTITUDE, AZIMUTH, DEPTH, VELOCITY, SensorSync, image_to_bgr,
                             radar_to_array, spawn_camera, spawn_radar)
from svs.sim.session import CarlaSession
from svs.sim.spawn import follow_spectator, spawn_ego_and_lead

log = logging.getLogger(__name__)

TARGET_SPEED_KMH = 15.0
STOP_GAP_M = 6.0
CORRIDOR_AZIMUTH_DEG = 5.0
CORRIDOR_ALTITUDE_DEG = 5.0


def front_radar_summary(points):
    """Robust range / range-rate of the detections in a narrow front corridor (Lab5 style)."""
    mask = ((np.abs(np.degrees(points[:, AZIMUTH])) < CORRIDOR_AZIMUTH_DEG)
            & (np.abs(np.degrees(points[:, ALTITUDE])) < CORRIDOR_ALTITUDE_DEG)
            & (points[:, DEPTH] > 1.0))
    front = points[mask]
    if len(front) == 0:
        return None, None, 0
    return float(np.percentile(front[:, DEPTH], 15)), float(np.median(front[:, VELOCITY])), len(front)


def speed_control(speed_mps, target_kmh):
    """Tiny P controller (Lab3 cruise logic) to creep at a constant speed."""
    error = target_kmh / 3.6 - speed_mps
    if target_kmh <= 0.0:
        return carla.VehicleControl(throttle=0.0, brake=1.0)
    if error > 0:
        return carla.VehicleControl(throttle=float(np.clip(0.3 + 0.15 * error, 0.0, 0.6)), brake=0.0)
    return carla.VehicleControl(throttle=0.0, brake=float(np.clip(-0.1 * error, 0.0, 0.3)))


def _fmt(value, spec):
    return "--" if value is None or not math.isfinite(value) else format(value, spec)


def draw_hud(frame_bgr, lines):
    for i, text in enumerate(lines):
        y = 26 + 24 * i
        cv2.putText(frame_bgr, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(frame_bgr, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)


def run(cfg, seconds=40.0, display=True, record_path=None):
    rng = random.Random(cfg.seed)
    writer = None

    with CarlaSession(cfg.carla) as sim:
        ego, lead, _ = spawn_ego_and_lead(sim.world, rng, gap_m=35.0)
        sim.track(ego)
        sim.track(lead)
        lead.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True))

        sync = SensorSync(timeout=cfg.sensor_timeout)
        sync.add("camera", sim.track(spawn_camera(sim.world, ego, cfg.camera)))
        sync.add("radar", sim.track(spawn_radar(sim.world, ego, cfg.radar)))

        if record_path is not None:
            record_path.parent.mkdir(parents=True, exist_ok=True)
            fps = 1.0 / sim.dt
            writer = cv2.VideoWriter(str(record_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                     (cfg.camera.width, cfg.camera.height))
            log.info("Recording to %s", record_path)

        steps = int(seconds / sim.dt)
        log_every = int(round(1.0 / sim.dt))
        missing = {"camera": 0, "radar": 0}
        step = -1
        try:
            for step in range(steps):
                frame = sim.tick()
                data = sync.get(frame)
                truth = lead_truth(ego, lead)

                for name, item in data.items():
                    missing[name] += item is None

                radar_range = radar_rate = None
                n_front = 0
                if data["radar"] is not None:
                    radar_range, radar_rate, n_front = front_radar_summary(radar_to_array(data["radar"]))

                target = 0.0 if truth.gap_m < STOP_GAP_M else TARGET_SPEED_KMH
                ego.apply_control(speed_control(truth.ego_speed_mps, target))
                follow_spectator(sim.spectator, ego.get_transform())

                lines = [
                    f"frame {frame}  ego {truth.ego_speed_mps * 3.6:5.1f} km/h",
                    f"GT    gap {truth.gap_m:6.2f} m  rate {truth.range_rate_mps:+6.2f} m/s  TTC {_fmt(truth.ttc_s, '5.1f')} s",
                    f"RADAR rng {_fmt(radar_range, '6.2f')} m  vel {_fmt(radar_rate, '+6.2f')} m/s  ({n_front} pts)",
                ]
                if step % log_every == 0:
                    log.info(" | ".join(lines))

                if data["camera"] is not None and (display or writer is not None):
                    image = image_to_bgr(data["camera"])
                    draw_hud(image, lines)
                    if writer is not None:
                        writer.write(image)
                    if display:
                        cv2.imshow("svs sensor check", image)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
        except KeyboardInterrupt:
            log.info("Interrupted")
        finally:
            if writer is not None:
                writer.release()
            if display:
                cv2.destroyAllWindows()

        log.info("Missing messages over %d frames: camera=%d radar=%d", step + 1, missing["camera"], missing["radar"])
