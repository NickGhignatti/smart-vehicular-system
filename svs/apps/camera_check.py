"""`python -m svs camera`: run the camera pipeline live and compare it with ground truth.

Scenario (as in Lab5): ego and a parked lead on a straight lane, ~50 m apart. The ego
drives slowly towards the lead and stops behind it, so one run sweeps the whole distance
range. Every frame is written to a CSV (camera gap, sigma, ground-truth gap) for plotting
the distance error vs distance in the report.
"""

import csv
import dataclasses
import logging
import random
import time

import carla
import cv2
import numpy as np

from svs.apps.common import (bumper_offset, camera_height_above_road, draw_hud, fmt, open_video_writer,
                             speed_control)
from svs.perception.camera_geometry import PinholeCamera
from svs.perception.camera_tracker import FrontCameraTracker
from svs.perception.detector import make_detector
from svs.sim.ground_truth import lead_truth
from svs.sim.sensors import SensorSync, image_to_bgr, image_to_bgra, spawn_camera
from svs.sim.session import CarlaSession
from svs.sim.spawn import follow_spectator, spawn_ego_and_lead

log = logging.getLogger(__name__)

START_GAP_M = 50.0
TARGET_SPEED_KMH = 15.0
STOP_GAP_M = 5.0

CSV_FIELDS = ["frame", "timestamp", "gt_gap_m", "gt_range_rate_mps", "ego_speed_mps",
              "cam_gap_m", "cam_sigma_m", "cam_azimuth_deg", "cam_method", "n_detections"]


def draw_detections(image, tracker):
    for det in tracker.detections:
        x1, y1, x2, y2 = (int(round(c)) for c in det.box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (160, 160, 160), 1)
    if tracker.lead is not None:
        x1, y1, x2, y2 = (int(round(c)) for c in tracker.lead.box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(image, "LEAD", (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 2)


def run(cfg, seconds=60.0, display=True, record_path=None, detector_name=None):
    pcfg = cfg.camera_perception
    if detector_name is not None:
        pcfg = dataclasses.replace(pcfg, detector=detector_name)
    rng = random.Random(cfg.seed)
    detector = make_detector(pcfg, rng=np.random.default_rng(cfg.seed))

    log_dir = cfg.output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / time.strftime("camera_check_%Y%m%d_%H%M%S.csv")
    writer = None

    with CarlaSession(cfg.carla) as sim, open(csv_path, "w", newline="") as csv_file:
        ego, lead, _ = spawn_ego_and_lead(sim.world, rng, gap_m=START_GAP_M)
        sim.track(ego)
        sim.track(lead)
        lead.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True))

        sync = SensorSync(timeout=cfg.sensor_timeout)
        rgb_camera = sim.track(spawn_camera(sim.world, ego, cfg.camera))
        sync.add("rgb", rgb_camera)
        if "instance" in detector.required_images:
            sync.add("instance", sim.track(spawn_camera(
                sim.world, ego, cfg.camera, role_name="svs_front_instance",
                blueprint="sensor.camera.instance_segmentation")))

        # One tick so the sensors exist in the world, then measure the real mount geometry.
        sync.get(sim.tick())
        mount_height = camera_height_above_road(sim.world, rgb_camera)
        offset = bumper_offset(ego, cfg.camera.x)
        log.info("Camera %.2f m above the road, ego bumper %.2f m ahead of it, detector=%s",
                 mount_height, offset, pcfg.detector)
        camera = PinholeCamera.from_config(cfg.camera, mount_height_m=mount_height)
        tracker = FrontCameraTracker(camera, detector, pcfg, bumper_offset_m=offset)

        rows = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        rows.writeheader()
        if record_path is not None:
            writer = open_video_writer(record_path, 1.0 / sim.dt, cfg.camera.width, cfg.camera.height)

        steps = int(seconds / sim.dt)
        log_every = int(round(1.0 / sim.dt))
        errors = []
        try:
            for step in range(steps):
                frame = sim.tick()
                data = sync.get(frame)
                truth = lead_truth(ego, lead)
                if any(item is None for item in data.values()):
                    continue

                images = {"rgb": image_to_bgr(data["rgb"])}
                if "instance" in data:
                    images["instance"] = image_to_bgra(data["instance"])
                m = tracker.update(frame, data["rgb"].timestamp, images)

                rows.writerow({
                    "frame": frame,
                    "timestamp": round(data["rgb"].timestamp, 4),
                    "gt_gap_m": round(truth.gap_m, 3),
                    "gt_range_rate_mps": round(truth.range_rate_mps, 3),
                    "ego_speed_mps": round(truth.ego_speed_mps, 3),
                    "cam_gap_m": "" if m is None else round(m.gap_m, 3),
                    "cam_sigma_m": "" if m is None else round(m.sigma_m, 3),
                    "cam_azimuth_deg": "" if m is None else round(np.degrees(m.azimuth_rad), 3),
                    "cam_method": "" if m is None else m.method,
                    "n_detections": len(tracker.detections),
                })
                if m is not None:
                    errors.append(m.gap_m - truth.gap_m)

                target = 0.0 if truth.gap_m < STOP_GAP_M else TARGET_SPEED_KMH
                ego.apply_control(speed_control(truth.ego_speed_mps, target))
                follow_spectator(sim.spectator, ego.get_transform())

                lines = [
                    f"GT  gap {truth.gap_m:6.2f} m   ego {truth.ego_speed_mps * 3.6:5.1f} km/h",
                    f"CAM gap {fmt(m and m.gap_m, '6.2f')} m +- {fmt(m and m.sigma_m, '4.2f')}  ({m.method if m else 'no lead'})",
                    f"err {fmt(m and m.gap_m - truth.gap_m, '+5.2f')} m   detections {len(tracker.detections)}",
                ]
                if step % log_every == 0:
                    log.info(" | ".join(lines))

                if display or writer is not None:
                    hud = images["rgb"]
                    draw_detections(hud, tracker)
                    draw_hud(hud, lines)
                    if writer is not None:
                        writer.write(hud)
                    if display:
                        cv2.imshow("svs camera check", hud)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
        except KeyboardInterrupt:
            log.info("Interrupted")
        finally:
            if writer is not None:
                writer.release()
            if display:
                cv2.destroyAllWindows()

    log.info("Per-frame log: %s", csv_path)
    if errors:
        e = np.asarray(errors)
        log.info("Camera gap error over %d frames: mean %+.2f m, std %.2f m, RMSE %.2f m",
                 e.size, e.mean(), e.std(), np.sqrt(np.mean(e ** 2)))
