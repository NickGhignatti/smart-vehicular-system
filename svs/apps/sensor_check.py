"""`python -m svs sensors`: end-to-end smoke test of the simulation setup.

Spawns ego + a parked lead vehicle on a straight lane, mounts the front camera and radar,
and creeps the ego towards the lead in synchronous mode. Each frame it shows the camera
image with the radar reading next to the ground truth.

This is also the first experiment of the project: CARLA's radar velocity sign
convention. As the ego closes in, the ground-truth range-rate is negative; check that
the radar range-rate has the same sign (Lab5 used abs() and lost it).
"""

import logging
import random

import carla
import cv2

from svs.apps.common import draw_hud, fmt, open_video_writer, speed_control
from svs.perception.radar import extract_radar_measurements
from svs.sim.ground_truth import lead_truth
from svs.sim.sensors import SensorSync, image_to_bgr, radar_to_array, spawn_camera, spawn_radar
from svs.sim.session import CarlaSession
from svs.sim.spawn import follow_spectator, spawn_ego_and_lead

log = logging.getLogger(__name__)

TARGET_SPEED_KMH = 15.0
STOP_GAP_M = 6.0


def run(cfg, seconds=40.0, display=True, record_path=None, lateral_offset_m=0.0):
    rng = random.Random(cfg.seed)
    writer = None

    with CarlaSession(cfg.carla) as sim:
        ego, lead, _ = spawn_ego_and_lead(sim.world, rng, gap_m=35.0, straight_len_m=120.0,
                                         lateral_offset_m=lateral_offset_m)
        sim.track(ego)
        sim.track(lead)
        lead.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True))

        sync = SensorSync(timeout=cfg.sensor_timeout)
        sync.add("camera", sim.track(spawn_camera(sim.world, ego, cfg.camera)))
        sync.add("radar", sim.track(spawn_radar(sim.world, ego, cfg.radar)))

        if record_path is not None:
            writer = open_video_writer(record_path, 1.0 / sim.dt, cfg.camera.width, cfg.camera.height)

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

                radar = None
                if data["radar"] is not None:
                    radar = extract_radar_measurements(radar_to_array(data["radar"]))

                target = 0.0 if truth.gap_m < STOP_GAP_M else TARGET_SPEED_KMH
                ego.apply_control(speed_control(truth.ego_speed_mps, target))
                follow_spectator(sim.spectator, ego.get_transform())

                lines = [
                    f"frame {frame}  ego {truth.ego_speed_mps * 3.6:5.1f} km/h",
                    f"GT    gap {truth.gap_m:6.2f} m  rate {truth.range_rate_mps:+6.2f} m/s  "
                    f"TTC {fmt(truth.ttc_s, '5.1f')} s  lat {truth.lateral_m:+5.2f} m",
                    f"RADAR rng {fmt(radar and radar.range_m, '6.2f')} m  "
                    f"vel {fmt(radar and radar.range_rate_mps, '+6.2f')} m/s  "
                    f"lat {fmt(radar and radar.lateral_m, '+5.2f')} m  ({radar.n_points if radar else 0} pts)",
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
