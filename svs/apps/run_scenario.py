"""`python -m svs run`: one scenario in CARLA with the full pipeline live.

Every frame: sensors -> clean boxes + radar points -> FrameData (saved) -> Pipeline.step
-> HUD widget. Vehicles:
- lead: follows the scenario speed profile (svs.scenarios), lane keeping;
- ego, --control warn: the scripted IDM driver (with distraction windows) drives, and the
  pipeline only warns; this is the recording to `evaluate` (all modes are compared on it);
- ego, --control acc: the GapController drives using the --mode estimate (closed loop);
- overtake / cut-in cars when the scenario has those events.
A safety brake on the ground-truth gap (< 1 m) prevents crashes from ruining a run; it is
logged, and should never trigger in warn runs.

The recording is saved to OUTPUT_DIR/recordings/carla/ (also on Ctrl+C).
"""

import dataclasses
import logging
import random
import time
from dataclasses import asdict

import carla
import cv2
import numpy as np

from svs.apps.common import bumper_offset, camera_height_above_road, open_video_writer
from svs.control.speed import AccelToSpeed, SpeedController
from svs.data.recording import FrameData, Recording, RecordingMeta
from svs.hud import draw_warning_hud
from svs.perception.detector import InstanceSegmentationDetector, YoloDetector
from svs.pipeline import Pipeline
from svs.scenarios import DriverModel, approach_accel, make_scenario
from svs.sim.ground_truth import nearest_lead_truth, speed_mps
from svs.sim.sensors import SensorSync, image_to_bgr, image_to_bgra, radar_to_array, spawn_camera, spawn_radar
from svs.sim.session import CarlaSession
from svs.sim.spawn import follow_spectator, pick_vehicle_bp, spawn_ego_and_lead
from svs.sim.traffic import LaneKeeper, adjacent_lane, driving_waypoint, spawn_in_lane

log = logging.getLogger(__name__)

SAFETY_GAP_M = 1.0


class ScriptedVehicle:
    """A vehicle driven by a speed target + lane keeping."""

    def __init__(self, carla_map, actor, lane_id=None):
        self.actor = actor
        self.speed = SpeedController()
        self.reference = AccelToSpeed(speed_mps(actor))
        self.lane = LaneKeeper(carla_map, actor, lane_id)

    def drive(self, dt, target_speed):
        throttle, brake = self.speed.step(dt, speed_mps(self.actor), target_speed)
        self.actor.apply_control(carla.VehicleControl(throttle=throttle, brake=brake, steer=self.lane.steer()))

    def drive_accel(self, dt, accel):
        """For drivers that output an acceleration (IDM)."""
        self.drive(dt, self.reference.step(dt, accel, speed_mps(self.actor)))


def _weather(name):
    preset = getattr(carla.WeatherParameters, name, None)
    if preset is None:
        raise SystemExit(f"Unknown weather preset {name!r} (see carla.WeatherParameters, e.g. ClearNoon, HardRainNoon)")
    return preset


def run(cfg, scenario="stop_and_go", seed=42, seconds=90.0, mode="fused", control="warn",
        detector_name=None, weather="ClearNoon", display=True, video=False):
    pcfg = cfg.camera_perception
    if detector_name:
        pcfg = dataclasses.replace(pcfg, detector=detector_name)
        cfg = dataclasses.replace(cfg, camera_perception=pcfg)
    dt = cfg.carla.fixed_delta_seconds
    plan = make_scenario(scenario, seed, seconds, dt)
    rng = np.random.default_rng(seed)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = cfg.output_dir / "recordings" / "carla" / f"{scenario}_{seed}_{control}_{stamp}.npz"
    recording = None
    writer = None

    with CarlaSession(cfg.carla) as sim:
        world, carla_map = sim.world, sim.world.get_map()
        world.set_weather(_weather(weather))

        ego, lead, start_wp = spawn_ego_and_lead(world, random.Random(seed), gap_m=plan.initial_gap_m + 4.5,
                                                 straight_len_m=200.0)
        sim.track(ego)
        sim.track(lead)
        vehicles = [lead]
        lead_driver = ScriptedVehicle(carla_map, lead)
        ego_lane = LaneKeeper(carla_map, ego)
        ego_speed_ctrl = SpeedController()
        ego_reference = AccelToSpeed()
        driver = DriverModel(plan)
        if scenario == "approach":
            lead.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True))

        # Sensors: RGB (+ aligned instance segmentation) camera and radar.
        sync = SensorSync(timeout=cfg.sensor_timeout)
        rgb_camera = sim.track(spawn_camera(world, ego, cfg.camera))
        sync.add("rgb", rgb_camera)
        if pcfg.detector == "instance":
            sync.add("instance", sim.track(spawn_camera(world, ego, cfg.camera, role_name="svs_front_instance",
                                                        blueprint="sensor.camera.instance_segmentation")))
            clean_detector, yolo = InstanceSegmentationDetector(pcfg.instance_min_pixels, 0.0, 0.0), None
        else:
            clean_detector, yolo = None, YoloDetector(pcfg.yolo_model, pcfg.yolo_confidence)
        sync.add("radar", sim.track(spawn_radar(world, ego, cfg.radar)))
        sync.get(sim.tick())

        meta = RecordingMeta(
            source="carla", scenario=scenario, seed=seed, dt=dt, camera=asdict(cfg.camera),
            camera_mount_height_m=camera_height_above_road(world, rgb_camera),
            camera_bumper_offset_m=bumper_offset(ego, cfg.camera.x),
            radar_bumper_offset_m=bumper_offset(ego, cfg.radar.x),
            detector=pcfg.detector, events=[e.as_list() for e in plan.events], replay_events=True,
            notes=f"map={carla_map.name} weather={weather} control={control} mode={mode} "
                  f"carla={sim.client.get_server_version()}")
        log.info("Camera %.2f m above road; bumper offsets camera %.2f m, radar %.2f m",
                 meta.camera_mount_height_m, meta.camera_bumper_offset_m, meta.radar_bumper_offset_m)
        recording = Recording(meta)
        pipeline = Pipeline(cfg, meta, mode, seed=seed, with_control=control == "acc")
        if video:
            writer = open_video_writer(out.with_suffix(".mp4"), 1.0 / dt, cfg.camera.width, cfg.camera.height)

        others = {}            # role -> ScriptedVehicle
        cut_in_driver = DriverModel(dataclasses.replace(plan, events=[]))
        safety_brakes = 0
        accel_log = []          # (commanded, actual) in --control acc, to check the pedal mapping
        previous_v = speed_mps(ego)
        log_every = int(round(1.0 / dt))
        try:
            for step in range(len(plan.lead_speed)):
                frame = sim.tick()
                data = sync.get(frame)
                t = step * dt
                ego_v = speed_mps(ego)
                ego_wp = driving_waypoint(carla_map, ego.get_location())

                def in_ego_lane(vehicle, ego_wp=ego_wp):
                    wp = driving_waypoint(carla_map, vehicle.get_location())
                    return wp is not None and ego_wp is not None and (wp.road_id, wp.lane_id) == (ego_wp.road_id, ego_wp.lane_id)

                truth = nearest_lead_truth(ego, vehicles, in_lane=in_ego_lane)

                # --- perception input ---
                if data.get("instance") is not None:
                    detections = clean_detector.clean_boxes(image_to_bgra(data["instance"]))
                elif yolo is not None and data.get("rgb") is not None:
                    detections = yolo.detect({"rgb": image_to_bgr(data["rgb"])})
                else:
                    detections = []
                points = radar_to_array(data["radar"]) if data.get("radar") is not None else np.empty((0, 4))
                frame_data = FrameData(frame, round(t, 6), ego_v, detections, points, truth)
                recording.append(frame_data)
                result = pipeline.step(frame_data, dt)

                # --- scripted traffic ---
                if scenario != "approach":      # the parked lead keeps its hand brake
                    lead_driver.drive(dt, float(plan.lead_speed[step]))
                _update_events(world, carla_map, plan, t, ego, lead, vehicles, others, sim, rng, ego_v)
                for role, car in others.items():
                    if car is None:
                        continue
                    if role == "overtake":
                        car.drive(dt, ego_v + 3.0)
                    else:   # cut-in car: follows the old lead like a normal driver
                        own = nearest_lead_truth(car.actor, [lead])
                        car.drive_accel(dt, cut_in_driver.accel(t, speed_mps(car.actor), own.gap_m if own else None,
                                                                own.lead_speed_mps if own else 0.0))

                # --- ego ---
                if control == "acc":
                    cmd = result.command
                    throttle, brake = cmd.throttle, cmd.brake
                    if cmd.reason in ("follow", "cruise") and ego_v > 1.0:
                        accel_log.append((cmd.accel_mps2, (ego_v - previous_v) / dt))
                else:
                    gap = truth.gap_m if truth else None
                    lead_v = truth.lead_speed_mps if truth else 0.0
                    accel = approach_accel(gap, ego_v) if scenario == "approach" else driver.accel(t, ego_v, gap, lead_v)
                    throttle, brake = ego_speed_ctrl.step(dt, ego_v, ego_reference.step(dt, accel, ego_v))
                previous_v = ego_v
                if truth is not None and truth.gap_m < SAFETY_GAP_M and ego_v > 0.1:
                    throttle, brake = 0.0, 1.0
                    safety_brakes += 1
                ego.apply_control(carla.VehicleControl(throttle=throttle, brake=brake, steer=ego_lane.steer()))
                follow_spectator(sim.spectator, ego.get_transform())

                # --- display ---
                if step % log_every == 0:
                    e = result.estimate
                    log.info("t=%5.1f ego %4.1f km/h | truth gap %s | %s gap %s TTC %s%s", t, ego_v * 3.6,
                             f"{truth.gap_m:5.1f}" if truth else "  -- ", mode,
                             f"{e.gap_m:5.1f}" if e.confirmed else "  -- ", f"{result.warning.ttc_s:4.1f}",
                             "  WARNING" if result.warning.active else "")
                if (display or writer is not None) and data.get("rgb") is not None:
                    hud = draw_warning_hud(image_to_bgr(data["rgb"]), result, mode, cfg.warning, truth)
                    if writer is not None:
                        writer.write(hud)
                    if display:
                        cv2.imshow("svs run", hud)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
        except KeyboardInterrupt:
            log.info("Interrupted")
        finally:
            if writer is not None:
                writer.release()
            if display:
                cv2.destroyAllWindows()
            if recording is not None and len(recording):
                recording.save(out)
                log.info("Recording (%d frames, %.0f s) saved to %s", len(recording), recording.duration_s, out)
            if safety_brakes:
                log.warning("Safety brake used on %d frames (ground-truth gap < %.1f m)", safety_brakes, SAFETY_GAP_M)
            if accel_log:
                pedal_mapping_report(np.array(accel_log), cfg.control)
    return out


def pedal_mapping_report(log_array, ccfg, lag_frames=6):
    """Compare commanded vs actual acceleration and suggest pedal gains (ControlConfig TUNE values).

    The actual acceleration lags the command (powertrain / brake build-up), so it is compared
    with the command lag_frames earlier. Ratio < 1: the car responds less than asked -> raise the gain.
    """
    cmd, actual = log_array[:-lag_frames, 0], log_array[lag_frames:, 1]
    lines = []
    for name, sel, key in (("accelerating", cmd > 0.3, "throttle_per_mps2"), ("braking", cmd < -0.5, "brake_per_mps2")):
        if sel.sum() < 20:
            lines.append(f"{name}: not enough samples")
            continue
        ratio = float(np.median(actual[sel] / cmd[sel]))
        current = getattr(ccfg, key)
        lines.append(f"{name}: actual/commanded = {ratio:.2f} -> suggest --set control.{key}={current / max(ratio, 0.1):.3f}")
    log.info("Pedal mapping check (%d samples): %s", len(log_array), "; ".join(lines))


def _update_events(world, carla_map, plan, t, ego, lead, vehicles, others, sim, rng, ego_v):
    """Spawn the overtaking / cut-in cars when their events start."""
    ego_wp = driving_waypoint(carla_map, ego.get_location())
    for event in plan.events:
        if not event.active(t) or event.kind not in ("overtake", "cut_in") or event.kind in others:
            continue
        side = adjacent_lane(ego_wp)
        if side is None:
            log.warning("No adjacent lane for the %s event at t=%.1f s; skipped", event.kind, t)
            others[event.kind] = None          # marker: do not retry every frame
            continue
        bp = pick_vehicle_bp(world, f"svs_{event.kind}", ("vehicle.audi.a2", "vehicle.nissan.micra"), rng)
        if event.kind == "overtake":
            actor = spawn_in_lane(world, bp, ego_wp, -10.0, side, speed_mps=ego_v + 3.0)
        else:
            gap = ego.get_location().distance(lead.get_location())
            actor = spawn_in_lane(world, bp, ego_wp, 0.5 * gap, side, speed_mps=speed_mps(lead))
        if actor is None:
            log.warning("Could not spawn the %s car at t=%.1f s", event.kind, t)
            others[event.kind] = None
            continue
        sim.track(actor)
        vehicles.append(actor)
        car = ScriptedVehicle(carla_map, actor, side.lane_id)
        if event.kind == "cut_in":
            car.lane.lane_id = ego_wp.lane_id      # steer into our lane: the cut-in
        others[event.kind] = car
        log.info("t=%.1f s: %s car spawned", t, event.kind)
