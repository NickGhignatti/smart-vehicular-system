"""Synthetic recordings: a 1-D traffic world rendered into CARLA-like sensor outputs.

Purpose: develop, test and dry-run tune the whole pipeline (perception -> fusion ->
warning -> control -> evaluation) without a simulator. Numbers obtained on synthetic data
are NOT results: the report must use CARLA recordings. The output has the exact format of
a CARLA recording (svs.data.recording), so every tool works on both.

Rendering:
- camera: each vehicle is a box (1.85 x 1.5 m, 4.5 m long); its image box uses the rear face
  for top/left/right and the rear-tire contact point (overhang ahead of the rear bumper) for
  the bottom, like a real car. Nearer vehicles hide farther ones (horizontal interval
  subtraction), as the instance-segmentation camera does.
- radar: points on the rear face of every vehicle, as many as the radar rays hitting it
  (points_per_second / FPS spread over the FOV), plus road and guardrail clutter. Velocity
  follows CARLA's convention (negative = towards the sensor, i.e. the range rate).
"""

import math
from dataclasses import asdict

import numpy as np

from svs.data.recording import FrameData, Recording, RecordingMeta
from svs.perception.camera_geometry import PinholeCamera
from svs.perception.detector import Detection
from svs.scenarios import DriverModel, approach_accel, make_scenario
from svs.sim.ground_truth import LeadTruth, time_to_collision

CAR_WIDTH, CAR_HEIGHT, CAR_LENGTH = 1.85, 1.5, 4.5
REAR_OVERHANG = 0.9         # rear bumper -> rear tire contact
BUMPER_HEIGHT = 0.3         # lowest point of the rear face above the road
EGO_HALF_LENGTH = 2.4       # ego origin -> front bumper
LANE_WIDTH = 3.5
IN_LANE = 1.5               # |lateral| below this = in our lane


class Vehicle:
    def __init__(self, x, lateral, speed, role):
        self.x = x              # rear bumper position along the road (world), m
        self.lateral = lateral
        self.speed = speed
        self.role = role        # "lead", "overtake", "cut_in"


def _sensor_offsets(cfg):
    return EGO_HALF_LENGTH - cfg.camera.x, EGO_HALF_LENGTH - cfg.radar.x


def _project_vehicle(camera, distance, lateral):
    """Image box (u1, v1, u2, v2) of a vehicle whose rear bumper is `distance` m ahead of the camera."""
    h = camera.mount_height_m
    face = np.array([[distance, lateral + s * CAR_WIDTH / 2, z - h]
                     for s in (-1, 1) for z in (BUMPER_HEIGHT, CAR_HEIGHT)])
    contact = np.array([[distance + REAR_OVERHANG, lateral, -h]])
    uv = camera.project(np.vstack([face, contact]))
    if np.isnan(uv).any():
        return None
    u1, u2 = uv[:4, 0].min(), uv[:4, 0].max()
    v1 = uv[:4, 1].min()
    v2 = max(uv[4, 1], uv[:4, 1].max())
    return u1, v1, u2, v2


def _render_camera(camera, vehicles, ego_x, camera_offset, occlusion):
    """Clean boxes of the visible vehicles (nearest first hides the farther ones)."""
    boxes = []
    for v in sorted(vehicles, key=lambda v: v.x):
        distance = v.x - (ego_x + EGO_HALF_LENGTH) + camera_offset
        if distance < 0.5:
            continue
        box = _project_vehicle(camera, distance, v.lateral)
        if box is None:
            continue
        u1, v1, u2, v2 = box
        if v.role == "lead" and occlusion is not None:
            kind, fraction = occlusion
            if kind == "camera_occlusion":
                continue
            u2 = u2 - fraction * (u2 - u1)      # partial: an occluder hides the right side
        # Hidden by nearer boxes: subtract their horizontal extents.
        segments = [(u1, u2)]
        for (n1, _, n2, nv2), _ in boxes:
            if nv2 >= v2:                        # nearer object reaches lower in the image
                segments = [piece for a, b in segments
                            for piece in ((a, min(b, n1)), (max(a, n2), b)) if piece[1] - piece[0] > 1]
        if not segments:
            continue
        u1, u2 = min(a for a, _ in segments), max(b for _, b in segments)
        u1, u2 = max(0.0, u1), min(float(camera.width), u2)
        v1, v2 = max(0.0, v1), min(float(camera.height), v2)
        if (u2 - u1) * (v2 - v1) < 40 or u2 <= 0 or u1 >= camera.width:
            continue
        boxes.append(((u1, v1, u2, v2), v))
    return [Detection(tuple(float(c) for c in box), 1.0, "car") for box, _ in boxes]


def _render_radar(cfg, vehicles, ego_x, ego_speed, radar_offset, rng):
    rc = cfg.radar
    rays = rc.points_per_second / (1.0 / rc.sensor_tick)
    rows = []
    for v in vehicles:
        x = v.x - (ego_x + EGO_HALF_LENGTH) + radar_offset
        if x < 0.5 or x > rc.range:
            continue
        az_width = math.degrees(2 * math.atan(CAR_WIDTH / 2 / x))
        el_height = math.degrees(math.atan((CAR_HEIGHT - BUMPER_HEIGHT) / x))
        expected = rays * min(az_width, rc.horizontal_fov) * min(el_height, rc.vertical_fov) / (
            rc.horizontal_fov * rc.vertical_fov)
        n = rng.poisson(expected)
        if n == 0:
            continue
        xs = x + rng.uniform(0.0, 0.3, n)
        ys = v.lateral + rng.uniform(-CAR_WIDTH / 2, CAR_WIDTH / 2, n)
        zs = rng.uniform(BUMPER_HEIGHT, CAR_HEIGHT, n) - rc.z
        rows.append(_polar(xs, ys, zs, v.speed - ego_speed))
    # Clutter: road surface (below the radar) and a guardrail, both stationary.
    n_road = rng.poisson(20)
    if n_road:
        xs = rng.uniform(3.0, 40.0, n_road)
        rows.append(_polar(xs, rng.uniform(-3, 3, n_road), np.full(n_road, -rc.z), -ego_speed))
    n_rail = rng.poisson(8)
    if n_rail:
        xs = rng.uniform(5.0, 60.0, n_rail)
        rows.append(_polar(xs, np.full(n_rail, -LANE_WIDTH * 1.5), rng.uniform(-0.6, -0.2, n_rail), -ego_speed))
    if not rows:
        return np.empty((0, 4))
    points = np.vstack(rows)
    half_az, half_el = math.radians(rc.horizontal_fov / 2), math.radians(rc.vertical_fov / 2)
    keep = (np.abs(points[:, 1]) <= half_az) & (np.abs(points[:, 2]) <= half_el)
    return points[keep]


def _polar(xs, ys, zs, relative_speed):
    depth = np.sqrt(xs ** 2 + ys ** 2 + zs ** 2)
    azimuth = np.arctan2(ys, xs)
    altitude = np.arcsin(zs / depth)
    # CARLA: velocity < 0 when the target moves towards the sensor (it is the range rate).
    velocity = relative_speed * np.cos(azimuth) * np.cos(altitude)
    return np.column_stack([depth, azimuth, altitude, velocity])


def generate(name, seed, cfg, duration_s=90.0):
    """Synthetic Recording of scenario `name` (see svs.scenarios) with the sensors of `cfg`."""
    dt = cfg.carla.fixed_delta_seconds
    plan = make_scenario(name, seed, duration_s, dt)
    rng = np.random.default_rng(seed + 10_000)
    camera = PinholeCamera.from_config(cfg.camera)
    camera_offset, radar_offset = _sensor_offsets(cfg)
    driver = DriverModel(plan)

    meta = RecordingMeta(
        source="synthetic", scenario=name, seed=seed, dt=dt, camera=asdict(cfg.camera),
        camera_mount_height_m=cfg.camera.z, camera_bumper_offset_m=camera_offset,
        radar_bumper_offset_m=radar_offset, detector="instance",
        events=[e.as_list() for e in plan.events if e.kind != "distraction"]
               + [e.as_list() for e in plan.events_of("distraction")],
        notes="Synthetic data for development only; not a result.")
    recording = Recording(meta)

    ego_x, ego_speed, ego_accel = 0.0, plan.ego_initial_speed_mps, 0.0
    lead = Vehicle(ego_x + EGO_HALF_LENGTH + plan.initial_gap_m, 0.0, float(plan.lead_speed[0]), "lead")
    vehicles = [lead]
    cut_in = next(iter(plan.events_of("cut_in")), None)
    overtake = next(iter(plan.events_of("overtake")), None)

    for i, target in enumerate(plan.lead_speed):
        t = i * dt
        # --- traffic ---
        for v in vehicles:
            if v.role in ("lead", "cut_in_lead"):
                v.speed += float(np.clip(target - v.speed, -6 * dt, 3 * dt))
        lead.lateral = 0.15 * math.sin(2 * math.pi * t / 7.0) if lead.role == "lead" else lead.lateral
        if overtake and overtake.active(t) and not any(v.role == "overtake" for v in vehicles):
            vehicles.append(Vehicle(ego_x - 8.0, LANE_WIDTH, ego_speed + 3.0, "overtake"))
        if cut_in and cut_in.active(t):
            car = next((v for v in vehicles if v.role.startswith("cut_in")), None)
            if car is None:
                gap = lead.x - (ego_x + EGO_HALF_LENGTH)
                car = Vehicle(ego_x + EGO_HALF_LENGTH + 0.5 * gap, LANE_WIDTH, lead.speed, "cut_in")
                vehicles.append(car)
            progress = (t - cut_in.t_start) / (cut_in.t_end - cut_in.t_start)
            car.lateral = LANE_WIDTH * (1.0 - progress)
            lead.lateral = -LANE_WIDTH * progress           # the old lead changes lane
        elif cut_in and t >= cut_in.t_end:
            car = next((v for v in vehicles if v.role == "cut_in"), None)
            if car is not None:
                car.role, car.lateral = "cut_in_lead", 0.0
                vehicles.remove(lead)
                lead = car
        for v in vehicles:
            if v.role == "overtake":
                v.speed = max(v.speed, ego_speed + 2.0)
            v.x += v.speed * dt
        vehicles = [v for v in vehicles if v.role != "overtake" or v.x - ego_x < 120]

        # --- ground truth: nearest vehicle ahead in our lane ---
        ahead = [v for v in vehicles if abs(v.lateral) < IN_LANE and v.x > ego_x]
        truth_vehicle = min(ahead, key=lambda v: v.x) if ahead else None
        truth = None
        if truth_vehicle is not None:
            gap = truth_vehicle.x - (ego_x + EGO_HALF_LENGTH)
            rate = truth_vehicle.speed - ego_speed
            truth = LeadTruth(gap, truth_vehicle.lateral, rate, ego_speed, truth_vehicle.speed,
                              time_to_collision(gap, rate))

        # --- sensors ---
        occlusion = None
        for kind in ("camera_occlusion", "partial_occlusion"):
            for e in plan.events_of(kind):
                if e.active(t):
                    occlusion = (kind, e.value)
        detections = _render_camera(camera, vehicles, ego_x, camera_offset, occlusion)
        radar = _render_radar(cfg, vehicles, ego_x, ego_speed, radar_offset, rng)
        recording.append(FrameData(i, round(t, 6), ego_speed, detections, radar, truth))

        # --- ego driver ---
        gap_seen = truth.gap_m if truth else None
        desired = driver.accel(t, ego_speed, gap_seen, truth.lead_speed_mps if truth else 0.0)
        if name == "approach":
            desired = approach_accel(gap_seen, ego_speed)
        ego_accel += (desired - ego_accel) * dt / 0.3          # actuator lag
        ego_speed = max(0.0, ego_speed + ego_accel * dt)
        ego_x += ego_speed * dt

    return recording
