"""Scenario helpers: vehicle blueprints, straight-lane search, ego/lead spawning.

Adapted from the Lab5 solution (spawn_vehicle_pair_straight), with a random.Random
instance passed in so spawn choices are reproducible from the configured seed.
"""

import logging
import math

import carla

log = logging.getLogger(__name__)


def angle_diff_deg(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def pick_vehicle_bp(world, role_name, preferred_ids, rng):
    lib = world.get_blueprint_library()
    bp = None
    for bp_id in preferred_ids:
        try:
            bp = lib.find(bp_id)
            break
        except (IndexError, RuntimeError):  # not available in this CARLA build
            continue

    if bp is None:
        four_wheeled = [
            cand for cand in lib.filter("vehicle.*")
            if not cand.has_attribute("number_of_wheels")
            or cand.get_attribute("number_of_wheels").as_int() == 4
        ]
        if not four_wheeled:
            raise RuntimeError("No suitable vehicle blueprint found")
        bp = rng.choice(four_wheeled)

    if bp.has_attribute("color"):
        colors = list(bp.get_attribute("color").recommended_values)
        if colors:
            bp.set_attribute("color", rng.choice(colors))
    if bp.has_attribute("role_name"):
        bp.set_attribute("role_name", role_name)
    return bp


def find_straight_lane_starts(carla_map, rng, length_m=250.0, step_m=2.0, max_yaw_step=4.0, max_candidates=60):
    """Waypoints from which the same lane continues straight for at least `length_m`."""
    spawn_points = list(carla_map.get_spawn_points())
    rng.shuffle(spawn_points)

    candidates = []
    for sp in spawn_points:
        wp0 = carla_map.get_waypoint(sp.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp0 is None:
            continue

        prev = wp0
        traveled = 0.0
        valid = True
        while traveled < length_m:
            nxt_list = prev.next(step_m)
            if not nxt_list:
                valid = False
                break
            nxt = min(nxt_list, key=lambda w: abs(angle_diff_deg(w.transform.rotation.yaw, prev.transform.rotation.yaw)))
            # Total heading drift from the start, not only per step, so slow curves are rejected too.
            if abs(angle_diff_deg(nxt.transform.rotation.yaw, wp0.transform.rotation.yaw)) > max_yaw_step:
                valid = False
                break
            if nxt.lane_id != wp0.lane_id or nxt.is_junction:
                valid = False
                break
            prev = nxt
            traveled += step_m

        if valid:
            candidates.append(wp0)
        if len(candidates) >= max_candidates:
            break
    return candidates


def spawn_ego_and_lead(world, rng, gap_m=25.0, straight_len_m=250.0):
    """Spawn ego and lead vehicle on the same straight lane, lead `gap_m` ahead (center to center).

    Both start braked with autopilot off. Returns (ego, lead, start_waypoint).
    The caller is responsible for registering the actors for cleanup.
    """
    ego_bp = pick_vehicle_bp(world, "svs_ego", ("vehicle.tesla.model3", "vehicle.audi.tt"), rng)
    lead_bp = pick_vehicle_bp(world, "svs_lead", ("vehicle.lincoln.mkz_2020", "vehicle.audi.a2", "vehicle.tesla.model3"), rng)

    starts = find_straight_lane_starts(world.get_map(), rng, length_m=straight_len_m)
    if not starts:
        raise RuntimeError(f"No straight lane of {straight_len_m:.0f} m found on this map")
    log.info("Found %d straight-lane candidates", len(starts))

    lift = carla.Location(z=0.3)
    for start_wp in starts:
        ego = world.try_spawn_actor(ego_bp, carla.Transform(start_wp.transform.location + lift, start_wp.transform.rotation))
        if ego is None:
            continue

        lead = None
        for lead_wp in start_wp.next(gap_m):
            if lead_wp.lane_id != start_wp.lane_id:
                continue
            lead = world.try_spawn_actor(lead_bp, carla.Transform(lead_wp.transform.location + lift, lead_wp.transform.rotation))
            if lead is not None:
                break

        if lead is None:
            ego.destroy()
            continue

        for vehicle in (ego, lead):
            vehicle.set_autopilot(False)
            vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
        log.info("Spawned ego %s and lead %s, gap %.1f m", ego.type_id, lead.type_id, gap_m)
        return ego, lead, start_wp

    raise RuntimeError("Could not spawn ego and lead on the same straight lane")


def follow_spectator(spectator, transform, distance=12.0, z=4.5, pitch=-15.0):
    yaw = math.radians(transform.rotation.yaw)
    back = carla.Location(x=-distance * math.cos(yaw), y=-distance * math.sin(yaw), z=z)
    spectator.set_transform(carla.Transform(transform.location + back, carla.Rotation(pitch=pitch, yaw=transform.rotation.yaw)))
