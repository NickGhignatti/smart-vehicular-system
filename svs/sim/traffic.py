"""CARLA traffic helpers for the scenario runner: lane keeping, adjacent lanes, spawning.

Steering follows Lab3's waypoint route exercise: aim at a waypoint ahead in the desired
lane and steer with the sign/size of the 2D cross product between heading and target
direction. Changing the desired lane id makes the vehicle change lane (cut-in).
"""

import logging
import math

import carla

log = logging.getLogger(__name__)


def driving_waypoint(carla_map, location):
    return carla_map.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)


def adjacent_lane(waypoint):
    """A neighbouring driving lane in the same direction, or None."""
    for neighbour in (waypoint.get_left_lane(), waypoint.get_right_lane()):
        if (neighbour is not None and neighbour.lane_type == carla.LaneType.Driving
                and neighbour.lane_id * waypoint.lane_id > 0):
            return neighbour
    return None


def lane_with_id(waypoint, lane_id, max_steps=4):
    """Walk sideways from `waypoint` to the lane with `lane_id` (same road), or None."""
    if waypoint.lane_id == lane_id:
        return waypoint
    for step in ("get_left_lane", "get_right_lane"):
        wp = waypoint
        for _ in range(max_steps):
            wp = getattr(wp, step)()
            if wp is None:
                break
            if wp.lane_id == lane_id:
                return wp
    return None


class LaneKeeper:
    def __init__(self, carla_map, vehicle, lane_id=None, gain=1.8, max_steer=0.5):
        self.map = carla_map
        self.vehicle = vehicle
        self.lane_id = lane_id if lane_id is not None else driving_waypoint(carla_map, vehicle.get_location()).lane_id
        self.gain = gain
        self.max_steer = max_steer

    def steer(self):
        tf = self.vehicle.get_transform()
        wp = driving_waypoint(self.map, tf.location)
        if wp is None:
            return 0.0
        wp = lane_with_id(wp, self.lane_id) or wp
        v = self.vehicle.get_velocity()
        lookahead = max(6.0, 1.2 * math.hypot(v.x, v.y))
        ahead = wp.next(lookahead)
        if not ahead:
            return 0.0
        target = min(ahead, key=lambda w: abs(w.lane_id - self.lane_id)).transform.location
        dx, dy = target.x - tf.location.x, target.y - tf.location.y
        norm = math.hypot(dx, dy) or 1.0
        forward = tf.get_forward_vector()
        cross = forward.x * dy / norm - forward.y * dx / norm
        return max(-self.max_steer, min(self.max_steer, self.gain * cross))


def spawn_in_lane(world, blueprint, reference_wp, offset_m, lane_wp=None, speed_mps=0.0, lift=0.3):
    """Spawn `blueprint` offset_m ahead (negative: behind) of reference_wp, in lane_wp's lane.

    Starts at speed_mps along the lane. Returns the actor or None.
    """
    base = lane_wp or reference_wp
    candidates = base.next(offset_m) if offset_m > 0 else base.previous(-offset_m) if offset_m < 0 else [base]
    for wp in candidates:
        tf = carla.Transform(wp.transform.location + carla.Location(z=lift), wp.transform.rotation)
        actor = world.try_spawn_actor(blueprint, tf)
        if actor is not None:
            if speed_mps > 0:
                fwd = wp.transform.get_forward_vector()
                actor.set_target_velocity(carla.Vector3D(fwd.x * speed_mps, fwd.y * speed_mps, 0.0))
            return actor
    return None
