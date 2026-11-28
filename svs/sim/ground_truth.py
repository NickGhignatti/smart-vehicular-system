"""Ground-truth gap, lateral offset, range-rate and TTC from the simulator state, used for evaluation only."""

import math
from dataclasses import dataclass


def _dot(a, b):
    return a.x * b.x + a.y * b.y + a.z * b.z


def speed_mps(actor):
    v = actor.get_velocity()
    return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


def time_to_collision(gap_m, range_rate_mps, min_closing_mps=0.1):
    """TTC = gap / closing speed while closing (negative range-rate), otherwise infinite."""
    if range_rate_mps < -min_closing_mps:
        return gap_m / -range_rate_mps
    return math.inf


@dataclass(frozen=True)
class LeadTruth:
    gap_m: float           # bumper-to-bumper distance along the ego heading
    lateral_m: float       # lateral distance from the ego and its lead
    range_rate_mps: float  # d(gap)/dt: negative = closing in, positive = opening
    ego_speed_mps: float
    lead_speed_mps: float
    ttc_s: float


def lead_truth(ego, lead):
    """Truth for one ego/lead pair, measured on the ego heading axis (through the ego origin).

    lateral_m has the same sign as the radar y, and the two are comparable only while the
    radar sits on that axis with no yaw (true for RadarConfig today); otherwise convert first.
    """
    ego_tf = ego.get_transform()
    forward = ego_tf.get_forward_vector()

    offset = lead.get_location() - ego_tf.location

    longitudinal = _dot(offset, forward)
    lateral = _dot(offset, ego_tf.get_right_vector())
    gap = longitudinal - ego.bounding_box.extent.x - lead.bounding_box.extent.x
    range_rate = _dot(lead.get_velocity() - ego.get_velocity(), forward)

    return LeadTruth(
        gap_m=gap,
        lateral_m=lateral,
        range_rate_mps=range_rate,
        ego_speed_mps=speed_mps(ego),
        lead_speed_mps=speed_mps(lead),
        ttc_s=time_to_collision(gap, range_rate),
    )


def nearest_lead_truth(ego, vehicles, lane_half_width_m=1.5, min_gap_m=-1.0, in_lane=None):
    """Truth for the nearest vehicle ahead in the ego lane, among `vehicles` (None if none).

    With several vehicles around (overtaking car, cut-in), "the lead" is whichever is
    closest ahead within the lane at that moment. A vehicle is in the lane when its lateral
    offset is small, or when in_lane(vehicle) says so (map lookup: on a curve a lead in our
    lane can be meters off the ego heading axis).
    """
    best = None
    for vehicle in vehicles:
        truth = lead_truth(ego, vehicle)
        same_lane = abs(truth.lateral_m) <= lane_half_width_m or (in_lane is not None and in_lane(vehicle))
        if same_lane and truth.gap_m >= min_gap_m:
            if best is None or truth.gap_m < best.gap_m:
                best = truth
    return best
