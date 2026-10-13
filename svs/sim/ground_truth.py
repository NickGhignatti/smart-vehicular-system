"""Ground-truth gap, range-rate and TTC from the simulator state, used for evaluation only."""

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
    range_rate_mps: float  # d(gap)/dt: negative = closing in, positive = opening
    ego_speed_mps: float
    lead_speed_mps: float
    ttc_s: float


def lead_truth(ego, lead):
    ego_tf = ego.get_transform()
    forward = ego_tf.get_forward_vector()

    longitudinal = _dot(lead.get_location() - ego_tf.location, forward)
    gap = longitudinal - ego.bounding_box.extent.x - lead.bounding_box.extent.x
    range_rate = _dot(lead.get_velocity() - ego.get_velocity(), forward)

    return LeadTruth(
        gap_m=gap,
        range_rate_mps=range_rate,
        ego_speed_mps=speed_mps(ego),
        lead_speed_mps=speed_mps(lead),
        ttc_s=time_to_collision(gap, range_rate),
    )
