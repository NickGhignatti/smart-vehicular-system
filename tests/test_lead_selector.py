from svs.perception.detector import Detection
from svs.perception.lead_selector import LeadSelector
from tests.synthetic import CAMERA, car_detection


def selector(**kwargs):
    return LeadSelector(CAMERA, **kwargs)


def test_picks_nearest_car_in_our_lane():
    near_adjacent = car_detection(CAMERA, 8.0, lateral_m=3.5)
    far_in_lane = car_detection(CAMERA, 30.0)
    near_in_lane = car_detection(CAMERA, 15.0, lateral_m=0.4)
    assert selector().update([near_adjacent, far_in_lane, near_in_lane]) == near_in_lane


def test_ignores_cars_in_other_lanes():
    assert selector().update([car_detection(CAMERA, 10.0, lateral_m=-3.5)]) is None


def test_hysteresis_keeps_a_lead_drifting_to_the_lane_edge():
    edge = car_detection(CAMERA, 20.0, lateral_m=2.0)  # outside 1.8 m, inside 1.8 + 0.5
    assert selector().update([edge]) is None             # never seen before: rejected

    s = selector()
    assert s.update([car_detection(CAMERA, 20.0, lateral_m=1.5)]) is not None
    assert s.update([edge]) == edge                      # tracked one frame ago: kept


def test_cut_in_closer_car_becomes_the_lead():
    s = selector()
    old_lead = car_detection(CAMERA, 30.0)
    assert s.update([old_lead]) == old_lead
    cut_in = car_detection(CAMERA, 12.0, lateral_m=0.8)
    assert s.update([old_lead, cut_in]) == cut_in


def test_memory_is_dropped_after_max_misses():
    s = selector(max_misses=2)
    s.update([car_detection(CAMERA, 20.0)])
    for _ in range(2):
        assert s.update([]) is None
    assert s.previous is not None   # short dropout (occlusion): still remembered
    s.update([])
    assert s.previous is None


def test_ignores_tiny_boxes():
    tiny = Detection((300.0, 170.0, 305.0, 175.0), 1.0, "car")
    assert selector(min_box_px=12).update([tiny]) is None
