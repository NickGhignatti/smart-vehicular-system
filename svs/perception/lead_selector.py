"""Pick the lead vehicle among the camera detections: the nearest one in our lane.

Same idea as the "front corridor" of Lab4/Lab5 (keep only what is straight ahead),
applied to boxes instead of points. A small hysteresis keeps the choice stable: the box
that was the lead in the previous frame gets a slightly wider corridor, so a lead
drifting near the lane edge does not flicker in and out.
"""

from svs.perception.detector import iou


def rough_distance(det, camera, car_width_m, border_px):
    """Quick distance used only to sort and gate boxes (ground plane, else width)."""
    if det.box[3] < camera.height - border_px:
        d = camera.ground_distance(det.bottom_v)
        if d is not None:
            return d
    return camera.width_distance(det.width, car_width_m)


class LeadSelector:
    def __init__(self, camera, lane_half_width_m=1.8, lane_margin_m=0.5, max_misses=5,
                 car_width_m=1.85, border_px=2, min_box_px=12, min_iou=0.3):
        self.camera = camera
        self.lane_half_width_m = lane_half_width_m
        self.lane_margin_m = lane_margin_m
        self.max_misses = max_misses
        self.car_width_m = car_width_m
        self.border_px = border_px
        self.min_box_px = min_box_px
        self.min_iou = min_iou

        self.previous = None  # last selected Detection
        self.misses = 0

    def update(self, detections):
        """Return the lead Detection for this frame, or None."""
        best, best_d = None, float("inf")
        for det in detections:
            if det.width < self.min_box_px or det.height < self.min_box_px:
                continue
            d = rough_distance(det, self.camera, self.car_width_m, self.border_px)
            if d is None or d <= 0:
                continue

            gate = self.lane_half_width_m
            if self.previous is not None and iou(det.box, self.previous.box) >= self.min_iou:
                gate += self.lane_margin_m
            if abs(self.camera.lateral_offset(det.center_u, d)) > gate:
                continue

            if d < best_d:
                best, best_d = det, d

        if best is None:
            self.misses += 1
            if self.misses > self.max_misses:
                self.previous = None
            return None

        self.previous = best
        self.misses = 0
        return best
