"""Scripted camera occlusions, applied to the detector output.

In CARLA runs, occlusion windows of the scenario (glare, dirt, a truck in the next lane)
are applied here instead of being staged physically, so they are deterministic, identical
in every mode, and can be changed without re-running the simulator. Synthetic recordings
render occlusions themselves (RecordingMeta.replay_events = False).
"""

from svs.perception.detector import Detection

CENTRAL_BAND = (0.3, 0.7)   # boxes centered here are lead candidates


def apply_camera_events(detections, events, t, image_width):
    """events: [t_start, t_end, kind, value] lists (RecordingMeta.events)."""
    for start, end, kind, *rest in events:
        if not start <= t < end:
            continue
        if kind == "camera_occlusion":
            return []
        if kind == "partial_occlusion":
            fraction = rest[0] if rest else 0.5
            out = []
            for d in detections:
                lo, hi = CENTRAL_BAND
                if lo * image_width <= d.center_u <= hi * image_width:
                    x1, y1, x2, y2 = d.box
                    d = Detection((x1, y1, x2 - fraction * (x2 - x1), y2), d.score, d.label)
                out.append(d)
            detections = out
    return detections
