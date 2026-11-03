"""Vehicle detectors for the front camera.

Every detector has the same interface: `detect(images) -> list[Detection]`, where
`images` maps an image name to a numpy array ("rgb": HxWx3 BGR, "instance": HxWx4 BGRA)
and `required_images` says which ones it needs.

- InstanceSegmentationDetector: built on CARLA's `sensor.camera.instance_segmentation`.
  It sees exactly which pixels belong to which vehicle (so occlusion is handled for free),
  then adds pixel noise and random misses so it behaves like a real detector.
- YoloDetector: a pretrained YOLO on the RGB image (optional, needs `ultralytics`).
"""

from dataclasses import dataclass

import numpy as np

# CARLA 0.9.14+ semantic tags (CityScapes palette) for road vehicles.
VEHICLE_TAGS = {14: "car", 15: "truck", 16: "bus"}


@dataclass(frozen=True)
class Detection:
    box: tuple  # (x1, y1, x2, y2) in pixels, x2/y2 exclusive
    score: float
    label: str

    @property
    def width(self):
        return self.box[2] - self.box[0]

    @property
    def height(self):
        return self.box[3] - self.box[1]

    @property
    def center_u(self):
        return (self.box[0] + self.box[2]) / 2.0

    @property
    def bottom_v(self):
        return float(self.box[3])


def iou(a, b):
    """Intersection over union of two (x1, y1, x2, y2) boxes."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def decode_instance_image(bgra):
    """CARLA instance-segmentation image -> (semantic tag, instance id) per pixel.

    The red channel holds the semantic tag; green and blue together encode the instance id.
    """
    tags = bgra[:, :, 2]
    ids = bgra[:, :, 1].astype(np.int32) + (bgra[:, :, 0].astype(np.int32) << 8)
    return tags, ids


class InstanceSegmentationDetector:
    required_images = ("instance",)

    def __init__(self, min_pixels=40, noise_px=1.5, miss_rate=0.05, rng=None):
        self.min_pixels = min_pixels
        self.noise_px = noise_px
        self.miss_rate = miss_rate
        self.rng = rng if rng is not None else np.random.default_rng()

    def detect(self, images):
        tags, ids = decode_instance_image(images["instance"])
        height, width = tags.shape
        vehicle = np.isin(tags, list(VEHICLE_TAGS))

        detections = []
        for instance_id in np.unique(ids[vehicle]):
            rows, cols = np.nonzero(vehicle & (ids == instance_id))
            if rows.size < self.min_pixels:
                continue
            if self.rng.random() < self.miss_rate:
                continue
            box = np.array([cols.min(), rows.min(), cols.max() + 1, rows.max() + 1], dtype=float)
            box += self.rng.normal(0.0, self.noise_px, size=4)
            box = np.clip(box, 0, [width, height, width, height])
            label = VEHICLE_TAGS[int(np.bincount(tags[rows, cols]).argmax())]
            detections.append(Detection(tuple(float(x) for x in box), 1.0, label))
        return detections


class YoloDetector:
    required_images = ("rgb",)
    # COCO class ids.
    CLASSES = {2: "car", 5: "bus", 7: "truck"}

    def __init__(self, model="yolov8n.pt", confidence=0.4):
        try:
            from ultralytics import YOLO
        except ImportError as err:
            raise ImportError("YOLO detector needs: pip install -r requirements-yolo.txt") from err
        self.model = YOLO(model)
        self.confidence = confidence

    def detect(self, images):
        result = self.model(images["rgb"], conf=self.confidence, classes=list(self.CLASSES), verbose=False)[0]
        boxes = result.boxes
        return [
            Detection(tuple(float(x) for x in xyxy), float(score), self.CLASSES[int(cls)])
            for xyxy, score, cls in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy())
        ]


def make_detector(cfg, rng=None):
    """Build the detector named in CameraPerceptionConfig.detector."""
    if cfg.detector == "instance":
        return InstanceSegmentationDetector(cfg.instance_min_pixels, cfg.instance_noise_px, cfg.instance_miss_rate, rng)
    if cfg.detector == "yolo":
        return YoloDetector(cfg.yolo_model, cfg.yolo_confidence)
    raise ValueError(f"Unknown detector {cfg.detector!r} (expected 'instance' or 'yolo')")
