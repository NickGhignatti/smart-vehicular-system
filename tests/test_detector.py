import importlib.util

import numpy as np
import pytest

from svs.config import CameraPerceptionConfig
from svs.perception.detector import (InstanceSegmentationDetector, decode_instance_image, iou,
                                     make_detector)


def instance_image(height=60, width=80):
    return np.zeros((height, width, 4), dtype=np.uint8)


def paint(img, x1, y1, x2, y2, tag, instance_id):
    """Write a rectangle the way CARLA encodes it: R = tag, G/B = instance id (BGRA order)."""
    img[y1:y2, x1:x2, 2] = tag
    img[y1:y2, x1:x2, 1] = instance_id & 0xFF
    img[y1:y2, x1:x2, 0] = instance_id >> 8


def test_iou():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_decode_instance_image():
    img = instance_image()
    paint(img, 0, 0, 5, 5, tag=14, instance_id=300)
    tags, ids = decode_instance_image(img)
    assert tags[0, 0] == 14 and ids[0, 0] == 300
    assert tags[10, 10] == 0 and ids[10, 10] == 0


def test_instance_detector_boxes_only_vehicles():
    img = instance_image()
    paint(img, 10, 20, 30, 40, tag=14, instance_id=301)   # car
    paint(img, 50, 10, 70, 50, tag=15, instance_id=302)   # truck
    paint(img, 0, 0, 8, 8, tag=12, instance_id=303)       # pedestrian: ignored
    paint(img, 75, 55, 78, 58, tag=14, instance_id=304)   # 9 px car: below min_pixels
    det = InstanceSegmentationDetector(min_pixels=40, noise_px=0.0, miss_rate=0.0)
    boxes = sorted((d.box, d.label) for d in det.detect({"instance": img}))
    assert boxes == [((10.0, 20.0, 30.0, 40.0), "car"), ((50.0, 10.0, 70.0, 50.0), "truck")]


def test_instance_detector_occluded_vehicle_keeps_only_visible_part():
    img = instance_image()
    paint(img, 10, 10, 50, 40, tag=14, instance_id=401)   # lead
    paint(img, 30, 15, 60, 45, tag=14, instance_id=402)   # car in front of it, covering its right side
    det = InstanceSegmentationDetector(min_pixels=10, noise_px=0.0, miss_rate=0.0)
    boxes = {d.box for d in det.detect({"instance": img})}
    assert (10.0, 10.0, 50.0, 40.0) in boxes  # left column still visible, so full extent
    assert (30.0, 15.0, 60.0, 45.0) in boxes


def test_instance_detector_noise_and_misses():
    img = instance_image()
    paint(img, 20, 20, 60, 50, tag=14, instance_id=1)
    noisy = InstanceSegmentationDetector(min_pixels=10, noise_px=2.0, miss_rate=0.0, rng=np.random.default_rng(0))
    x1 = np.array([noisy.detect({"instance": img})[0].box[0] for _ in range(2000)])
    assert x1.mean() == pytest.approx(20.0, abs=0.2)
    assert x1.std() == pytest.approx(2.0, rel=0.1)

    always_miss = InstanceSegmentationDetector(min_pixels=10, noise_px=0.0, miss_rate=1.0)
    assert always_miss.detect({"instance": img}) == []


def test_make_detector():
    assert isinstance(make_detector(CameraPerceptionConfig()), InstanceSegmentationDetector)
    with pytest.raises(ValueError):
        make_detector(CameraPerceptionConfig(detector="nope"))


def test_yolo_without_ultralytics_gives_clear_error():
    if importlib.util.find_spec("ultralytics") is not None:
        pytest.skip("ultralytics is installed")
    with pytest.raises(ImportError, match="requirements-yolo.txt"):
        make_detector(CameraPerceptionConfig(detector="yolo"))
