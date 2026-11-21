"""Recorded runs: the per-frame sensor inputs and ground truth of one scenario.

Recording once and replaying offline is what makes the camera / radar / fused comparison
fair (every mode sees the same frames) and lets us tune without the simulator.

What is stored is the *input* of the pipeline, before any noise model:
- clean camera boxes (instance-segmentation detector) or real detector boxes (YOLO);
- raw radar detections [depth, azimuth, altitude, velocity];
- ego speed (the "ego-vehicle speed" sensor of the assignment);
- ground truth (evaluation only).
Noise, dropouts and ghosts are applied at replay time, with a seed.

File format: one numpy .npz (no pickle). Variable-length data (boxes, radar points) is
stored flat with an index array: the items of frame i are rows index[i]:index[i+1].
"""

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from svs.perception.detector import Detection
from svs.sim.ground_truth import LeadTruth, time_to_collision

FORMAT_VERSION = 1


@dataclass
class RecordingMeta:
    source: str                      # "carla" or "synthetic"
    scenario: str
    seed: int
    dt: float
    camera: dict                     # width, height, fov, pitch, x, z (CameraConfig fields)
    camera_mount_height_m: float     # measured camera height above the road
    camera_bumper_offset_m: float    # ego front bumper ahead of the camera
    radar_bumper_offset_m: float     # ego front bumper ahead of the radar
    detector: str = "instance"       # how the boxes were produced
    # Time windows [t_start, t_end, kind] of scripted events (occlusion, cut-in, ...),
    # used to annotate plots and to split metrics.
    events: list = field(default_factory=list)
    notes: str = ""
    format_version: int = FORMAT_VERSION


@dataclass
class FrameData:
    frame: int
    timestamp: float
    ego_speed_mps: float
    detections: list                 # list[Detection], clean
    radar_points: np.ndarray         # (N, 4) [depth, azimuth, altitude, velocity]
    truth: Optional[LeadTruth] = None


class Recording:
    def __init__(self, meta, frames=None):
        self.meta = meta
        self.frames = list(frames or [])

    def __len__(self):
        return len(self.frames)

    def __iter__(self):
        return iter(self.frames)

    def append(self, frame):
        self.frames.append(frame)

    @property
    def duration_s(self):
        return len(self.frames) * self.meta.dt

    def truth_array(self, name):
        """Ground-truth field over time as a float array (NaN where unknown)."""
        return np.array([getattr(f.truth, name) if f.truth is not None else np.nan for f in self.frames])

    def timestamps(self):
        return np.array([f.timestamp for f in self.frames])

    # --- persistence ---------------------------------------------------------------------

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        frames = self.frames

        det_index = np.cumsum([0] + [len(f.detections) for f in frames])
        boxes = [d.box for f in frames for d in f.detections]
        radar_index = np.cumsum([0] + [len(f.radar_points) for f in frames])
        radar = [np.asarray(f.radar_points, dtype=np.float32).reshape(-1, 4) for f in frames]

        def truth(name):
            return np.array([getattr(f.truth, name) if f.truth else np.nan for f in frames], dtype=np.float64)

        np.savez_compressed(
            path,
            meta_json=np.array(json.dumps(asdict(self.meta))),
            frame=np.array([f.frame for f in frames], dtype=np.int64),
            timestamp=np.array([f.timestamp for f in frames], dtype=np.float64),
            ego_speed=np.array([f.ego_speed_mps for f in frames], dtype=np.float64),
            det_index=det_index.astype(np.int64),
            det_boxes=np.array(boxes, dtype=np.float32).reshape(-1, 4),
            det_scores=np.array([d.score for f in frames for d in f.detections], dtype=np.float32),
            det_labels=np.array([d.label for f in frames for d in f.detections], dtype="<U16"),
            radar_index=radar_index.astype(np.int64),
            radar_points=np.vstack(radar) if radar else np.empty((0, 4), dtype=np.float32),
            gt_gap=truth("gap_m"),
            gt_lateral=truth("lateral_m"),
            gt_range_rate=truth("range_rate_mps"),
            gt_ego_speed=truth("ego_speed_mps"),
            gt_lead_speed=truth("lead_speed_mps"),
        )
        return path

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            meta = RecordingMeta(**json.loads(str(z["meta_json"])))
            frames = []
            di, ri = z["det_index"], z["radar_index"]
            boxes, scores, labels = z["det_boxes"], z["det_scores"], z["det_labels"]
            radar = z["radar_points"]
            for i in range(len(z["frame"])):
                dets = [Detection(tuple(float(c) for c in boxes[k]), float(scores[k]), str(labels[k]))
                        for k in range(di[i], di[i + 1])]
                gap = float(z["gt_gap"][i])
                truth = None
                if not math.isnan(gap):
                    rate = float(z["gt_range_rate"][i])
                    truth = LeadTruth(gap, float(z["gt_lateral"][i]), rate, float(z["gt_ego_speed"][i]),
                                      float(z["gt_lead_speed"][i]), time_to_collision(gap, rate))
                frames.append(FrameData(int(z["frame"][i]), float(z["timestamp"][i]), float(z["ego_speed"][i]),
                                        dets, radar[ri[i]:ri[i + 1]].astype(np.float64), truth))
        return cls(meta, frames)
