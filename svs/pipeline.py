"""The whole per-frame processing chain, shared by live CARLA runs and offline replays.

    clean boxes --noise/misses--> FrontCameraTracker --> CameraMeasurement --+
                                                                             +--> estimator (KF / raw)
    radar points --degradation--> extract_radar_measurements -> RadarMeasurement-+         |
                                                                                   TrackEstimate
                                                                                   |          |
                                                               ForwardCollisionWarning   GapController

Perception (noise, detection, measurements) does not depend on the mode, and every random
source is seeded per run: replaying one recording in camera / radar / fused mode gives all
modes exactly the same measurements, so differences come from the estimator alone.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from svs.control.acc import ControlCommand, GapController
from svs.decision.warning import ForwardCollisionWarning, WarningState
from svs.fusion.measurements import GapMeasurement, from_camera, from_radar
from svs.fusion.tracker import TrackEstimate, make_estimator
from svs.perception.camera_geometry import PinholeCamera
from svs.perception.camera_tracker import CameraMeasurement, FrontCameraTracker
from svs.perception.detector import ReplayDetector
from svs.perception.occlusion import apply_camera_events
from svs.perception.radar import RadarMeasurement, extract_radar_measurements
from svs.perception.radar_noise import RadarDegrader


@dataclass(frozen=True)
class StepResult:
    timestamp: float
    camera: Optional[CameraMeasurement]      # raw camera measurement (before the filter)
    radar: Optional[RadarMeasurement]        # raw radar measurement (before the filter)
    camera_gap: Optional[GapMeasurement]
    radar_gap: Optional[GapMeasurement]
    estimate: TrackEstimate
    warning: WarningState
    command: Optional[ControlCommand]
    detections: list                         # boxes after the noise model (for drawing)


class Pipeline:
    def __init__(self, cfg, meta, mode="fused", seed=0, with_control=False):
        """
        cfg: AppConfig; meta: RecordingMeta (camera geometry and sensor offsets of the run);
        mode: one of svs.fusion.tracker.MODES; seed: noise seed (use the same for every mode).
        """
        self.cfg = cfg
        self.meta = meta
        self.mode = mode
        cam = meta.camera
        self.camera_model = PinholeCamera(int(cam["width"]), int(cam["height"]), float(cam["fov"]),
                                          meta.camera_mount_height_m, float(cam["pitch"]))
        pc = cfg.camera_perception
        noisy = meta.detector == "instance"   # real detectors (YOLO) already have their own noise
        rng_camera, rng_radar = (np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2))
        detector = ReplayDetector(self.camera_model.width, self.camera_model.height,
                                  pc.instance_noise_px if noisy else 0.0,
                                  pc.instance_miss_rate if noisy else 0.0, rng_camera)
        self.camera_tracker = FrontCameraTracker(self.camera_model, detector, pc, meta.camera_bumper_offset_m)
        self.radar_degrader = RadarDegrader(cfg.radar_perception, rng_radar)
        self.estimator = make_estimator(mode, cfg.fusion)
        self.warning = ForwardCollisionWarning(cfg.warning)
        self.controller = GapController(cfg.control) if with_control else None

    def step(self, frame, dt=None):
        """frame: svs.data.recording.FrameData; returns a StepResult."""
        dt = self.meta.dt if dt is None else dt
        cfg = self.cfg

        detections = frame.detections
        if self.meta.replay_events:
            detections = apply_camera_events(detections, self.meta.events, frame.timestamp, self.camera_model.width)
        camera = self.camera_tracker.update(frame.frame, frame.timestamp, {"detections": detections})
        points = self.radar_degrader.apply(frame.radar_points, frame.ego_speed_mps, dt)
        radar = extract_radar_measurements(points) if len(points) else None

        camera_gap = from_camera(camera, cfg.fusion.camera_sigma_scale)
        radar_gap = from_radar(radar, self.meta.radar_bumper_offset_m, cfg.radar_perception,
                               cfg.fusion.radar_sigma_scale)

        estimate = self.estimator.step(dt, camera=camera_gap, radar=radar_gap)
        warning = self.warning.step(dt, estimate, frame.ego_speed_mps)
        command = self.controller.step(dt, estimate, frame.ego_speed_mps) if self.controller else None
        return StepResult(frame.timestamp, camera, radar, camera_gap, radar_gap, estimate, warning, command,
                          self.camera_tracker.detections)


def replay(recording, cfg, mode, seed=None):
    """Run the pipeline over a whole recording (open loop); returns the list of StepResults."""
    seed = recording.meta.seed if seed is None else seed
    pipeline = Pipeline(cfg, recording.meta, mode, seed)
    return [pipeline.step(frame) for frame in recording]
