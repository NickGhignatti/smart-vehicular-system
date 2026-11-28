"""Command-line entrypoint: `python -m svs <command>`.

Simulator commands (need a CARLA server):
  check       read-only connection check
  sensors     ego + parked lead, camera and radar vs ground truth (smoke test)
  camera      camera lead detection + distance vs ground truth, CSV log
  run         a scenario with the full pipeline live: HUD, warning, optional ACC; saves a recording

Offline commands (no simulator):
  synth       generate synthetic recordings (development / dry runs only)
  evaluate    replay recordings in every mode: metrics table + report figures
  calibrate   sensor bias / noise from recordings -> suggested config values
  tune        grid search of fusion / warning parameters on recordings

Any config value can be changed without editing code:
  --set fusion.sigma_accel_mps2=2.0 --set warning.ttc_on_s=2.2
  --config-file calibration.json --config-file best.json   (later files win, --set wins over files)
"""

import argparse
import dataclasses
import glob
import logging
import sys
from pathlib import Path

from svs.config import load_config, load_overrides_file, with_overrides
from svs.fusion.tracker import MODES
from svs.scenarios import SCENARIOS


def _seeds(text):
    """"3" -> [3]; "1-5" -> [1..5]; "1,4,9" -> [1, 4, 9]."""
    seeds = []
    for part in text.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    return seeds


def _recordings(patterns):
    """Expand files, directories (all .npz inside) and globs (PowerShell does not expand them)."""
    paths = []
    for pattern in patterns:
        p = Path(pattern)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.npz")))
        elif any(ch in pattern for ch in "*?["):
            paths.extend(Path(x) for x in sorted(glob.glob(pattern)))
        else:
            paths.append(p)
    missing = [str(p) for p in paths if not p.exists()]
    if missing or not paths:
        raise SystemExit(f"No recordings found for {patterns}" + (f" (missing: {missing})" if missing else ""))
    return paths


def build_parser():
    parser = argparse.ArgumentParser(prog="python -m svs", description="Lead-vehicle sensor fusion in CARLA")
    parser.add_argument("--host", help="CARLA server host (overrides CARLA_HOST)")
    parser.add_argument("--port", type=int, help="CARLA server port (overrides CARLA_PORT)")
    parser.add_argument("--town", help="map to load, e.g. Town04 (overrides CARLA_TOWN)")
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.FIELD=VALUE",
                        help="override a config value, e.g. --set fusion.sigma_accel_mps2=2.0 (repeatable)")
    parser.add_argument("--config-file", type=Path, action="append", default=[],
                        help="JSON of config overrides (from calibrate / tune); repeatable, later files win")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="read-only connection check")

    sensors = sub.add_parser("sensors", help="camera + radar smoke test with ground truth")
    sensors.add_argument("--seconds", type=float, default=40.0, help="simulated duration")
    sensors.add_argument("--no-display", action="store_true", help="no OpenCV window (headless)")
    sensors.add_argument("--lateral", type=float, default=0.0, help="lead lateral offset in m (+ = right of lane centre)")
    sensors.add_argument("--record", action="store_true", help="save the HUD video to OUTPUT_DIR")

    camera = sub.add_parser("camera", help="camera lead detection + distance vs ground truth")
    camera.add_argument("--seconds", type=float, default=60.0, help="simulated duration")
    camera.add_argument("--detector", choices=("instance", "yolo"), help="override CameraPerceptionConfig.detector")
    camera.add_argument("--no-display", action="store_true", help="no OpenCV window (headless)")
    camera.add_argument("--record", action="store_true", help="save the HUD video to OUTPUT_DIR")

    run = sub.add_parser("run", help="scenario with the live pipeline; saves a recording for evaluate")
    run.add_argument("--scenario", choices=SCENARIOS, default="stop_and_go")
    run.add_argument("--seed", type=int, default=None, help="scenario seed (default: SEED from .env)")
    run.add_argument("--seconds", type=float, default=90.0, help="simulated duration")
    run.add_argument("--mode", choices=MODES, default="fused", help="estimator shown on the HUD / used by ACC")
    run.add_argument("--control", choices=("warn", "acc"), default="warn",
                     help="warn: a scripted driver drives the ego; acc: the gap controller drives it")
    run.add_argument("--detector", choices=("instance", "yolo"), help="override CameraPerceptionConfig.detector")
    run.add_argument("--weather", default="ClearNoon", help="carla.WeatherParameters preset (Lab2), e.g. HardRainNoon")
    run.add_argument("--no-display", action="store_true", help="no OpenCV window (headless)")
    run.add_argument("--video", action="store_true", help="also save the HUD video (demo deliverable)")

    synth = sub.add_parser("synth", help="generate synthetic recordings (no simulator)")
    synth.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    synth.add_argument("--seeds", type=_seeds, default=_seeds("1-3"), help='e.g. "1-5" or "1,4,9"')
    synth.add_argument("--seconds", type=float, default=90.0)
    synth.add_argument("--out", type=Path, help="default: OUTPUT_DIR/recordings/synthetic")

    evaluate = sub.add_parser("evaluate", help="metrics + figures for recordings, all modes")
    evaluate.add_argument("recordings", nargs="+", help=".npz files, directories or globs")
    evaluate.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    evaluate.add_argument("--out", type=Path, help="default: OUTPUT_DIR/report")
    evaluate.add_argument("--no-plots", action="store_true")

    calibrate = sub.add_parser("calibrate", help="sensor bias/noise from recordings -> suggested config")
    calibrate.add_argument("recordings", nargs="+", help="best: `run --scenario approach` recordings")
    calibrate.add_argument("--out", type=Path, help="default: OUTPUT_DIR/calibration")

    tune = sub.add_parser("tune", help="grid search of fusion / warning parameters")
    tune.add_argument("recordings", nargs="+", help="the TUNING set (keep separate test recordings!)")
    tune.add_argument("--out", type=Path, help="default: OUTPUT_DIR/tuning")
    tune.add_argument("--quick", action="store_true", help="smaller grid")
    return parser


def build_config(args):
    cfg = load_config()
    overrides = {k: v for k, v in (("host", args.host), ("port", args.port), ("town", args.town)) if v is not None}
    if overrides:
        cfg = dataclasses.replace(cfg, carla=dataclasses.replace(cfg.carla, **overrides))
    for path in args.config_file:
        cfg = with_overrides(cfg, load_overrides_file(path))
    pairs = {}
    for item in args.set:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--set expects SECTION.FIELD=VALUE, got {item!r}")
        pairs[key.strip()] = value.strip()
    return with_overrides(cfg, pairs) if pairs else cfg


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg = build_config(args)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else cfg.log_level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("svs")
    out = cfg.output_dir

    if args.command in ("check", "sensors", "camera", "run"):
        log.info("Target server %s:%d", cfg.carla.host, cfg.carla.port)
    try:
        if args.command == "check":
            from svs.apps import check
            check.run(cfg)
        elif args.command == "sensors":
            from svs.apps import sensor_check
            record_path = out / "videos" / "sensor_check.mp4" if args.record else None
            sensor_check.run(cfg, seconds=args.seconds, display=not args.no_display, record_path=record_path,
                             lateral_offset_m=args.lateral)
        elif args.command == "camera":
            from svs.apps import camera_check
            record_path = out / "videos" / "camera_check.mp4" if args.record else None
            camera_check.run(cfg, seconds=args.seconds, display=not args.no_display, record_path=record_path,
                             detector_name=args.detector)
        elif args.command == "run":
            from svs.apps import run_scenario
            run_scenario.run(cfg, scenario=args.scenario, seed=cfg.seed if args.seed is None else args.seed,
                             seconds=args.seconds, mode=args.mode, control=args.control,
                             detector_name=args.detector, weather=args.weather,
                             display=not args.no_display, video=args.video)
        elif args.command == "synth":
            from svs.apps import synth_app
            synth_app.run(cfg, args.scenarios, args.seeds, args.seconds, args.out or out / "recordings" / "synthetic")
        elif args.command == "evaluate":
            from svs.apps import evaluate
            evaluate.run(cfg, _recordings(args.recordings), args.out or out / "report", args.modes,
                         plots=not args.no_plots)
        elif args.command == "calibrate":
            from svs.apps import calibrate
            calibrate.run(cfg, _recordings(args.recordings), args.out or out / "calibration")
        elif args.command == "tune":
            from svs.apps import tune
            tune.run(cfg, _recordings(args.recordings), args.out or out / "tuning", quick=args.quick)
    except RuntimeError as err:
        # carla raises RuntimeError for timeouts / unreachable server.
        log.error("%s", err)
        if args.command in ("check", "sensors", "camera", "run"):
            log.error("Is the simulator running and reachable at %s:%d? Check .env.", cfg.carla.host, cfg.carla.port)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
