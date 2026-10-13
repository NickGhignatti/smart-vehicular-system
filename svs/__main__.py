"""Command-line entrypoint: `python -m svs <command>`.

Commands:
  check     read-only connection check against the simulator
  sensors   spawn ego + lead, stream camera and radar, compare with ground truth
"""

import argparse
import dataclasses
import logging
import sys

from svs.config import load_config


def build_parser():
    parser = argparse.ArgumentParser(prog="python -m svs", description="Lead-vehicle sensor fusion in CARLA")
    parser.add_argument("--host", help="CARLA server host (overrides CARLA_HOST)")
    parser.add_argument("--port", type=int, help="CARLA server port (overrides CARLA_PORT)")
    parser.add_argument("--town", help="map to load, e.g. Town04 (overrides CARLA_TOWN)")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="read-only connection check")

    sensors = sub.add_parser("sensors", help="camera + radar smoke test with ground truth")
    sensors.add_argument("--seconds", type=float, default=40.0, help="simulated duration")
    sensors.add_argument("--no-display", action="store_true", help="no OpenCV window (headless)")
    sensors.add_argument("--record", action="store_true", help="save the HUD video to OUTPUT_DIR")
    return parser


def apply_overrides(cfg, args):
    overrides = {k: v for k, v in (("host", args.host), ("port", args.port), ("town", args.town)) if v is not None}
    if overrides:
        cfg = dataclasses.replace(cfg, carla=dataclasses.replace(cfg.carla, **overrides))
    return cfg


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg = apply_overrides(load_config(), args)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else cfg.log_level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("svs")
    log.info("Target server %s:%d", cfg.carla.host, cfg.carla.port)

    try:
        if args.command == "check":
            from svs.apps import check
            check.run(cfg)
        elif args.command == "sensors":
            from svs.apps import sensor_check
            record_path = cfg.output_dir / "videos" / "sensor_check.mp4" if args.record else None
            sensor_check.run(cfg, seconds=args.seconds, display=not args.no_display, record_path=record_path)
    except RuntimeError as err:
        # carla raises RuntimeError for timeouts / unreachable server.
        log.error("%s", err)
        log.error("Is the simulator running and reachable at %s:%d? Check .env.", cfg.carla.host, cfg.carla.port)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
