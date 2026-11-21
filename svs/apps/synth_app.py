"""`python -m svs synth`: generate synthetic recordings (no simulator needed).

For development and dry-run tuning only; report numbers must come from CARLA recordings.
"""

import logging

from svs.synth import generate

log = logging.getLogger(__name__)


def run(cfg, scenarios, seeds, duration_s, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for scenario in scenarios:
        for seed in seeds:
            recording = generate(scenario, seed, cfg, duration_s)
            path = recording.save(out_dir / f"synth_{scenario}_{seed}.npz")
            log.info("%s: %d frames -> %s", scenario, len(recording), path)
            paths.append(path)
    return paths
