"""`python -m svs check`: verify the connection to the simulator without changing anything.

Read-only: no map load, no settings change, no actors spawned, so it is safe to run
against a server other people are using.
"""

import logging
import time
from collections import Counter

from svs.sim.session import CarlaSession

log = logging.getLogger(__name__)


def run(cfg):
    with CarlaSession(cfg.carla, synchronous=False, load_town=False) as sim:
        samples = []
        for _ in range(5):
            t0 = time.perf_counter()
            sim.client.get_server_version()
            samples.append((time.perf_counter() - t0) * 1000.0)
        log.info("RPC round-trip: avg %.1f ms, max %.1f ms", sum(samples) / len(samples), max(samples))

        settings = sim.world.get_settings()
        log.info("Server settings: synchronous=%s fixed_delta=%s no_rendering=%s",
                 settings.synchronous_mode, settings.fixed_delta_seconds, settings.no_rendering_mode)
        if settings.synchronous_mode:
            log.warning("Server is already in synchronous mode: someone else may be running, "
                        "or a previous run did not restore its settings.")

        actors = sim.world.get_actors()
        kinds = Counter(a.type_id.split(".")[0] for a in actors)
        log.info("Actors on server: %d (%s)", len(actors), ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())))
        ours = [a for a in actors if a.attributes.get("role_name", "").startswith("svs_")]
        if ours:
            log.warning("%d actors from this project are still alive (role_name svs_*)", len(ours))

        maps = sorted(m.split("/")[-1] for m in sim.client.get_available_maps())
        log.info("Available maps: %s", ", ".join(maps))
        log.info("Spawn points on current map: %d", len(sim.world.get_map().get_spawn_points()))
    log.info("Connection check OK")
