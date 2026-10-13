"""Connection to the CARLA server and lifecycle of one simulation run."""

import logging

import carla

log = logging.getLogger(__name__)


class CarlaSession:
    """Context manager owning the client, the world settings and every spawned actor.

    On exit it stops sensors, destroys tracked actors and restores the original world
    settings, so an interrupted run never leaves the (shared) server stuck in
    synchronous mode waiting for ticks that will never come.
    """

    def __init__(self, cfg, synchronous=True, load_town=True):
        self.cfg = cfg
        self.synchronous = synchronous
        self.load_town = load_town
        self.client = None
        self.world = None
        self._traffic_manager = None
        self._original_settings = None
        self._actors = []

    def __enter__(self):
        try:
            self._connect()
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def _connect(self):
        self.client = carla.Client(self.cfg.host, self.cfg.port)
        self.client.set_timeout(self.cfg.timeout)

        client_version = self.client.get_client_version()
        server_version = self.client.get_server_version()
        log.info("Connected to %s:%d (client %s, server %s)",
                 self.cfg.host, self.cfg.port, client_version, server_version)
        if client_version != server_version:
            log.warning("Client/server version mismatch: %s vs %s", client_version, server_version)

        self.world = self.client.get_world()
        town = self.cfg.town
        if self.load_town and town and not self.world.get_map().name.endswith(town):
            log.info("Loading map %s", town)
            self.world = self.client.load_world(town)
        log.info("Map: %s", self.world.get_map().name)

        if self.synchronous:
            self._original_settings = self.world.get_settings()
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = self.cfg.fixed_delta_seconds
            self.world.apply_settings(settings)
            log.info("Synchronous mode on, dt=%.3fs", self.cfg.fixed_delta_seconds)

    @property
    def traffic_manager(self):
        """Created on first use: only needed when some vehicle runs on autopilot."""
        if self._traffic_manager is None:
            self._traffic_manager = self.client.get_trafficmanager(self.cfg.tm_port)
            self._traffic_manager.set_synchronous_mode(self.synchronous)
        return self._traffic_manager

    @property
    def spectator(self):
        return self.world.get_spectator()

    @property
    def dt(self):
        return self.cfg.fixed_delta_seconds

    def track(self, actor):
        """Register an actor for cleanup and return it."""
        self._actors.append(actor)
        return actor

    def tick(self):
        """Advance the simulation one step and return the new frame id."""
        if self.synchronous:
            return self.world.tick(self.cfg.timeout)
        return self.world.wait_for_tick(self.cfg.timeout).frame

    def close(self):
        # Reverse order: sensors were spawned after (and attached to) their vehicles.
        for actor in reversed(self._actors):
            if isinstance(actor, carla.Sensor):
                try:
                    actor.stop()
                except RuntimeError:
                    pass
            try:
                actor.destroy()
            except RuntimeError:
                pass
        if self._actors:
            log.info("Destroyed %d actors", len(self._actors))
        self._actors = []

        if self._original_settings is not None and self.world is not None:
            try:
                self.world.apply_settings(self._original_settings)
                log.info("Original world settings restored")
            except RuntimeError as err:
                log.error("Could not restore world settings: %s", err)
            self._original_settings = None

        if self._traffic_manager is not None:
            try:
                self._traffic_manager.set_synchronous_mode(False)
            except RuntimeError:
                pass
            self._traffic_manager = None
