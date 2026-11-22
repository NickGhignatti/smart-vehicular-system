"""A tiny fake CARLA server for smoke tests of the simulator apps (no Unreal needed).

World: a straight road along +x with two same-direction lanes (lane -1 at y=0, lane -2 at
y=+3.5). Vehicles are point masses driven by throttle/brake. Sensors produce data on every
tick: an RGB image (blank), an instance-segmentation image with every vehicle painted as
its box, and radar detections on the rear of every vehicle ahead. Real carla value types
(Location, Transform, Vector3D, ...) are used, so the code under test runs unchanged.

Usage: monkeypatch.setattr(carla, "Client", FakeClient)
"""

import itertools
import math
from types import SimpleNamespace as NS

import carla
import numpy as np

from svs.perception.camera_geometry import PinholeCamera

ROAD_LENGTH = 2000.0
LANES = {-1: 0.0, -2: 3.5}
_ids = itertools.count(100)


class FakeAttribute:
    def __init__(self, value):
        self.value = value
        self.recommended_values = ["255,0,0"]

    def as_int(self):
        return int(self.value)


class FakeBlueprint:
    def __init__(self, bp_id):
        self.id = bp_id
        self.attrs = {"role_name": FakeAttribute("")}
        if bp_id.startswith("vehicle."):
            self.attrs.update(number_of_wheels=FakeAttribute("4"), color=FakeAttribute("255,0,0"))
        else:
            for key in ("image_size_x", "image_size_y", "fov", "sensor_tick", "horizontal_fov", "vertical_fov",
                        "points_per_second", "range"):
                self.attrs[key] = FakeAttribute("0")

    def has_attribute(self, key):
        return key in self.attrs

    def set_attribute(self, key, value):
        self.attrs[key] = FakeAttribute(value)

    def get_attribute(self, key):
        return self.attrs[key]


class FakeLibrary:
    IDS = ["vehicle.tesla.model3", "vehicle.audi.tt", "vehicle.audi.a2", "vehicle.lincoln.mkz_2020",
           "sensor.camera.rgb", "sensor.camera.instance_segmentation", "sensor.other.radar"]

    def find(self, bp_id):
        if bp_id not in self.IDS:
            raise IndexError(bp_id)
        return FakeBlueprint(bp_id)

    def filter(self, pattern):
        prefix = pattern.rstrip("*")
        return [FakeBlueprint(i) for i in self.IDS if i.startswith(prefix)]


class FakeWaypoint:
    def __init__(self, x, lane_id=-1):
        self.x = max(0.0, min(ROAD_LENGTH, x))
        self.lane_id = lane_id
        self.road_id = 1
        self.lane_type = carla.LaneType.Driving
        self.is_junction = False
        self.transform = carla.Transform(carla.Location(x=self.x, y=LANES[lane_id], z=0.0), carla.Rotation(yaw=0.0))

    def next(self, d):
        return [FakeWaypoint(self.x + d, self.lane_id)] if self.x + d < ROAD_LENGTH else []

    def previous(self, d):
        return [FakeWaypoint(self.x - d, self.lane_id)]

    def get_left_lane(self):
        return FakeWaypoint(self.x, -1) if self.lane_id == -2 else None

    def get_right_lane(self):
        return FakeWaypoint(self.x, -2) if self.lane_id == -1 else None


class FakeMap:
    name = "Carla/Maps/FakeTown"

    def get_spawn_points(self):
        return [carla.Transform(carla.Location(x=x, y=0.0, z=0.5)) for x in (20.0, 300.0, 600.0)]

    def get_waypoint(self, location, project_to_road=True, lane_type=None):
        lane = min(LANES, key=lambda k: abs(LANES[k] - location.y))
        return FakeWaypoint(location.x, lane)


class FakeActor:
    def __init__(self, world, bp, transform, parent=None):
        self.world, self.id, self.type_id, self.parent = world, next(_ids), bp.id, parent
        self.attributes = {"role_name": bp.attrs["role_name"].value}
        self.local = transform
        self.alive = True

    def destroy(self):
        self.alive = False
        if self in self.world.actors:
            self.world.actors.remove(self)


class FakeVehicle(FakeActor):
    def __init__(self, world, bp, transform):
        super().__init__(world, bp, transform)
        self.x, self.y, self.v = transform.location.x, transform.location.y, 0.0
        self.control = carla.VehicleControl()
        self.bounding_box = NS(location=carla.Location(z=0.7), extent=carla.Vector3D(2.4, 0.95, 0.75))

    def get_transform(self):
        return carla.Transform(carla.Location(x=self.x, y=self.y, z=0.0), carla.Rotation(yaw=0.0))

    def get_location(self):
        return self.get_transform().location

    def get_velocity(self):
        return carla.Vector3D(x=self.v)

    def apply_control(self, control):
        self.control = control

    def set_autopilot(self, enabled=True, *args):
        pass

    def set_target_velocity(self, velocity):
        self.v = velocity.x

    def physics(self, dt):
        c = self.control
        accel = 3.5 * c.throttle - 8.0 * max(c.brake, 1.0 if c.hand_brake else 0.0) - 0.15 * (self.v > 0)
        self.v = max(0.0, self.v + accel * dt)
        self.x += self.v * dt
        # Steering pulls the car towards the lane it aims at.
        self.y += 3.0 * c.steer * dt * min(1.0, self.v / 3.0)


class FakeSensor(FakeActor):
    def __init__(self, world, bp, transform, parent):
        super().__init__(world, bp, transform, parent)
        self.attrs = {k: a.value for k, a in bp.attrs.items()}
        self.callback = None

    def listen(self, callback):
        self.callback = callback

    def stop(self):
        self.callback = None

    def get_transform(self):
        p = self.parent.get_transform().location
        return carla.Transform(carla.Location(x=p.x + self.local.location.x, y=p.y, z=self.local.location.z),
                               self.local.rotation)

    def emit(self, frame, timestamp):
        if self.callback is None:
            return
        if self.type_id.startswith("sensor.camera"):
            w, h = int(float(self.attrs["image_size_x"])), int(float(self.attrs["image_size_y"]))
            img = np.zeros((h, w, 4), np.uint8)
            if self.type_id.endswith("instance_segmentation"):
                self._paint_vehicles(img)
            self.callback(NS(frame=frame, timestamp=timestamp, raw_data=img.tobytes(), width=w, height=h))
        else:
            self.callback(_RadarData(frame, timestamp, self._radar_detections()))

    def _others(self):
        return [a for a in self.world.actors if isinstance(a, FakeVehicle) and a is not self.parent]

    def _paint_vehicles(self, img):
        h, w = img.shape[:2]
        cam = PinholeCamera(w, h, float(self.attrs["fov"]), self.local.location.z, self.local.rotation.pitch)
        me = self.get_transform().location
        for car in sorted(self._others(), key=lambda c: -c.x):        # far first, near paints over
            rear = car.x - 2.4 - me.x
            if rear < 1.0:
                continue
            corners = np.array([[rear + dx, car.y - me.y + dy, dz - cam.mount_height_m]
                                for dx in (0, 4.8) for dy in (-0.95, 0.95) for dz in (0.0, 1.5)])
            uv = cam.project(corners)
            u1, v1 = np.clip(np.nanmin(uv, axis=0), 0, [w, h]).astype(int)
            u2, v2 = np.clip(np.nanmax(uv, axis=0), 0, [w, h]).astype(int)
            img[v1:v2, u1:u2, 2] = 14
            img[v1:v2, u1:u2, 1] = car.id & 0xFF
            img[v1:v2, u1:u2, 0] = car.id >> 8

    def _radar_detections(self):
        me = self.get_transform().location
        out = []
        for car in self._others():
            x = car.x - 2.4 - me.x
            if not 1.0 < x < 80.0:
                continue
            for dy in (-0.6, 0.0, 0.6):
                y, z = car.y - me.y + dy, -0.3
                depth = math.sqrt(x * x + y * y + z * z)
                az, alt = math.atan2(y, x), math.asin(z / depth)
                vel = -(car.v - self.parent.v) * math.cos(az) * math.cos(alt)   # CARLA: + towards sensor
                out.append(NS(depth=depth, azimuth=az, altitude=alt, velocity=vel))
        return out


class _RadarData(list):
    def __init__(self, frame, timestamp, detections):
        super().__init__(detections)
        self.frame, self.timestamp = frame, timestamp


class FakeWorld:
    def __init__(self):
        self.map = FakeMap()
        self.actors = []
        self.settings = NS(synchronous_mode=False, fixed_delta_seconds=None, no_rendering_mode=False)
        self.frame = 1
        self.spectator = NS(set_transform=lambda tf: None)
        self.debug = NS(draw_string=lambda *a, **k: None, draw_point=lambda *a, **k: None)

    def get_map(self):
        return self.map

    def get_settings(self):
        return NS(**vars(self.settings))

    def apply_settings(self, settings):
        self.settings = NS(**vars(settings))

    def get_blueprint_library(self):
        return FakeLibrary()

    def get_spectator(self):
        return self.spectator

    def set_weather(self, weather):
        self.weather = weather

    def get_actors(self):
        return list(self.actors)

    def try_spawn_actor(self, bp, transform, attach_to=None):
        if bp.id.startswith("vehicle."):
            for a in self.actors:
                if isinstance(a, FakeVehicle) and abs(a.x - transform.location.x) < 5 and abs(a.y - transform.location.y) < 1:
                    return None
            actor = FakeVehicle(self, bp, transform)
        else:
            actor = FakeSensor(self, bp, transform, attach_to)
        self.actors.append(actor)
        return actor

    def spawn_actor(self, bp, transform, attach_to=None):
        actor = self.try_spawn_actor(bp, transform, attach_to)
        if actor is None:
            raise RuntimeError("Spawn failed because of collision at spawn position")
        return actor

    def tick(self, seconds=10.0):
        dt = self.settings.fixed_delta_seconds or 0.05
        for a in list(self.actors):
            if isinstance(a, FakeVehicle):
                a.physics(dt)
        self.frame += 1
        for a in list(self.actors):
            if isinstance(a, FakeSensor):
                a.emit(self.frame, self.frame * dt)
        return self.frame

    def wait_for_tick(self, seconds=10.0):
        return NS(frame=self.tick())


class FakeClient:
    world = None

    def __init__(self, host="localhost", port=2000):
        if FakeClient.world is None:
            FakeClient.world = FakeWorld()

    def set_timeout(self, seconds):
        pass

    def get_client_version(self):
        return "0.9.16"

    def get_server_version(self):
        return "0.9.16"

    def get_world(self):
        return FakeClient.world

    def load_world(self, name):
        return FakeClient.world

    def get_available_maps(self):
        return ["/Game/Carla/Maps/FakeTown"]

    def get_trafficmanager(self, port=8000):
        return NS(set_synchronous_mode=lambda enabled: None)
