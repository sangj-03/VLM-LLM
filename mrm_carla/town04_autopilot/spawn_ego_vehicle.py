#!/usr/bin/env python3

"""Spawn one persistent Town04 ego vehicle for normal driving and MRM tests."""

import argparse
import glob
import os
import random
import signal
import sys
import threading
import time


CARLA_PYTHONAPI_DIR = "/home/tmo/carla/PythonAPI"
sys.path.insert(0, os.path.join(CARLA_PYTHONAPI_DIR, "carla"))
egg_matches = glob.glob(os.path.join(
    CARLA_PYTHONAPI_DIR,
    "carla/dist/carla-*%d.%d-linux-x86_64.egg" % (
        sys.version_info.major, sys.version_info.minor)))
if egg_matches:
    sys.path.insert(0, egg_matches[0])

import carla  # noqa: E402


CAMERA_SPECS = (
    ("rgb_front", 1.3, 0.0, 2.3, 0.0, 1200, 900),
    ("rgb_left", 1.3, 0.0, 2.3, -60.0, 400, 300),
    ("rgb_right", 1.3, 0.0, 2.3, 60.0, 400, 300),
    ("rgb_rear", -1.3, 0.0, 2.3, 180.0, 400, 300),
)


def attribute_value(attribute):
    if hasattr(attribute, "as_string"):
        return attribute.as_string()
    return str(attribute)


def actor_alive(actor):
    if actor is None:
        return False
    try:
        return bool(actor.is_alive)
    except RuntimeError:
        return False


def wait_for_actor_alive(world, actor, timeout=5.0):
    """Wait through the spawn/tick race before entering the retention loop."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if actor_alive(actor):
            return
        try:
            world.wait_for_tick(0.5)
        except RuntimeError:
            time.sleep(0.05)
    raise RuntimeError(
        "Spawned ego actor %s did not become alive within %.1fs" % (
            getattr(actor, "id", "unknown"), timeout))


def wait_for_world(host, port):
    """Wait indefinitely so this terminal can be opened with the server terminal."""
    client = carla.Client(host, port)
    client.set_timeout(2.0)
    printed = False
    while True:
        try:
            return client, client.get_world()
        except RuntimeError:
            if not printed:
                print("CARLA server waiting at %s:%d ..." % (host, port), flush=True)
                printed = True
            time.sleep(1.0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Spawn and retain one ego vehicle in Town04_Opt")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--role-name", default="town06_ego")
    parser.add_argument("--vehicle", default="vehicle.tesla.model3")
    parser.add_argument(
        "--spawn-index", type=int, default=None,
        help=("explicit CARLA spawn index; by default choose a spawn whose "
              "outward driving lanes reach a wide Shoulder"))
    parser.add_argument(
        "--minimum-shoulder-width-m", type=float, default=3.0,
        help="minimum Shoulder width used by automatic spawn selection")
    parser.add_argument("--npc-vehicles", type=int, default=0)
    parser.add_argument("--npc-walkers", type=int, default=0)
    parser.add_argument("--npc-tm-port", type=int, default=8002)
    parser.add_argument("--npc-speed-difference", type=float, default=0.0,
                        help=(
                            "NPC speed difference percentage. Positive is "
                            "slower than the speed limit; 0 follows it."))
    parser.add_argument("--no-npc", action="store_true")
    parser.add_argument("--no-cameras", action="store_true",
                        help="spawn only the ego vehicle, without its RGB cameras")
    parser.add_argument("--camera-tick", type=float, default=0.05,
                        help="RGB camera capture period in seconds")
    parser.add_argument("--camera-view", action="store_true",
                        help="show the four ego RGB cameras in an OpenCV window")
    parser.add_argument("--camera-window-name", default="Town04 Ego Cameras",
                        help="OpenCV window title used with --camera-view")
    return parser.parse_args()


def vehicle_speed(actor):
    velocity = actor.get_velocity()
    return 3.6 * (
        velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) ** 0.5


def same_direction(source, candidate):
    source_forward = source.transform.get_forward_vector()
    candidate_forward = candidate.transform.get_forward_vector()
    return (source_forward.x * candidate_forward.x
            + source_forward.y * candidate_forward.y
            + source_forward.z * candidate_forward.z) >= 0.8


def outward_neighbor(waypoint, lane_type):
    """Return the adjacent same-direction lane farther from road centre."""
    for candidate in (waypoint.get_right_lane(), waypoint.get_left_lane()):
        if (candidate is None
                or candidate.lane_type != lane_type
                or candidate.road_id != waypoint.road_id
                or not same_direction(waypoint, candidate)):
            continue
        if (candidate.lane_id == 0
                or candidate.lane_id * waypoint.lane_id <= 0
                or abs(candidate.lane_id) <= abs(waypoint.lane_id)):
            continue
        return candidate
    return None


def reachable_wide_shoulder(waypoint, minimum_width):
    """Find a wide Shoulder outside this lane without crossing road centre."""
    current = waypoint
    visited = set()
    while current is not None and current.lane_type == carla.LaneType.Driving:
        key = (current.road_id, current.section_id, current.lane_id)
        if key in visited:
            return None
        visited.add(key)
        shoulder = outward_neighbor(current, carla.LaneType.Shoulder)
        if (shoulder is not None
                and float(shoulder.lane_width) >= minimum_width
                and not shoulder.is_junction):
            return shoulder
        current = outward_neighbor(current, carla.LaneType.Driving)
    return None


def shoulder_roadside_clearance(world, shoulder, ego_half_width,
                                minimum_shoulder_width,
                                inward_bias=0.55):
    """Measure the smallest physical rail margin along the merge corridor."""
    left = shoulder.get_left_lane()
    right = shoulder.get_right_lane()
    if (left is not None and left.lane_type == carla.LaneType.Driving
            and same_direction(shoulder, left)):
        inward_offset = -inward_bias
    elif (right is not None and right.lane_type == carla.LaneType.Driving
          and same_direction(shoulder, right)):
        inward_offset = inward_bias
    else:
        return float("-inf")
    outward_sign = 1.0 if inward_offset < 0.0 else -1.0
    obstacle_labels = {
        carla.CityObjectLabel.GuardRail,
        carla.CityObjectLabel.Fences,
        carla.CityObjectLabel.Walls,
        carla.CityObjectLabel.Poles,
        carla.CityObjectLabel.Static,
        carla.CityObjectLabel.Other,
        carla.CityObjectLabel.Roads,
        carla.CityObjectLabel.Sidewalks,
        carla.CityObjectLabel.Terrain,
        carla.CityObjectLabel.Bridge,
    }
    minimum_clearance = float("inf")
    distance = 0.0
    while distance <= 60.0:
        if distance == 0.0:
            candidate = shoulder
        else:
            candidate = next((item for item in shoulder.next(distance)
                              if item.road_id == shoulder.road_id
                              and item.lane_id == shoulder.lane_id
                              and float(item.lane_width)
                              >= minimum_shoulder_width), None)
        if candidate is None:
            return float("-inf")
        lane_right = candidate.transform.get_right_vector()
        centre = candidate.transform.location + carla.Location(
            x=lane_right.x * inward_offset,
            y=lane_right.y * inward_offset)
        for height in (0.25, 0.40, 0.70, 1.05):
            start = carla.Location(
                x=centre.x, y=centre.y, z=centre.z + height)
            end = carla.Location(
                x=start.x + lane_right.x * outward_sign * 4.0,
                y=start.y + lane_right.y * outward_sign * 4.0,
                z=start.z)
            try:
                hits = world.cast_ray(start, end)
            except (AttributeError, RuntimeError, TypeError):
                return float("-inf")
            for hit in hits:
                if hit.label in obstacle_labels:
                    minimum_clearance = min(
                        minimum_clearance,
                        start.distance(hit.location) - ego_half_width)
        distance += 3.0
    return minimum_clearance


def ordered_spawn_candidates(world, carla_map, spawn_points, explicit_index,
                             minimum_shoulder_width):
    indexed = list(enumerate(spawn_points))
    if explicit_index is not None:
        start = explicit_index % len(indexed)
        return indexed[start:] + indexed[:start], 0

    ranked = []
    ego_half_width = max(1.0, (minimum_shoulder_width - 0.8) * 0.5)
    for item in indexed:
        index, transform = item
        waypoint = carla_map.get_waypoint(
            transform.location, project_to_road=True,
            lane_type=carla.LaneType.Driving)
        shoulder = (reachable_wide_shoulder(
            waypoint, minimum_shoulder_width)
            if waypoint is not None and not waypoint.is_junction else None)
        if shoulder is not None:
            clearance = shoulder_roadside_clearance(
                world, shoulder, ego_half_width,
                minimum_shoulder_width)
            if clearance >= 0.25:
                ranked.append((clearance, index, item))
    ranked.sort(key=lambda entry: (-entry[0], entry[1]))
    preferred = [entry[2] for entry in ranked]
    return preferred, len(preferred)


def actor_blueprints(world, pattern, generation="All"):
    blueprints = world.get_blueprint_library().filter(pattern)
    if generation.lower() == "all":
        return list(blueprints)
    selected = []
    for blueprint in blueprints:
        if not blueprint.has_attribute("generation"):
            continue
        if attribute_value(blueprint.get_attribute("generation")) == generation:
            selected.append(blueprint)
    return selected


def spawn_npc_vehicles(world, traffic_manager, ego_vehicle, count):
    if count <= 0:
        return []
    blueprints = actor_blueprints(world, "vehicle.*", "All")
    safe_blueprints = [
        blueprint for blueprint in blueprints
        if blueprint.has_attribute("base_type")
        and attribute_value(blueprint.get_attribute("base_type")) == "car"
    ]
    if safe_blueprints:
        blueprints = safe_blueprints
    if not blueprints:
        print("NPC vehicles skipped: no vehicle blueprints found.", flush=True)
        return []

    ego_location = ego_vehicle.get_location()
    spawn_points = list(world.get_map().get_spawn_points())
    nearby_spawn_points = [
        transform for transform in spawn_points
        if 12.0 <= transform.location.distance(ego_location) <= 120.0
    ]
    far_spawn_points = [
        transform for transform in spawn_points
        if transform not in nearby_spawn_points
    ]
    random.shuffle(nearby_spawn_points)
    random.shuffle(far_spawn_points)
    spawn_points = nearby_spawn_points + far_spawn_points
    spawned = []
    for transform in spawn_points:
        if len(spawned) >= count:
            break
        if transform.location.distance(ego_location) < 10.0:
            continue
        blueprint = random.choice(blueprints)
        # The local 0.9.15-dirty server rejects some otherwise valid CARLA
        # colour attribute values while serialising actors ("invalid color").
        # Leave the blueprint default colour untouched for compatibility.
        if blueprint.has_attribute("driver_id"):
            blueprint.set_attribute(
                "driver_id",
                random.choice(
                    blueprint.get_attribute("driver_id").recommended_values))
        blueprint.set_attribute("role_name", "autopilot")
        actor = world.try_spawn_actor(blueprint, transform)
        if actor is None:
            continue
        actor.set_autopilot(True, traffic_manager.get_port())
        spawned.append(actor)
    if len(spawned) < count:
        print(
            "NPC vehicles: spawned %d/%d; remaining spawn points were occupied."
            % (len(spawned), count),
            flush=True)
    return spawned


def spawn_npc_walkers(client, world, ego_vehicle, count):
    if count <= 0:
        return [], []
    blueprints = actor_blueprints(world, "walker.pedestrian.*", "2")
    if not blueprints:
        blueprints = actor_blueprints(world, "walker.pedestrian.*", "All")
    if not blueprints:
        print("NPC walkers skipped: no walker blueprints found.", flush=True)
        return [], []

    walker_batch = []
    walker_speeds = []
    spawn_count = 0
    ego_location = ego_vehicle.get_location()
    attempts = max(count * 12, count)
    for _ in range(attempts):
        if spawn_count >= count:
            break
        location = world.get_random_location_from_navigation()
        if location is None:
            continue
        if location.distance(ego_location) > 100.0:
            continue
        spawn_point = carla.Transform(location)
        blueprint = random.choice(blueprints)
        if blueprint.has_attribute("is_invincible"):
            blueprint.set_attribute("is_invincible", "false")
        if blueprint.has_attribute("speed"):
            walker_speeds.append(
                blueprint.get_attribute("speed").recommended_values[1])
        else:
            walker_speeds.append("1.4")
        walker_batch.append(carla.command.SpawnActor(blueprint, spawn_point))
        spawn_count += 1

    walker_ids = []
    walker_speeds_ok = []
    for index, response in enumerate(client.apply_batch_sync(
            walker_batch, True)):
        if response.error:
            continue
        walker_ids.append(response.actor_id)
        walker_speeds_ok.append(walker_speeds[index])

    controller_bp = world.get_blueprint_library().find("controller.ai.walker")
    controller_batch = [
        carla.command.SpawnActor(
            controller_bp, carla.Transform(), walker_id)
        for walker_id in walker_ids
    ]
    controller_ids = []
    for response in client.apply_batch_sync(
            controller_batch, True):
        if not response.error:
            controller_ids.append(response.actor_id)

    actors = world.get_actors(controller_ids + walker_ids)
    controllers = actors.filter("controller.ai.walker")
    walkers = actors.filter("walker.pedestrian.*")
    world.set_pedestrians_cross_factor(0.0)
    for controller, speed in zip(controllers, walker_speeds_ok):
        controller.start()
        target = None
        for _ in range(20):
            candidate = world.get_random_location_from_navigation()
            if candidate is not None and candidate.distance(ego_location) <= 120.0:
                target = candidate
                break
        if target is None:
            target = world.get_random_location_from_navigation()
        if target is not None:
            controller.go_to_location(target)
        controller.set_max_speed(float(speed))

    if len(walkers) < count:
        print(
            "NPC walkers: spawned %d/%d; navigation spawn points were limited."
            % (len(walkers), count),
            flush=True)
    return list(walkers), list(controllers)


def stop_and_destroy_actors(client, actors, label):
    alive = [actor for actor in actors if actor_alive(actor)]
    if not alive:
        return
    print("Destroying %d %s." % (len(alive), label), flush=True)
    client.apply_batch([
        carla.command.DestroyActor(actor.id)
        for actor in alive
    ])


class CameraViewer:
    """Keep the latest four sensor frames and render them as a 2x2 mosaic."""

    def __init__(self, enabled, window_name):
        self.enabled = enabled
        self.window_name = window_name
        self.frames = {}
        self.lock = threading.Lock()
        self.cv2 = None
        self.np = None
        if not enabled:
            return
        try:
            import cv2
            import numpy as np
            self.cv2 = cv2
            self.np = np
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        except Exception as exc:
            self.enabled = False
            print("Camera view disabled: %s" % exc, flush=True)

    def on_image(self, sensor_id, image):
        if not self.enabled:
            return
        bgra = self.np.frombuffer(image.raw_data, dtype=self.np.uint8).reshape(
            (image.height, image.width, 4))
        with self.lock:
            self.frames[sensor_id] = bgra[:, :, :3].copy()

    def render(self):
        """Draw one mosaic frame; closing it leaves the ego vehicle running."""
        if not self.enabled:
            return True
        with self.lock:
            frames = dict(self.frames)
        if len(frames) < len(CAMERA_SPECS):
            return True
        tiles = []
        for sensor_id, *_unused in CAMERA_SPECS:
            tile = self.cv2.resize(frames[sensor_id], (640, 360))
            self.cv2.putText(tile, sensor_id, (15, 35),
                             self.cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                             (0, 255, 255), 2, self.cv2.LINE_AA)
            tiles.append(tile)
        mosaic = self.np.vstack((
            self.np.hstack((tiles[0], tiles[1])),
            self.np.hstack((tiles[2], tiles[3])),
        ))
        self.cv2.imshow(self.window_name, mosaic)
        key = self.cv2.waitKey(1) & 0xFF
        visible = self.cv2.getWindowProperty(
            self.window_name, self.cv2.WND_PROP_VISIBLE) >= 1
        if not visible or key in (27, ord("q")):
            self.close()
            self.enabled = False
            print("Camera view closed; ego vehicle remains active.", flush=True)

    def close(self):
        if self.enabled:
            self.cv2.destroyWindow(self.window_name)


def attach_ego_cameras(world, vehicle, role_name, sensor_tick, image_callback=None):
    """Attach the four driving cameras owned by the persistent ego spawner."""
    library = world.get_blueprint_library()
    cameras = []
    for sensor_id, x, y, z, yaw, width, height in CAMERA_SPECS:
        blueprint = library.find("sensor.camera.rgb")
        blueprint.set_attribute("image_size_x", str(width))
        blueprint.set_attribute("image_size_y", str(height))
        blueprint.set_attribute("fov", "100")
        blueprint.set_attribute("sensor_tick", str(sensor_tick))
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", "%s_%s" % (role_name, sensor_id))
        sensor = world.spawn_actor(
            blueprint,
            carla.Transform(
                carla.Location(x=x, y=y, z=z), carla.Rotation(yaw=yaw)),
            attach_to=vehicle)
        if image_callback is None:
            # Registering a no-op callback starts sensor production without
            # retaining image buffers in the spawner process.
            sensor.listen(lambda _image: None)
        else:
            sensor.listen(
                lambda image, name=sensor_id: image_callback(name, image))
        cameras.append(sensor)
    print(
        "EGO CAMERAS READY  count=%d ids=%s" % (
            len(cameras), ",".join(spec[0] for spec in CAMERA_SPECS)),
        flush=True)
    return cameras


def main():
    args = parse_args()
    if args.camera_tick <= 0.0:
        raise ValueError("--camera-tick must be positive")
    if args.minimum_shoulder_width_m <= 0.0:
        raise ValueError("--minimum-shoulder-width-m must be positive")
    camera_viewer = CameraViewer(args.camera_view, args.camera_window_name)
    client, world = wait_for_world(args.host, args.port)
    map_name = world.get_map().name.rsplit("/", 1)[-1]
    if map_name != "Town04_Opt":
        raise RuntimeError(
            "Expected Town04_Opt, but the server loaded %s" % map_name)

    existing = [
        actor for actor in world.get_actors().filter("vehicle.*")
        if actor.attributes.get("role_name") == args.role_name]
    reused = bool(existing)
    used_index = None

    npc_vehicles = []
    npc_walkers = []
    npc_controllers = []
    ego_cameras = []

    if reused:
        vehicle = min(existing, key=lambda actor: actor.id)
    else:
        blueprint = world.get_blueprint_library().find(args.vehicle)
        blueprint.set_attribute("role_name", args.role_name)
        # Do not override the default colour: this server's dirty build can
        # serialise an explicit colour as an invalid actor attribute.

        carla_map = world.get_map()
        spawn_points = carla_map.get_spawn_points()
        if not spawn_points:
            raise RuntimeError("Town04_Opt has no vehicle spawn points")

        candidates, preferred_count = ordered_spawn_candidates(
            world, carla_map, spawn_points, args.spawn_index,
            args.minimum_shoulder_width_m)
        if args.spawn_index is None:
            print(
                    "SPAWN PREFLIGHT  rail-safe shoulder candidates=%d/%d "
                "minimum_width=%.2fm" % (
                    preferred_count, len(spawn_points),
                    args.minimum_shoulder_width_m),
                flush=True)
            if preferred_count == 0:
                print(
                    "SPAWN ERROR: no rail-safe Shoulder spawn found.",
                    flush=True)
        vehicle = None
        for candidate_index, transform in candidates:
            vehicle = world.try_spawn_actor(blueprint, transform)
            if vehicle is not None:
                used_index = candidate_index
                break
        if vehicle is None:
            raise RuntimeError("Could not find an empty vehicle spawn point")

        vehicle.apply_control(carla.VehicleControl(hand_brake=True))

    wait_for_actor_alive(world, vehicle)

    if not args.no_cameras:
        ego_cameras = attach_ego_cameras(
            world, vehicle, args.role_name, args.camera_tick,
            camera_viewer.on_image if camera_viewer.enabled else None)

    location = vehicle.get_location()
    if reused:
        print(
            "EGO REUSED  id=%d role_name=%s location=(%.1f, %.1f, %.1f)" % (
                vehicle.id, args.role_name,
                location.x, location.y, location.z),
            flush=True)
    else:
        print(
            "EGO READY  id=%d role_name=%s spawn_index=%d "
            "location=(%.1f, %.1f, %.1f)" % (
                vehicle.id, args.role_name, used_index,
                location.x, location.y, location.z),
            flush=True)
    if not args.no_npc and (args.npc_vehicles > 0 or args.npc_walkers > 0):
        traffic_manager = client.get_trafficmanager(args.npc_tm_port)
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        traffic_manager.global_percentage_speed_difference(
            args.npc_speed_difference)
        npc_vehicles = spawn_npc_vehicles(
            world, traffic_manager, vehicle, args.npc_vehicles)
        npc_walkers, npc_controllers = spawn_npc_walkers(
            client, world, vehicle, args.npc_walkers)
        print(
            "NPC READY  vehicles=%d walkers=%d tm_port=%d speed_diff=%.1f%%" % (
                len(npc_vehicles), len(npc_walkers),
                args.npc_tm_port, args.npc_speed_difference),
            flush=True)
        print(
            "Keep this terminal open. Ctrl+C destroys ego and these NPC actors.",
            flush=True)
    else:
        print("Keep this terminal open. Ctrl+C destroys only this ego vehicle.", flush=True)

    stop_requested = False

    def request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    try:
        while not stop_requested and actor_alive(vehicle):
            camera_viewer.render()
            time.sleep(0.05 if camera_viewer.enabled else 0.5)
    finally:
        camera_viewer.close()
        stop_and_destroy_actors(client, ego_cameras, "ego cameras")
        for controller in npc_controllers:
            if actor_alive(controller):
                try:
                    controller.stop()
                except RuntimeError:
                    pass
        stop_and_destroy_actors(client, npc_controllers + npc_walkers, "NPC walkers")
        stop_and_destroy_actors(client, npc_vehicles, "NPC vehicles")
        if actor_alive(vehicle):
            try:
                vehicle.destroy()
                print("Ego vehicle destroyed.", flush=True)
            except RuntimeError as exc:
                print("Ego vehicle destroy skipped: %s" % exc, flush=True)


if __name__ == "__main__":
    main()
