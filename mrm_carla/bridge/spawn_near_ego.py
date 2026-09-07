#!/usr/bin/env python3
"""Spawn CARLA Traffic Manager vehicles near the existing ego vehicle.

Only standard CARLA map spawn points are used, so NPCs are not deliberately
created on top of the ego or in arbitrary road geometry.  The process owns
only the NPC actor ids it creates; Ctrl+C removes those NPCs and never the ego.
"""

import argparse
import glob
import os
import random
import signal
import sys
import time


def add_carla_egg() -> None:
    """Make the locally installed CARLA Python API importable when needed."""
    patterns = [
        "/home/tmo/CARLA_0.9.10.1/PythonAPI/carla/dist/"
        "carla-*%d.%d-linux-x86_64.egg" % (
            sys.version_info.major, sys.version_info.minor),
        "../carla/dist/carla-*%d.%d-linux-x86_64.egg" % (
            sys.version_info.major, sys.version_info.minor),
    ]
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            sys.path.append(matches[0])
            return


add_carla_egg()

import carla  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--role-name", default="town06_ego")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument(
        "--radius", type=float, default=150.0,
        help="maximum Euclidean distance from the ego in metres (default: 150)",
    )
    parser.add_argument(
        "--minimum-distance", type=float, default=12.0,
        help="do not use spawn points closer than this to the ego (default: 12)",
    )
    parser.add_argument("--tm-port", type=int, default=8002)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--speed-difference", type=float, default=0.0,
        help="Traffic Manager speed difference percent; positive is slower",
    )
    parser.add_argument("--car-lights-on", action="store_true")
    return parser.parse_args()


def find_ego(world: carla.World, role_name: str) -> carla.Vehicle:
    vehicles = [
        actor for actor in world.get_actors().filter("vehicle.*")
        if actor.attributes.get("role_name") == role_name
    ]
    if not vehicles:
        raise RuntimeError("No ego vehicle found with role_name=%r" % role_name)
    return min(vehicles, key=lambda actor: actor.id)


def attribute_value(attribute) -> str:
    """Support both older and newer CARLA ActorAttribute bindings."""
    if hasattr(attribute, "as_string"):
        return attribute.as_string()
    return str(attribute)


def safe_vehicle_blueprints(world: carla.World):
    blueprints = list(world.get_blueprint_library().filter("vehicle.*"))
    cars = [
        blueprint for blueprint in blueprints
        if blueprint.has_attribute("base_type")
        and attribute_value(blueprint.get_attribute("base_type")) == "car"
    ]
    return cars or blueprints


def main() -> None:
    args = parse_args()
    if args.count < 1:
        raise ValueError("--count must be positive")
    if args.radius <= 0.0 or args.minimum_distance < 0.0:
        raise ValueError("--radius must be positive and --minimum-distance non-negative")
    if args.minimum_distance >= args.radius:
        raise ValueError("--minimum-distance must be smaller than --radius")

    random.seed(args.seed)
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    ego = find_ego(world, args.role_name)
    ego_location = ego.get_location()

    # Sort by distance for locality, then shuffle only equal-distance groups
    # indirectly through blueprint selection.  Existing CARLA spawn points make
    # failed/overlapping spawns safe no-ops rather than forced collisions.
    candidates = [
        transform for transform in world.get_map().get_spawn_points()
        if args.minimum_distance <= transform.location.distance(ego_location)
        <= args.radius
    ]
    candidates.sort(key=lambda transform: transform.location.distance(ego_location))
    if not candidates:
        raise RuntimeError(
            "No map spawn points within %.1f-%.1f m of ego. Increase --radius."
            % (args.minimum_distance, args.radius))

    try:
        traffic_manager = client.get_trafficmanager(args.tm_port)
    except RuntimeError as exc:
        raise RuntimeError(
            "Could not create/use Traffic Manager port %d: %s. "
            "Try --tm-port 8003." % (args.tm_port, exc)) from exc
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)
    traffic_manager.global_percentage_speed_difference(args.speed_difference)

    blueprints = safe_vehicle_blueprints(world)
    if not blueprints:
        raise RuntimeError("No usable vehicle blueprints available")

    spawned_ids = []
    for transform in candidates:
        if len(spawned_ids) >= args.count:
            break
        blueprint = random.choice(blueprints)
        if blueprint.has_attribute("driver_id"):
            choices = blueprint.get_attribute("driver_id").recommended_values
            if choices:
                blueprint.set_attribute("driver_id", random.choice(choices))
        blueprint.set_attribute("role_name", "near_ego_npc")
        actor = world.try_spawn_actor(blueprint, transform)
        if actor is None:
            continue
        actor.set_autopilot(True, traffic_manager.get_port())
        if args.car_lights_on:
            actor.set_light_state(carla.VehicleLightState(
                carla.VehicleLightState.Position
                | carla.VehicleLightState.LowBeam))
        spawned_ids.append(actor.id)

    print(
        "NEAR-EGO NPC READY  spawned=%d/%d candidates=%d radius=%.1fm "
        "tm_port=%d ids=%s" % (
            len(spawned_ids), args.count, len(candidates), args.radius,
            args.tm_port, spawned_ids),
        flush=True)
    if len(spawned_ids) < args.count:
        print(
            "Only %d safe/empty nearby spawn points were available. "
            "Increase --radius to add more." % len(spawned_ids), flush=True)
    print("Keep this terminal open. Ctrl+C removes only these NPC vehicles.", flush=True)

    stop_requested = False

    def request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        while not stop_requested:
            world.wait_for_tick(1.0)
    finally:
        if spawned_ids:
            client.apply_batch([carla.command.DestroyActor(actor_id)
                                for actor_id in spawned_ids])
        print("Destroyed %d near-ego NPC vehicles." % len(spawned_ids), flush=True)


if __name__ == "__main__":
    main()
