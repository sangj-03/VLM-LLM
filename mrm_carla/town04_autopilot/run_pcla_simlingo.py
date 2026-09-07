#!/usr/bin/env python3
"""Minimal external launcher for the unmodified official PCLA SimLingo agent."""

import argparse
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

import carla

PCLA_ROOT = "/home/tmo/carla_tools/PCLA"
if PCLA_ROOT not in sys.path:
    sys.path.insert(0, PCLA_ROOT)

from PCLA import PCLA
from pcla_functions import location_to_waypoint, route_maker


def find_vehicle(world, role_name):
    for actor in world.get_actors().filter("vehicle.*"):
        if actor.attributes.get("role_name") == role_name:
            return actor
    raise RuntimeError("No ego vehicle with role_name=%r" % role_name)


def forward_destination(world, vehicle, distance):
    waypoint = world.get_map().get_waypoint(vehicle.get_location(), project_to_road=True)
    if waypoint is None:
        raise RuntimeError("Ego vehicle is not on a driving lane")
    travelled = 0.0
    while travelled < distance:
        candidates = waypoint.next(min(2.0, distance - travelled))
        if not candidates:
            break
        yaw = waypoint.transform.rotation.yaw
        waypoint = min(candidates, key=lambda item: abs((item.transform.rotation.yaw - yaw + 180.0) % 360.0 - 180.0))
        travelled += 2.0
    if travelled < 4.0:
        raise RuntimeError("Could not create a forward CARLA route")
    return waypoint.transform.location


def update_spectator(world, vehicle):
    transform = vehicle.get_transform()
    location = carla.Location(x=-7.0, y=0.0, z=3.5)
    transform.transform(location)
    world.get_spectator().set_transform(carla.Transform(
        location,
        carla.Rotation(pitch=-16.0, yaw=transform.rotation.yaw, roll=0.0),
    ))


def copy_control(control):
    if control is None:
        return carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)
    return carla.VehicleControl(
        throttle=control.throttle,
        steer=control.steer,
        brake=control.brake,
        hand_brake=control.hand_brake,
        reverse=control.reverse,
        manual_gear_shift=control.manual_gear_shift,
        gear=control.gear,
    )


def start_inference_worker(pcla):
    """Keep costly VLM inference off CARLA's fixed-rate simulation loop."""
    stop_event = threading.Event()
    state = {"control": carla.VehicleControl(brake=1.0), "error": None}
    lock = threading.Lock()

    def run():
        while not stop_event.is_set():
            try:
                action = pcla.get_action()
                if action is not None:
                    with lock:
                        state["control"] = copy_control(action)
            except Exception as exc:
                with lock:
                    state["error"] = exc
                stop_event.set()

    worker = threading.Thread(target=run, name="pcla-simlingo-inference", daemon=True)
    worker.start()
    return stop_event, worker, state, lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--role-name", default="town06_ego")
    parser.add_argument("--route-distance", type=float, default=80.0)
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)
    world = client.get_world()
    map_name = world.get_map().name.rsplit("/", 1)[-1]
    if map_name != "Town04_Opt":
        raise RuntimeError(
            "Town04_Opt is required, but current map is %s" % map_name)
    vehicle = find_vehicle(world, args.role_name)

    destination = forward_destination(world, vehicle, args.route_distance)
    route_waypoints = location_to_waypoint(client, vehicle.get_location(), destination, distance=2)
    route_path = Path(tempfile.gettempdir()) / ("pcla_simlingo_%d.xml" % vehicle.id)
    route_maker(route_waypoints, str(route_path))

    original = world.get_settings()
    original_sync = original.synchronous_mode
    original_delta = original.fixed_delta_seconds
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    pcla = None
    stop_event = None
    worker = None
    try:
        pcla = PCLA("simlingo_simlingo", vehicle, str(route_path), client)
        stop_event, worker, inference_state, inference_lock = start_inference_worker(pcla)
        print("PCLA SimLingo ready (20 Hz spectator/tick, async inference). Ctrl+C stops the external launcher.", flush=True)
        next_tick = time.monotonic()
        while True:
            with inference_lock:
                worker_error = inference_state["error"]
                action = copy_control(inference_state["control"])
            if worker_error is not None:
                raise RuntimeError("PCLA SimLingo inference worker failed") from worker_error
            update_spectator(world, vehicle)
            vehicle.apply_control(action)
            world.tick()
            next_tick += 0.05
            time.sleep(max(0.0, next_tick - time.monotonic()))
    finally:
        if stop_event is not None:
            stop_event.set()
        if worker is not None:
            worker.join(timeout=5.0)
        # The official PCLA cleanup owns and destroys the ego actor as well.
        if pcla is not None:
            pcla.cleanup()
        route_path.unlink(missing_ok=True)
        restore = world.get_settings()
        restore.synchronous_mode = original_sync
        restore.fixed_delta_seconds = original_delta
        world.apply_settings(restore)


if __name__ == "__main__":
    main()
