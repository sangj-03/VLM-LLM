#!/usr/bin/env python3

"""Protocol and local HTTP tests; no CARLA server or model is required."""

import json
import os
import sys
import threading
import unittest
import urllib.error
import urllib.request
from unittest import mock

import numpy as np


sys.path.insert(0, os.path.dirname(__file__))
from dgx_llm_control_server import (  # noqa: E402
    ControlServer,
    DecisionEngine,
    extract_json_object,
    validate_driver_assessment,
)
from carla_llm_overlay import (  # noqa: E402
    RemoteControlClient,
    carla,
    copy_control_with_overlay,
)
from carla_driver_monitor import motion_state  # noqa: E402
from view_cabin_live import (  # noqa: E402
    CONTENT_BOTTOM_HEIGHT,
    CONTENT_TOP,
    center_cabin_frame,
    draw_gui_overlay,
)
from carla_safety_supervisor import (  # noqa: E402
    ShoulderLanePlanner,
    WarningDisplayManager,
    WarningDisplayPolicy,
    driver_response,
    extract_driver_state,
    parse_args,
    upcoming_stop_signal,
    set_vehicle_autopilot,
)


ONE_PIXEL_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+A8AAQUBAScY42YAAAAASUVORK5CYII=")


def telemetry(request_id=7):
    return {
        "protocol_version": 1,
        "request_id": request_id,
        "vehicle": {"speed_kph": 20.0},
        "road": {"lane_offset_m": 0.3, "heading_error_deg": 2.0},
        "nearby_vehicles": [],
        "nearby_walkers": [],
    }


def observation(observation_id=3, include_image=True):
    payload = {
        "protocol_version": 1,
        "observation_id": observation_id,
        "captured_at_unix_s": 1.0,
        "telemetry": telemetry(observation_id),
        "driving_summary": {"motion_state": "moving"},
    }
    if include_image:
        payload["cabin_image"] = {
            "mime_type": "image/png",
            "data_base64": ONE_PIXEL_PNG_BASE64,
        }
    return payload


class CountingEngine(DecisionEngine):
    def __init__(self):
        super().__init__("dry-run", "none", "", "", 1.0)
        self.control_calls = 0
        self.monitor_calls = 0

    def decide(self, payload):
        self.control_calls += 1
        return super().decide(payload)

    def assess_driver(self, payload, cabin_image):
        self.monitor_calls += 1
        return super().assess_driver(payload, cabin_image)


def post_json(url, payload, token="test-token"):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=2.0) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


class ProtocolTest(unittest.TestCase):
    def test_mrm_defaults_start_on_first_no_visible_response(self):
        with mock.patch.object(sys, "argv", ["carla_safety_supervisor.py"]):
            args = parse_args()

        self.assertEqual(args.required_no_response_observations, 1)
        # The reduced side margin makes the required full shoulder width
        # exactly 0.30 m narrower than the previous 0.40 m-per-side setting.
        self.assertEqual(ShoulderLanePlanner.SHOULDER_SIDE_MARGIN_M, 0.25)
        self.assertEqual(ShoulderLanePlanner.ROADSIDE_BARRIER_CLEARANCE_M, 0.15)
        self.assertEqual(ShoulderLanePlanner.SETTLE_TICKS, 1)

    def test_extracts_fenced_json(self):
        result = extract_json_object(
            '```json\n{"steer": -0.2, "brake": 0, "reason": "ok"}\n```')
        self.assertEqual(result["steer"], -0.2)

    def test_dry_run_corrects_right_offset_to_left(self):
        engine = DecisionEngine("dry-run", "none", "", "", 1.0)
        command = engine.decide(telemetry())
        self.assertLess(command["steer"], 0.0)
        self.assertEqual(command["brake"], 0.0)

    def test_motion_state_uses_noise_tolerant_thresholds(self):
        self.assertEqual(motion_state(0.0), "stationary")
        self.assertEqual(motion_state(0.1), "stationary")
        self.assertEqual(motion_state(0.5), "creeping")
        self.assertEqual(motion_state(10.0), "moving")

    def test_driver_assessment_validation(self):
        result = validate_driver_assessment({
            "driver_present": True,
            "looking_forward": False,
            "eyes_closed": False,
            "phone_use": True,
            "unsafe_behavior": True,
            "risk_level": "caution",
            "confidence": 0.9,
            "reason": "phone use",
        })
        self.assertEqual(result["risk_level"], "caution")
        self.assertTrue(result["unsafe_behavior"])
        self.assertEqual(result["driver_response"], "reduced")
        self.assertTrue(result["assessment_valid"])

    def test_safety_supervisor_uses_driver_response_not_risk(self):
        # A legacy high risk must not override an explicit active response.
        self.assertEqual(driver_response({
            "driver_response": "active",
            "risk_level": "critical",
            "confidence": 0.9,
        }, 0.7), "active")

        # Conversely, no_visible_response remains actionable even if a legacy
        # risk field says normal. The main loop debounces it across observations.
        self.assertEqual(driver_response({
            "driver_response": "no_visible_response",
            "risk_level": "normal",
            "confidence": 0.9,
        }, 0.7), "no_visible_response")
        self.assertEqual(driver_response({
            "driver_response": "reduced",
            "confidence": 0.85,
        }, 0.7), "reduced")

    def test_safety_supervisor_rejects_unusable_response(self):
        self.assertIsNone(driver_response({
            "driver_response": "no_visible_response",
            "confidence": 0.5,
        }, 0.7))
        self.assertIsNone(driver_response({
            "driver_response": "no_visible_response",
            "assessment_valid": False,
        }, 0.7))
        self.assertIsNone(driver_response({
            "driver_response": "unexpected_label",
        }, 0.7))
        # The removed legacy label is not accepted as a fourth state.
        self.assertIsNone(driver_response({
            "driver_response": "uncertain",
        }, 0.7))
        # The response-only model may omit confidence; its explicit label is
        # still accepted instead of silently disabling the safety path.
        self.assertEqual(driver_response({
            "driver_response": "no_visible_response",
        }, 0.7), "no_visible_response")
        self.assertEqual(driver_response({
            "driver_response": "no_visible",
        }, 0.7), "no_visible_response")

    def test_extract_driver_state_accepts_bridge_envelope_or_direct_state(self):
        nested = {"driver_response": "active"}
        self.assertIs(extract_driver_state({"driver_state": nested}), nested)
        direct = {"driver_response": "reduced"}
        self.assertIs(extract_driver_state(direct), direct)

    def test_driver_response_overlay_renders_three_states(self):
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        for response in ("active", "reduced", "no_visible_response"):
            with self.subTest(driver_response=response):
                rendered = draw_gui_overlay(image, {
                    "observation_id": 23,
                    "inference_ms": 727.03,
                    "result_age_s": 0.2,
                    "driver_state": {
                        "driver_response": response,
                        "eye_state": "open",
                        "head_pose": "upright",
                        "upper_body_posture": "upright",
                        "gaze": "forward",
                    },
                })
                self.assertEqual(rendered.shape, image.shape)
                self.assertFalse(np.array_equal(rendered, image))

        # Missing/invalid output renders a neutral waiting panel, not another
        # driver-response condition.
        rendered = draw_gui_overlay(image, {
            "observation_id": 24,
            "driver_state": {},
        })
        self.assertEqual(rendered.shape, image.shape)

    def test_cabin_frame_is_aspect_fitted_and_centered(self):
        source = np.empty((100, 200, 3), dtype=np.uint8)
        source[:] = (80, 120, 160)
        canvas = center_cabin_frame(source, 400, 400)

        self.assertEqual(canvas.shape, (400, 400, 3))
        background = np.array((18, 22, 28), dtype=np.uint8)
        image_mask = np.any(canvas != background, axis=2)
        ys, xs = np.where(image_mask)
        self.assertAlmostEqual((xs.min() + xs.max()) / 2.0, 199.5, delta=1.0)
        content_center = (
            CONTENT_TOP + (400 - CONTENT_BOTTOM_HEIGHT)) / 2.0
        self.assertAlmostEqual(
            (ys.min() + ys.max()) / 2.0, content_center, delta=1.0)
        displayed_ratio = (xs.max() - xs.min() + 1) / float(
            ys.max() - ys.min() + 1)
        self.assertAlmostEqual(displayed_ratio, 2.0, delta=0.02)

    def test_warning_display_manager_starts_once_and_closes_on_active(self):
        process = mock.Mock()
        process.pid = 4321
        process.poll.return_value = None
        process.wait.return_value = 0
        secret = "bridge-secret-not-for-command-line"
        with mock.patch(
                "carla_safety_supervisor.subprocess.Popen",
                return_value=process) as popen:
            manager = WarningDisplayManager(
                __file__, "http://127.0.0.1:8000/driver_state/latest",
                secret, 5.0, "Driver Response Warning")
            self.assertTrue(manager.ensure_running())
            self.assertTrue(manager.ensure_running())
            popen.assert_called_once()

            command = popen.call_args.args[0]
            self.assertNotIn(secret, command)
            self.assertEqual(
                popen.call_args.kwargs["env"]["LLM_BRIDGE_TOKEN"], secret)
            self.assertIn("running pid=4321", manager.status())

            manager.stop()
            process.terminate.assert_called_once()
            process.wait.assert_called_once_with(timeout=2.0)

    def test_warning_display_policy_uses_two_reduced_and_three_active(self):
        policy = WarningDisplayPolicy(
            required_reduced_observations=2,
            required_active_observations=3)

        self.assertEqual(policy.observe("reduced"), (False, False))
        self.assertEqual(policy.reduced_observations, 1)
        self.assertEqual(policy.observe("reduced"), (True, False))
        self.assertEqual(policy.reduced_observations, 2)

        self.assertEqual(policy.observe("active"), (False, False))
        self.assertEqual(policy.observe("active"), (False, False))
        self.assertEqual(policy.observe("active"), (False, True))
        self.assertEqual(policy.active_observations, 3)

    def test_warning_display_policy_defaults_to_four_active_observations(self):
        policy = WarningDisplayPolicy()

        for _ in range(3):
            self.assertEqual(policy.observe("active"), (False, False))
        self.assertEqual(policy.observe("active"), (False, True))

    def test_warning_display_policy_requires_three_no_visible_observations(self):
        policy = WarningDisplayPolicy(2, 3, 3)
        self.assertEqual(policy.observe("reduced"), (False, False))
        self.assertEqual(policy.observe(None), (False, False))
        self.assertEqual(policy.reduced_observations, 0)
        self.assertEqual(policy.observe("reduced"), (False, False))

        self.assertEqual(
            policy.observe("no_visible_response"), (False, False))
        self.assertEqual(policy.no_visible_observations, 1)
        self.assertEqual(
            policy.observe("no_visible_response"), (False, False))
        self.assertEqual(policy.no_visible_observations, 2)
        self.assertEqual(
            policy.observe("no_visible_response"), (True, False))
        self.assertEqual(policy.reduced_observations, 0)
        self.assertEqual(policy.active_observations, 0)
        self.assertEqual(policy.no_visible_observations, 3)

    def test_upcoming_stop_signal_returns_current_direction_stop_line(self):
        world = mock.Mock()
        carla_map = mock.Mock()
        world.get_map.return_value = carla_map
        light = mock.Mock()
        light.id = 41
        light.get_state.return_value = carla.TrafficLightState.Red
        world.get_actors.return_value.filter.return_value = [light]

        vehicle = mock.Mock()
        vehicle.get_transform.return_value = carla.Transform(
            carla.Location(), carla.Rotation(yaw=0.0))
        vehicle.get_velocity.return_value = carla.Vector3D()
        vehicle.get_traffic_light.return_value = light
        current = mock.Mock()
        current.road_id = 7
        current.lane_width = 3.5
        current.transform = carla.Transform(
            carla.Location(), carla.Rotation(yaw=0.0))
        trigger = mock.Mock()
        trigger.road_id = 7
        trigger.transform = carla.Transform(
            carla.Location(x=10.0), carla.Rotation(yaw=0.0))
        carla_map.get_waypoint.return_value = trigger

        with mock.patch(
                "carla_safety_supervisor.get_trafficlight_trigger_location",
                return_value=carla.Location(x=10.0)):
            result = upcoming_stop_signal(world, vehicle, current)

        self.assertIsNotNone(result)
        state, location, distance = result
        self.assertEqual(state, carla.TrafficLightState.Red)
        self.assertAlmostEqual(location.x, 10.0)
        self.assertAlmostEqual(distance, 10.0)

    def test_warning_display_manager_can_be_disabled(self):
        with mock.patch(
                "carla_safety_supervisor.subprocess.Popen") as popen:
            manager = WarningDisplayManager(
                __file__, "http://127.0.0.1:8000/driver_state/latest",
                "", 5.0, "Driver Response Warning", enabled=False)
            self.assertFalse(manager.ensure_running())
            self.assertEqual(manager.status(), "disabled")
            popen.assert_not_called()

    def test_safety_supervisor_autopilot_toggle(self):
        class MockVehicle:
            def __init__(self):
                self.autopilot_enabled = None
                self.tm_port = None
            def set_autopilot(self, enabled, tm_port=8000):
                self.autopilot_enabled = enabled
                self.tm_port = tm_port

        veh = MockVehicle()
        self.assertTrue(set_vehicle_autopilot(veh, False, 8000))
        self.assertFalse(veh.autopilot_enabled)
        self.assertEqual(veh.tm_port, 8000)

        self.assertTrue(set_vehicle_autopilot(veh, True, 8000))
        self.assertTrue(veh.autopilot_enabled)

    def test_shoulder_planner_selects_outward_lane_in_both_directions(self):
        class MockWaypoint:
            def __init__(self, lane_id):
                self.road_id = 7
                self.lane_id = lane_id
                self.lane_type = carla.LaneType.Driving
                self.transform = carla.Transform()
                self.right = None
                self.left = None

            def get_right_lane(self):
                return self.right

            def get_left_lane(self):
                return self.left

        planner = ShoulderLanePlanner.__new__(ShoulderLanePlanner)
        for source_id, outward_id, inward_id in ((-2, -3, -1), (2, 3, 1)):
            with self.subTest(source_lane=source_id):
                source = MockWaypoint(source_id)
                source.right = MockWaypoint(outward_id)
                source.left = MockWaypoint(inward_id)
                side, target = planner._outward_lane(source)
                self.assertEqual(side, "right")
                self.assertEqual(target.lane_id, outward_id)

    def test_outermost_lane_is_guarded_when_shoulder_is_unusable(self):
        class MockWaypoint:
            def __init__(self, lane_id):
                self.road_id = 7
                self.lane_id = lane_id
                self.lane_type = carla.LaneType.Driving
                self.transform = carla.Transform()
                self.right = None
                self.left = None

            def get_right_lane(self):
                return self.right

            def get_left_lane(self):
                return self.left

        planner = ShoulderLanePlanner.__new__(ShoulderLanePlanner)
        planner._shoulder_is_wide_enough = mock.Mock(return_value=False)
        source = MockWaypoint(-2)
        source.left = MockWaypoint(-1)
        self.assertEqual(planner._roadside_edge(source), "right")

    def test_shoulder_planner_selects_outward_shoulder(self):
        class MockWaypoint:
            def __init__(self, lane_id, lane_type, lane_width=3.5):
                self.road_id = 7
                self.lane_id = lane_id
                self.lane_type = lane_type
                self.lane_width = lane_width
                self.is_junction = False
                self.transform = carla.Transform()
                self.right = None
                self.left = None

            def get_right_lane(self):
                return self.right

            def get_left_lane(self):
                return self.left

        planner = ShoulderLanePlanner.__new__(ShoulderLanePlanner)
        planner.minimum_shoulder_width_m = 2.7
        planner._shoulder_is_wide_enough = mock.Mock(
            side_effect=lambda waypoint: waypoint.lane_width >= 2.7)
        driving = MockWaypoint(-3, carla.LaneType.Driving)
        driving.right = MockWaypoint(-4, carla.LaneType.Shoulder)
        driving.left = MockWaypoint(-2, carla.LaneType.Driving)
        side, target = planner._outward_shoulder(driving)
        self.assertEqual(side, "right")
        self.assertEqual(target.lane_type, carla.LaneType.Shoulder)

        driving.right = MockWaypoint(
            -4, carla.LaneType.Shoulder, lane_width=0.635)
        side, target = planner._outward_shoulder(driving)
        self.assertIsNone(side)
        self.assertIsNone(target)

    def test_safe_zone_rejects_sidewalk_and_accepts_road_parking(self):
        class MockWaypoint:
            def __init__(self, lane_type):
                self.road_id = 7
                self.section_id = 0
                self.s = 12.0
                self.lane_id = (-4 if lane_type != carla.LaneType.Driving
                                else -3)
                self.lane_type = lane_type
                self.lane_width = 3.5
                self.is_junction = False
                self.transform = carla.Transform()
                self.left = None
                self.right = None

            def get_left_lane(self):
                return self.left

            def get_right_lane(self):
                return self.right

        planner = ShoulderLanePlanner.__new__(ShoulderLanePlanner)
        planner.minimum_shoulder_width_m = 2.7
        planner.vehicle = mock.Mock()
        planner.vehicle.bounding_box.extent.y = 1.0
        planner.world = mock.Mock()
        planner._safe_surface_cache = {}
        parking = MockWaypoint(carla.LaneType.Parking)
        parking.left = MockWaypoint(carla.LaneType.Driving)

        def surface_hit(label):
            hit = mock.Mock()
            hit.label = label
            hit.location = carla.Location(z=0.0)
            return [hit]

        planner.world.cast_ray.return_value = surface_hit(
            carla.CityObjectLabel.Sidewalks)
        self.assertFalse(planner._shoulder_is_wide_enough(parking))

        planner._safe_surface_cache.clear()
        planner.world.cast_ray.return_value = surface_hit(
            carla.CityObjectLabel.Roads)
        self.assertTrue(planner._shoulder_is_wide_enough(parking))

    def test_stop_target_prefers_nearest_safe_candidate(self):
        planner = ShoulderLanePlanner.__new__(ShoulderLanePlanner)
        source = mock.Mock()
        source.road_id = 7
        source.transform = carla.Transform()
        near = mock.Mock()
        near.road_id = 7
        near.lane_type = carla.LaneType.Shoulder
        near.transform = carla.Transform()
        far = mock.Mock()
        far.road_id = 7
        far.lane_type = carla.LaneType.Shoulder
        far.transform = carla.Transform()
        source.next.side_effect = lambda distance: (
            [near] if distance == planner.STOP_TARGET_MIN_AHEAD_M else
            [far] if distance == planner.STOP_TARGET_MIN_AHEAD_M + 2.0 else [])
        planner._shoulder_stop_target_score = mock.Mock(
            side_effect=lambda candidate: 1.0 if candidate is near else 999.0)

        self.assertIs(planner._find_shoulder_stop_target(source), near)

    def test_shoulder_planner_targets_available_corridor_centre(self):
        class MockWaypoint:
            def __init__(self, lane_type):
                self.road_id = 7
                self.lane_id = -5 if lane_type == carla.LaneType.Shoulder else -4
                self.lane_type = lane_type
                self.lane_width = 3.5
                self.transform = carla.Transform()
                self.left = None
                self.right = None

            def get_left_lane(self):
                return self.left

            def get_right_lane(self):
                return self.right

        shoulder = MockWaypoint(carla.LaneType.Shoulder)
        shoulder.left = MockWaypoint(carla.LaneType.Driving)
        planner = ShoulderLanePlanner.__new__(ShoulderLanePlanner)
        planner.vehicle = mock.Mock()
        planner.vehicle.bounding_box.extent.y = 1.0
        planner.world = mock.Mock()
        planner.world.cast_ray.return_value = []
        self.assertEqual(planner._shoulder_target_offset(shoulder), 0.0)

        def guardrail_hit(start, _end):
            hit = mock.Mock()
            hit.label = carla.CityObjectLabel.GuardRail
            hit.location = carla.Location(
                x=start.x, y=start.y + 1.4, z=start.z)
            return [hit]

        planner.world.cast_ray.side_effect = guardrail_hit
        # The obstacle leaves a safe centre range of [-0.65, 0.25]m on the
        # outward axis, so its midpoint is 0.20m toward the driving lane.
        self.assertAlmostEqual(
            planner._shoulder_target_offset(shoulder), -0.20, places=2)
        self.assertAlmostEqual(
            planner._roadside_barrier_clearance(shoulder), 0.6,
            places=2)
        self.assertTrue(planner._shoulder_track_error_safe(-0.1))
        self.assertTrue(planner._shoulder_track_error_safe(0.1))
        self.assertFalse(planner._shoulder_track_error_safe(0.2))

    def test_shoulder_planner_crawls_through_junction_to_keep_searching(self):
        planner = ShoulderLanePlanner.__new__(ShoulderLanePlanner)
        planner.world = mock.Mock()
        planner.map = mock.Mock()
        planner.vehicle = mock.Mock()
        planner.caution_speed_kph = 12.0
        planner.target_speed_kph = 18.0
        waypoint = mock.Mock()
        planner.map.get_waypoint.return_value = waypoint
        expected = (0.1, 0.0, 0.1, None, "seeking shoulder")

        with mock.patch(
                "carla_safety_supervisor.forward_hazard_reason",
                return_value="junction ahead"), mock.patch.object(
                    planner, "_continue_current_lane",
                    return_value=expected) as continue_lane:
            self.assertEqual(planner.step(), expected)

        continue_lane.assert_called_once()
        self.assertIn("junction ahead", continue_lane.call_args.args[1])
        self.assertEqual(
            continue_lane.call_args.kwargs["target_speed_kph"], 18.0)

    def test_shoulder_planner_stops_for_real_forward_obstacle(self):
        planner = ShoulderLanePlanner.__new__(ShoulderLanePlanner)
        planner.world = mock.Mock()
        planner.map = mock.Mock()
        planner.vehicle = mock.Mock()
        waypoint = mock.Mock()
        planner.map.get_waypoint.return_value = waypoint
        expected = (0.0, 0.3, 0.0, None, "holding current lane")

        with mock.patch(
                "carla_safety_supervisor.forward_hazard_reason",
                return_value="forward path occupied"), mock.patch.object(
                    planner, "_hold_current_lane",
                    return_value=expected) as hold_lane:
            self.assertEqual(planner.step(), expected)

        hold_lane.assert_called_once_with(
            waypoint, "forward path occupied")

    def test_shoulder_stop_requires_aligned_stable_pose(self):
        planner = ShoulderLanePlanner.__new__(ShoulderLanePlanner)
        planner.stop_target = mock.Mock()
        planner.stop_target.lane_id = -4
        planner.stop_target.transform.location = carla.Location(x=1.0)
        planner.phase = "stop_target"
        planner.stop_stable_ticks = planner.STOP_SETTLE_TICKS - 1
        planner.stop_ready = False
        planner.caution_speed_kph = 12.0
        planner.vehicle = mock.Mock()
        planner.vehicle.get_location.return_value = carla.Location()
        planner.vehicle.get_velocity.return_value.length.return_value = 0.0
        waypoint = mock.Mock()
        waypoint.lane_type = carla.LaneType.Shoulder
        waypoint.transform.get_forward_vector.return_value = carla.Vector3D(x=1.0)
        planner._shoulder_stop_target_clear = mock.Mock(return_value=True)
        planner._lane_center_control = mock.Mock(
            return_value=carla.VehicleControl())
        planner._limit_control = mock.Mock(side_effect=lambda control: control)
        planner._lane_alignment = mock.Mock(return_value=(0.0, 0.0))
        planner._shoulder_target_offset = mock.Mock(return_value=0.0)
        planner._shoulder_track_error_safe = mock.Mock(return_value=True)

        planner._approach_stop_target(waypoint)

        self.assertTrue(planner.stop_ready)
        self.assertEqual(
            planner.stop_stable_ticks, planner.STOP_SETTLE_TICKS)

    def test_recovery_completes_only_after_driving_lane_alignment(self):
        planner = ShoulderLanePlanner.__new__(ShoulderLanePlanner)
        planner.map = mock.Mock()
        planner.vehicle = mock.Mock()
        planner.vehicle.get_velocity.return_value.length.return_value = 0.0
        planner.stable_ticks = planner.SETTLE_TICKS - 1
        waypoint = mock.Mock()
        waypoint.lane_type = carla.LaneType.Driving
        planner.map.get_waypoint.return_value = waypoint
        planner._lane_alignment = mock.Mock(return_value=(0.05, 1.0))
        planner._lane_center_control = mock.Mock(
            return_value=carla.VehicleControl())

        result = planner.recovery_step()

        self.assertTrue(result[-1])
        self.assertIn("centering driving lane", result[-2])

    def test_shoulder_planner_crawls_for_distant_traffic_and_signal(self):
        for reason in ("forward traffic ahead", "red traffic light ahead"):
            with self.subTest(reason=reason):
                planner = ShoulderLanePlanner.__new__(ShoulderLanePlanner)
                planner.world = mock.Mock()
                planner.map = mock.Mock()
                planner.vehicle = mock.Mock()
                planner.caution_speed_kph = 12.0
                planner.target_speed_kph = 18.0
                waypoint = mock.Mock()
                planner.map.get_waypoint.return_value = waypoint
                expected = (0.0, 0.0, 0.1, None, "seeking shoulder")
                with mock.patch(
                        "carla_safety_supervisor.forward_hazard_reason",
                        return_value=reason), mock.patch.object(
                            planner, "_continue_current_lane",
                            return_value=expected) as continue_lane:
                    self.assertEqual(planner.step(), expected)
                continue_lane.assert_called_once()
                self.assertEqual(
                    continue_lane.call_args.kwargs["target_speed_kph"], 18.0)

    def test_shoulder_planner_does_not_finish_on_outermost_driving_lane(self):
        planner = ShoulderLanePlanner.__new__(ShoulderLanePlanner)
        planner.world = mock.Mock()
        planner.map = mock.Mock()
        planner.vehicle = mock.Mock()
        planner.planner = mock.Mock()
        planner.planner.done.return_value = True
        planner.target_lane_id = None
        planner.target_speed_kph = 18.0
        waypoint = mock.Mock()
        waypoint.lane_type = carla.LaneType.Driving
        planner.map.get_waypoint.return_value = waypoint
        expected = (0.0, 0.0, 0.1, None, "seeking shoulder")

        with mock.patch(
                "carla_safety_supervisor.forward_hazard_reason",
                return_value=None), mock.patch.object(
                    planner, "_outward_lane", return_value=(None, None)), \
                mock.patch.object(
                    planner, "_outward_shoulder",
                    return_value=(None, None)), mock.patch.object(
                    planner, "_continue_current_lane",
                    return_value=expected) as continue_lane:
            self.assertEqual(planner.step(), expected)

        continue_lane.assert_called_once_with(
            waypoint, "no adjacent shoulder yet; searching ahead",
            target_speed_kph=18.0, center_guard=True)


    def test_client_validates_response_and_preserves_autopilot_brake(self):
        client = RemoteControlClient(
            "http://127.0.0.1:1/control", 1.0, "", 0.65)
        command = client._parse_response({
            "protocol_version": 1,
            "request_id": 9,
            "command": {"steer": 0.25, "brake": 0.1, "reason": "test"},
            "valid_for_ms": 1000,
        }, 9)
        self.assertEqual(command.steer, 0.25)

        base = carla.VehicleControl()
        base.throttle = 0.6
        base.steer = -0.1
        base.brake = 0.7
        output = copy_control_with_overlay(base, command.steer, command.brake, True)
        self.assertAlmostEqual(output.steer, 0.25, places=5)
        self.assertAlmostEqual(output.brake, 0.7, places=5)
        self.assertAlmostEqual(output.throttle, 0.0, places=5)

    def test_http_image_gate_and_auth(self):
        engine = CountingEngine()
        server = ControlServer(("127.0.0.1", 0), engine, "test-token", 1200)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base_url = "http://127.0.0.1:%d" % server.server_port
            body = json.dumps(telemetry(42)).encode("utf-8")
            unauthorized = urllib.request.Request(
                base_url + "/telemetry", data=body,
                headers={"Content-Type": "application/json"})
            with self.assertRaises(urllib.error.HTTPError) as context:
                urllib.request.urlopen(unauthorized, timeout=2.0)
            self.assertEqual(context.exception.code, 401)

            status, result = post_json(
                base_url + "/telemetry", telemetry(42))
            self.assertEqual(status, 202)
            self.assertFalse(result["inference_started"])
            self.assertEqual(engine.control_calls, 0)
            self.assertEqual(engine.monitor_calls, 0)

            with self.assertRaises(urllib.error.HTTPError) as context:
                post_json(
                    base_url + "/monitor",
                    observation(4, include_image=False))
            self.assertEqual(context.exception.code, 422)
            self.assertEqual(engine.monitor_calls, 0)

            status, result = post_json(
                base_url + "/monitor", observation(5))
            self.assertEqual(status, 200)
            self.assertTrue(result["inference_started"])
            self.assertEqual(result["observation_id"], 5)
            self.assertEqual(result["driver_state"]["risk_level"], "normal")
            self.assertEqual(engine.monitor_calls, 1)

            status, result = post_json(base_url + "/monitor/complete", {
                "protocol_version": 1,
                "last_observation_id": 5,
                "completed_at_unix_s": 123.0,
            })
            self.assertEqual(status, 202)
            self.assertTrue(result["monitoring_complete"])
            self.assertFalse(result["inference_started"])
            latest_request = urllib.request.Request(
                base_url + "/driver_state/latest",
                headers={"Authorization": "Bearer test-token"})
            with urllib.request.urlopen(
                    latest_request, timeout=2.0) as response:
                latest = json.loads(response.read().decode("utf-8"))
            self.assertTrue(latest["monitoring_complete"])
            self.assertEqual(latest["completed_observation_id"], 5)

            # A later image starts a new session and clears the old marker.
            status, _ = post_json(
                base_url + "/monitor", observation(6))
            self.assertEqual(status, 200)
            with urllib.request.urlopen(
                    latest_request, timeout=2.0) as response:
                latest = json.loads(response.read().decode("utf-8"))
            self.assertFalse(latest["monitoring_complete"])

            with self.assertRaises(urllib.error.HTTPError) as context:
                post_json(base_url + "/control", telemetry(43))
            self.assertEqual(context.exception.code, 410)
            self.assertEqual(engine.control_calls, 0)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2.0)


if __name__ == "__main__":
    unittest.main()
