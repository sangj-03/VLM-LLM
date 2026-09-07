#!/usr/bin/env python3
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(__file__))
import run_behavior_autopilot as drive


class IntersectionControlTest(unittest.TestCase):
    def test_stop_at_line_unwinds_steering(self):
        vehicle = mock.Mock()
        vehicle.bounding_box.extent.x = 2.4
        vehicle.get_velocity.return_value = drive.carla.Vector3D(0.2, 0, 0)
        control = drive.carla.VehicleControl(
            throttle=0.7, steer=0.55, brake=0.0)
        result = drive.stop_at_line(control, vehicle, 0.3)
        self.assertEqual(result.throttle, 0.0)
        self.assertEqual(result.steer, 0.0)
        self.assertEqual(result.brake, 1.0)

    def test_early_stop_creeps_toward_line(self):
        vehicle = mock.Mock()
        vehicle.bounding_box.extent.x = 2.4
        vehicle.get_velocity.return_value = drive.carla.Vector3D()
        control = drive.carla.VehicleControl(
            throttle=0.0, steer=0.4, brake=1.0)
        # The default reference point is just before the trigger line.
        result = drive.stop_at_line(control, vehicle, 4.0)
        self.assertGreater(result.throttle, 0.0)
        self.assertEqual(result.brake, 0.0)
        self.assertEqual(result.steer, 0.0)

    def test_signal_stop_target_is_immediately_before_line(self):
        vehicle = mock.Mock()
        vehicle.bounding_box.extent.x = 2.4
        vehicle.get_velocity.return_value = drive.carla.Vector3D()
        control = drive.carla.VehicleControl(throttle=0.4, steer=0.2)
        result = drive.stop_at_line(control, vehicle, 0.3)
        self.assertEqual(drive.STOP_TARGET_MARGIN_M, 0.3)
        self.assertEqual(result.throttle, 0.0)
        self.assertEqual(result.brake, 1.0)

    def test_signal_detection_range_grows_with_speed(self):
        vehicle = mock.Mock()
        vehicle.bounding_box.extent.x = 2.4
        vehicle.get_velocity.return_value = drive.carla.Vector3D(0, 0, 0)
        self.assertEqual(drive.traffic_light_detection_distance(vehicle), 20.0)
        vehicle.get_velocity.return_value = drive.carla.Vector3D(15, 0, 0)
        self.assertGreater(
            drive.traffic_light_detection_distance(vehicle), 35.0)

    def test_signal_is_latched_far_away_but_braking_starts_at_six_metres(self):
        self.assertFalse(drive.signal_stop_is_due(20.0))
        self.assertTrue(drive.signal_stop_is_due(6.0))
        self.assertTrue(drive.signal_stop_is_due(0.3))

    def test_stop_distance_uses_vehicle_heading(self):
        vehicle = mock.Mock()
        vehicle.get_location.return_value = drive.carla.Location(0.0, 0.0, 0.0)
        vehicle.get_transform.return_value = drive.carla.Transform(
            drive.carla.Location(), drive.carla.Rotation(yaw=0.0))
        self.assertAlmostEqual(
            drive.distance_to_stop_line(
                vehicle, drive.carla.Location(8.0, 3.0, 0.0)),
            8.0)

    def test_cruise_speed_cap_limits_speed_limit_oscillation(self):
        agent = mock.Mock()
        agent._speed_limit = 50.0
        agent._update_information.side_effect = (
            lambda: setattr(agent, "_speed_limit", 50.0))

        drive.cap_agent_cruise_speed(agent, 35.0)
        agent._update_information()

        self.assertEqual(agent._speed_limit, 35.0)

    def test_short_time_to_line_commands_full_brake(self):
        vehicle = mock.Mock()
        vehicle.bounding_box.extent.x = 2.4
        vehicle.get_velocity.return_value = drive.carla.Vector3D(10, 0, 0)
        control = drive.carla.VehicleControl(throttle=0.8)
        result = drive.stop_at_line(control, vehicle, 9.0)
        self.assertEqual(result.throttle, 0.0)
        self.assertEqual(result.brake, 1.0)

    def test_green_rollout_caps_speed_throttle_and_steer(self):
        vehicle = mock.Mock()
        vehicle.get_velocity.return_value = drive.carla.Vector3D(4.0, 0, 0)
        control = drive.carla.VehicleControl(
            throttle=0.8, steer=0.8, brake=0.0)
        result = drive.limit_junction_control(
            control, vehicle, target_speed_kph=8.0,
            max_steer=0.38, max_throttle=0.20)
        self.assertEqual(result.throttle, 0.0)
        self.assertAlmostEqual(result.steer, 0.38)
        self.assertGreaterEqual(result.brake, 0.18)

    def test_route_junction_check_uses_planned_waypoints(self):
        agent = mock.Mock()
        agent._vehicle.get_location.return_value = drive.carla.Location()
        near = mock.Mock()
        near.transform.location = drive.carla.Location(x=5.0)
        near.is_junction = True
        agent._local_planner.get_plan.return_value = [
            (near, drive.RoadOption.RIGHT)]
        self.assertTrue(drive.route_near_junction(agent, 25.0))

    def test_junction_speed_gate_uses_current_waypoint(self):
        carla_map = mock.Mock()
        vehicle = mock.Mock()
        waypoint = mock.Mock()
        carla_map.get_waypoint.return_value = waypoint

        waypoint.is_junction = False
        self.assertFalse(drive.vehicle_in_junction(carla_map, vehicle))
        waypoint.is_junction = True
        self.assertTrue(drive.vehicle_in_junction(carla_map, vehicle))

    def test_lane_change_signal_selects_direction_and_preserves_lights(self):
        vehicle = mock.Mock()
        vehicle.get_light_state.return_value = (
            drive.carla.VehicleLightState.LowBeam
            | drive.carla.VehicleLightState.RightBlinker)

        drive.set_lane_change_signal(vehicle, "left")

        applied = vehicle.set_light_state.call_args.args[0]
        self.assertTrue(
            int(applied) & int(drive.carla.VehicleLightState.LowBeam))
        self.assertTrue(
            int(applied) & int(drive.carla.VehicleLightState.LeftBlinker))
        self.assertFalse(
            int(applied) & int(drive.carla.VehicleLightState.RightBlinker))


if __name__ == "__main__":
    unittest.main()
