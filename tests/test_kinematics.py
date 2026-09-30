import math
import unittest

import numpy as np

from fly_brain_rc.config import DriveConfig
from fly_brain_rc.sim.kinematics import BicycleState, step_bicycle, yaw_quaternion


class KinematicsTests(unittest.TestCase):
    def test_yaw_quaternion_is_unit_rotation_about_z(self):
        q = yaw_quaternion(math.pi / 2)

        np.testing.assert_allclose(q, [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)])
        self.assertTrue(math.isclose(float(np.linalg.norm(q)), 1.0))


    def test_straight_command_does_not_change_heading(self):
        drive = DriveConfig(max_speed=1.0, linear_accel=100.0, linear_decel=100.0)
        state = BicycleState(x=0.0, y=0.0, yaw=0.25, speed=0.0)

        result = step_bicycle(state, throttle=1.0, steer=0.0, drive=drive, dt=0.1)

        self.assertTrue(math.isclose(result.yaw, 0.25))
        self.assertGreater(result.x, 0.09)
        self.assertGreater(result.y, 0.02)


    def test_left_steer_follows_bicycle_turn_radius(self):
        drive = DriveConfig(
            max_speed=1.0,
            wheelbase=0.2,
            max_steer_rad=math.atan(0.2),
            linear_accel=100.0,
            linear_decel=100.0,
            angular_accel=100.0,
        )
        state = BicycleState(x=0.0, y=0.0, yaw=0.0, speed=0.0)

        result = step_bicycle(state, throttle=1.0, steer=1.0, drive=drive, dt=0.1)

        self.assertTrue(math.isclose(result.yaw, 0.1, rel_tol=1e-6))
        self.assertGreater(result.y, 0.0)


    def test_speed_change_is_limited_once_by_dt(self):
        drive = DriveConfig(max_speed=1.0, linear_accel=2.0, linear_decel=4.0)

        accelerated = step_bicycle(BicycleState(), 1.0, 0.0, drive, 0.1)
        braked = step_bicycle(BicycleState(speed=1.0), 0.0, 0.0, drive, 0.1)

        self.assertTrue(math.isclose(accelerated.speed, 0.2))
        self.assertTrue(math.isclose(braked.speed, 0.6))

    def test_yaw_rate_change_is_limited_once_by_dt(self):
        drive = DriveConfig(
            max_speed=1.0,
            wheelbase=0.2,
            max_steer_rad=0.5,
            linear_accel=100.0,
            angular_accel=2.0,
        )

        result = step_bicycle(BicycleState(speed=1.0), 1.0, 1.0, drive, 0.1)

        self.assertTrue(math.isclose(result.yaw_rate, 0.2))
        self.assertTrue(math.isclose(result.yaw, 0.02))


if __name__ == "__main__":
    unittest.main()
