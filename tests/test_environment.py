import unittest
import math

import mujoco

from fly_brain_rc.config import DriveConfig, SimConfig
from fly_brain_rc.sim.env import FlyCarEnv
from fly_brain_rc.sim.car_model import build_model_xml
from fly_brain_rc.sim.kinematics import BicycleState


class EnvironmentGeometryTests(unittest.TestCase):
    def make_env(self):
        env = FlyCarEnv.__new__(FlyCarEnv)
        env.sim_cfg = SimConfig(arena_half=3.0, obstacle_radius=0.28)
        env.drive = DriveConfig(collision_radius=0.17)
        env.obstacles = [((1.0, 0.0), (1.0, 0.0, 0.0))]
        return env

    def test_candidate_inside_obstacle_is_blocked(self):
        env = self.make_env()
        self.assertTrue(env._pose_collides(BicycleState(x=0.7, y=0.0)))
        self.assertFalse(env._pose_collides(BicycleState(x=0.4, y=0.0)))

    def test_candidate_crossing_inner_wall_is_blocked(self):
        env = self.make_env()
        self.assertTrue(env._pose_collides(BicycleState(x=2.9, y=0.0)))
        self.assertFalse(env._pose_collides(BicycleState(x=2.7, y=0.0)))

    def test_contact_count_only_changes_on_contact_edge(self):
        env = self.make_env()
        env.collisions = 0
        env._contact_active = False

        env._record_contact(True)
        env._record_contact(True)
        env._record_contact(False)
        env._record_contact(True)

        self.assertEqual(env.collisions, 2)

    def test_collision_radius_encloses_front_wheels(self):
        drive = DriveConfig()
        front_corner = ((drive.wheelbase + drive.wheel_radius) ** 2 + (drive.track / 2 + 0.022) ** 2) ** 0.5
        self.assertGreaterEqual(drive.collision_radius, front_corner)

    def test_model_has_rear_drive_wheels_at_state_origin(self):
        drive = DriveConfig()
        xml = build_model_xml(SimConfig(), drive, [], (1.0, 0.0))
        model = mujoco.MjModel.from_xml_string(xml)
        self.assertEqual(model.body("wheel_rl").pos[0], 0.0)
        self.assertEqual(model.body("wheel_rr").pos[0], 0.0)
        self.assertAlmostEqual(model.body("front_axle").pos[0], drive.wheelbase)


class EnvironmentIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.env = FlyCarEnv(SimConfig(n_obstacles=0), DriveConfig(), seed=3)

    def tearDown(self):
        self.env.renderer.close()

    def test_synced_pose_uses_z_axis_quaternion_and_reads_back_yaw(self):
        self.env.state = BicycleState(yaw=math.pi / 2)
        self.env._sync_kinematic()
        self.assertAlmostEqual(self.env.d.qpos[4], 0.0)
        self.assertAlmostEqual(self.env.d.qpos[5], 0.0)
        self.assertAlmostEqual(self.env.d.qpos[3], 2 ** -0.5)
        self.assertAlmostEqual(self.env.d.qpos[6], 2 ** -0.5)
        self.assertAlmostEqual(self.env._yaw(), math.pi / 2)

    def test_blocked_candidate_rolls_back_and_reverse_can_escape(self):
        self.env.obstacles = [((0.8, 0.0), (0.2, 0.5, 0.8))]
        self.env.drive.max_speed = 5.0
        self.env.drive.linear_accel = 100.0

        self.env.step_kinematic(1.0, 0.0, 0.1)
        self.assertAlmostEqual(self.env.state.x, 0.0)
        self.assertTrue(self.env._contact_active)

        self.env.step_kinematic(-1.0, 0.0, 0.1)
        self.assertLess(self.env.state.x, 0.0)
        self.assertFalse(self.env._contact_active)


if __name__ == "__main__":
    unittest.main()
