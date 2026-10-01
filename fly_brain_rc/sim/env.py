import math
import random

import numpy as np
import mujoco

from ..config import SimConfig, DriveConfig
from ..contracts import ActuatorCommand, SensorFrame
from .car_model import build_model_xml
from .kinematics import BicycleState, step_bicycle, yaw_quaternion
from .scenarios import Scenario


class FlyCarEnv:
    def __init__(
        self,
        sim_cfg: SimConfig,
        drive: DriveConfig,
        seed: int = 0,
        scenario: Scenario | None = None,
        render: bool = True,
    ):
        self.sim_cfg = sim_cfg
        self.drive = drive
        self._rng = random.Random(seed)
        self.scenario = scenario
        self.render_enabled = render
        self.obstacles = []
        self.goal = (2.0, 0.0)
        self.renderer = None
        self.collisions = 0
        self.penetrations = 0
        self.sim_steps = 0
        self.start_pose = None
        self.reset()

    def _build_episode(self):
        cfg = self.sim_cfg
        if self.scenario is not None:
            self.obstacles = list(self.scenario.obstacles)
            self.goal = self.scenario.goal
            xml = build_model_xml(cfg, self.drive, self.obstacles, self.goal)
            self.m = mujoco.MjModel.from_xml_string(xml)
            self.d = mujoco.MjData(self.m)
            return
        o = cfg.arena_half
        start = (0.0, 0.0)
        obstacles = []
        for _ in range(cfg.n_obstacles):
            for attempt in range(200):
                x = self._rng.uniform(-o + 1.0, o - 1.0)
                y = self._rng.uniform(-o + 1.0, o - 1.0)
                if math.hypot(x - start[0], y - start[1]) < 1.2:
                    continue
                if all(math.hypot(x - ox, y - oy) > 1.0 for (ox, oy), _ in obstacles):
                    break
            obstacles.append(
                (
                    (x, y),
                    (
                        self._rng.uniform(0.20, 0.45),
                        self._rng.uniform(0.45, 0.75),
                        self._rng.uniform(0.65, 0.95),
                    ),
                )
            )
        gx, gy = None, None
        for attempt in range(200):
            gx = self._rng.uniform(-o + 1.2, o - 1.2)
            gy = self._rng.uniform(-o + 1.2, o - 1.2)
            if math.hypot(gx, gy) > 1.6 and all(math.hypot(gx - ox, gy - oy) > 1.0 for (ox, oy), _ in obstacles):
                break
        self.obstacles = obstacles
        self.goal = (float(gx), float(gy))
        xml = build_model_xml(cfg, self.drive, obstacles, self.goal)
        self.m = mujoco.MjModel.from_xml_string(xml)
        self.d = mujoco.MjData(self.m)

    def reset(self):
        self._build_episode()
        self._indices()
        self.start_pose = (0.0, 0.0, 0.0)
        self.collisions = 0
        self.penetrations = 0
        self.sim_steps = 0
        self.speed = 0.0
        self.heading = float(self.start_pose[2])
        self.state = BicycleState(self.start_pose[0], self.start_pose[1], self.heading, 0.0)
        self._w_cmd = 0.0
        self._w_cmd_prev = 0.0
        self._contact_active = False
        for i in range(self.m.nq):
            self.d.qpos[i] = 0.0
        self.d.qpos[2] = 0.05
        self.d.qpos[3] = 1.0
        x, y = self.start_pose[0], self.start_pose[1]
        self.d.qpos[0], self.d.qpos[1] = x, y
        self._sync_kinematic()
        self._last_encoder_position = self._encoder_position()
        if self.renderer is not None:
            self.renderer.close()
        self.renderer = (
            mujoco.Renderer(self.m, self.sim_cfg.cam_width, self.sim_cfg.cam_height)
            if self.render_enabled else None
        )
        return self.observe() if self.renderer is not None else None

    def _indices(self):
        m = self.m
        take = lambda x: int(np.asarray(x).reshape(-1)[0])
        self.i_motor_l = m.actuator("motor_rl").id
        self.i_motor_r = m.actuator("motor_rr").id
        self.adr_b_fl = take(m.sensor("b_fl").adr)
        self.adr_b_fr = take(m.sensor("b_fr").adr)
        self.adr_b_fc = take(m.sensor("b_fc").adr)
        self.adr_gyro = take(m.sensor("gyro").adr)
        self.adr_wl = m.jnt_dofadr[m.joint("wheel_rl").id]
        self.adr_wr = m.jnt_dofadr[m.joint("wheel_rr").id]
        self.jq_steer = m.jnt_qposadr[m.joint("steer_joint").id]
        self.jq_fl = m.jnt_qposadr[m.joint("wheel_fl").id]
        self.jq_fr = m.jnt_qposadr[m.joint("wheel_fr").id]
        self.jq_wl = m.jnt_qposadr[m.joint("wheel_rl").id]
        self.jq_wr = m.jnt_qposadr[m.joint("wheel_rr").id]
        self._have_indices = True

    def observe(self):
        if not hasattr(self, "_have_indices"):
            self._indices()
        m, d = self.m, self.d
        sd = d.sensordata
        take = lambda adr: float(np.asarray(sd[adr]).reshape(-1)[0])
        front_l = take(self.adr_b_fl)
        front_r = take(self.adr_b_fr)
        front_c = take(self.adr_b_fc)
        prox = lambda v: v if 0.0 < v < 1.2 else 1e3
        ranges = {
            "front_left": prox(front_l),
            "front_right": prox(front_r),
            "center": prox(front_c),
            "contact": self._contact_active,
        }
        yaw = self._yaw()
        if self.sim_cfg.kinematic:
            yaw_rate_deg = math.degrees(getattr(self, "_w_cmd_prev", 0.0))
        else:
            yaw_rate_deg = math.degrees(take(self.adr_gyro))
        x, y = float(d.qpos[0]), float(d.qpos[1])
        gx, gy = self.goal
        goal_bearing_deg = math.degrees(math.atan2(gy - y, gx - x)) - math.degrees(yaw)
        goal_bearing_deg = (goal_bearing_deg + 180.0) % 360.0 - 180.0
        self.renderer.update_scene(d, camera="eye")
        frame = np.asarray(self.renderer.render(), dtype=np.uint8)
        gray = cv_gray(frame)
        odom = {
            "x": x,
            "y": y,
            "yaw": math.degrees(yaw),
            "yaw_rate_deg": yaw_rate_deg,
            "goal_bearing_deg": goal_bearing_deg,
            "dist_to_goal": math.hypot(gx - x, gy - y),
        }
        return dict(frame=frame, gray=gray, ranges=ranges, odom=odom)

    def _yaw(self):
        w, x, y, z = self.d.qpos[3:7]
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def car_pose(self):
        x, y = float(self.d.qpos[0]), float(self.d.qpos[1])
        return x, y, self._yaw()

    def goal_distance(self) -> float:
        """Ground truth for episode evaluation, outside the sensor contract."""
        x, y, _ = self.car_pose()
        gx, gy = self.goal
        return math.hypot(gx - x, gy - y)

    def goal_surface_clearance(self) -> float:
        """Conservative car-envelope to target-sphere gap, evaluator only.

        The bicycle pose is at the rear axle, not at the nose. The car's
        collision envelope has radius ``collision_radius`` and the rendered
        target sphere has radius 0.22 m. Safe arrival is a stand-off from these
        surfaces; the image area band supplies the upper proximity bound.
        """
        return self.goal_distance() - self.drive.collision_radius - 0.22

    def _encoder_position(self) -> float:
        rear_wheel_angle = (float(self.d.qpos[self.jq_wl]) + float(self.d.qpos[self.jq_wr])) / 2.0
        return rear_wheel_angle * self.drive.wheel_radius

    def sensor_frame(self) -> SensorFrame:
        if self.renderer is None:
            raise RuntimeError("sensor_frame requires a camera renderer")
        self.renderer.update_scene(self.d, camera="eye")
        rgb = np.asarray(self.renderer.render(), dtype=np.uint8)
        center_range = float(self.d.sensordata[self.adr_b_fc])
        tof = center_range if 0.0 < center_range < 1.2 else 1.2
        yaw_rate = (
            self._w_cmd_prev if self.sim_cfg.kinematic
            else float(self.d.sensordata[self.adr_gyro + 2])
        )
        encoder_position = self._encoder_position()
        encoder_delta = encoder_position - self._last_encoder_position
        self._last_encoder_position = encoder_position
        return SensorFrame(
            rgb=rgb,
            gray=cv_gray(rgb),
            yaw_rate=yaw_rate,
            encoder_delta=encoder_delta,
            tof=tof,
            contact=self._contact_active,
            timestamp=self.sim_steps * self.drive.timestep,
        )

    def apply(self, command: ActuatorCommand) -> None:
        self.step(command.traction, command.steering, observe=False)

    def step(self, throttle: float, steer: float, ctrl_substeps: int = 1, *, observe=True):
        dt = self.drive.timestep
        self._indices()
        if self.sim_cfg.kinematic:
            self.step_kinematic(throttle, steer, dt)
        else:
            self.step_physical(throttle, steer, ctrl_substeps)
        self.sim_steps += 1
        if not self.sim_cfg.kinematic:
            physical_contact = any(
                0 < float(self.d.sensordata[adr]) < 0.06
                for adr in (self.adr_b_fl, self.adr_b_fr, self.adr_b_fc)
            )
            self._record_contact(physical_contact)
        return self.observe() if observe else None

    def step_kinematic(self, throttle: float, steer: float, dt: float):
        before = self.state
        candidate = step_bicycle(before, throttle, steer, self.drive, dt)
        hit = self._motion_collides(before, candidate)
        self._record_contact(hit)
        if hit:
            candidate = BicycleState(before.x, before.y, before.yaw, 0.0, 0.0)
        self.state = candidate
        self.speed = candidate.speed
        self.heading = candidate.yaw
        self._w_cmd_prev = candidate.yaw_rate
        wheel_spin = candidate.speed / self.drive.wheel_radius
        self._sync_kinematic(wheel_spin, steer)
        if self._pose_collides(self.state):
            self.penetrations += 1

    def _pose_collides(self, state: BicycleState) -> bool:
        limit = self.sim_cfg.arena_half - self.drive.collision_radius
        if abs(state.x) > limit or abs(state.y) > limit:
            return True
        threshold = self.sim_cfg.obstacle_radius + self.drive.collision_radius
        for (ox, oy), _ in self.obstacles:
            if math.hypot(state.x - ox, state.y - oy) < threshold:
                return True
        return False

    def _motion_collides(self, before: BicycleState, after: BicycleState) -> bool:
        distance = math.hypot(after.x - before.x, after.y - before.y)
        samples = max(1, math.ceil(distance / max(self.drive.collision_radius * 0.5, 1e-6)))
        for index in range(1, samples + 1):
            fraction = index / samples
            state = BicycleState(
                x=before.x + (after.x - before.x) * fraction,
                y=before.y + (after.y - before.y) * fraction,
                yaw=before.yaw + (after.yaw - before.yaw) * fraction,
            )
            if self._pose_collides(state):
                return True
        return False

    def _record_contact(self, colliding: bool):
        if colliding and not self._contact_active:
            self.collisions += 1
        self._contact_active = colliding

    def _sync_kinematic(self, wheel_spin: float = 0.0, steer: float = 0.0):
        self.d.qpos[0] = self.state.x
        self.d.qpos[1] = self.state.y
        self.d.qpos[2] = 0.05
        self.d.qpos[3:7] = yaw_quaternion(self.state.yaw)
        self.d.qpos[self.jq_steer] = steer * self.drive.max_steer_rad
        self.d.qpos[self.jq_fl] += wheel_spin * self.drive.timestep
        self.d.qpos[self.jq_fr] += wheel_spin * self.drive.timestep
        self.d.qpos[self.jq_wl] += wheel_spin * self.drive.timestep
        self.d.qpos[self.jq_wr] += wheel_spin * self.drive.timestep
        mujoco.mj_forward(self.m, self.d)

    def step_physical(self, throttle: float, steer: float, ctrl_substeps: int = 1):
        drive = self.drive
        v = throttle * drive.max_speed
        w = steer * drive.max_omega
        wbase = v / drive.wheel_radius
        wdiff = w * drive.track / 2 / drive.wheel_radius
        if not hasattr(self, "_ramp"):
            self._ramp = 0.0
        for _ in range(ctrl_substeps):
            self._ramp += 0.02 * (wbase - self._ramp)
            wl = float(self.d.qvel[self.adr_wl])
            wr = float(self.d.qvel[self.adr_wr])
            self.d.ctrl[self.i_motor_l] = drive.wheel_gain * ((self._ramp - wdiff) - wl)
            self.d.ctrl[self.i_motor_r] = drive.wheel_gain * ((self._ramp + wdiff) - wr)
            mujoco.mj_step(self.m, self.d)


def cv_gray(rgb):
    arr = np.ascontiguousarray(rgb[..., :3])
    return arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114
