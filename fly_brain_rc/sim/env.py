import math
import random

import numpy as np
import mujoco

from ..config import SimConfig, DriveConfig
from .car_model import build_model_xml


class FlyCarEnv:
    def __init__(self, sim_cfg: SimConfig, drive: DriveConfig, seed: int = 0):
        self.sim_cfg = sim_cfg
        self.drive = drive
        self._rng = random.Random(seed)
        self.obstacles = []
        self.goal = (2.0, 0.0)
        self._build_episode()
        self.renderer = mujoco.Renderer(self.m, sim_cfg.cam_width, sim_cfg.cam_height)
        self.collisions = 0
        self.sim_steps = 0
        self.start_pose = None
        self.reset()

    def _build_episode(self):
        cfg = self.sim_cfg
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
            obstacles.append(((x, y), (self._rng.uniform(0.3, 1.0), self._rng.uniform(0.3, 1.0), self._rng.uniform(0.3, 1.0))))
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
        self.start_pose = self.car_pose()
        self.collisions = 0
        self.sim_steps = 0
        self.speed = 0.0
        self.heading = float(self.start_pose[2])
        self._w_cmd = 0.0
        self._w_cmd_prev = 0.0
        for i in range(self.m.nq):
            self.d.qpos[i] = 0.0
        self.d.qpos[2] = 0.05
        self.d.qpos[3] = 1.0
        x, y = self.start_pose[0], self.start_pose[1]
        self.d.qpos[0], self.d.qpos[1] = x, y
        self.d.qpos[3], self.d.qpos[4], self.d.qpos[5], self.d.qpos[6] = (
            math.cos(self.heading / 2), 0.0, math.sin(self.heading / 2), math.cos(self.heading / 2))
        self._sync_kinematic()
        return self.observe()

    def _indices(self):
        m = self.m
        take = lambda x: int(np.asarray(x).reshape(-1)[0])
        self.i_motor_l = m.actuator("motor_fl").id
        self.i_motor_r = m.actuator("motor_fr").id
        self.adr_b_fl = take(m.sensor("b_fl").adr)
        self.adr_b_fr = take(m.sensor("b_fr").adr)
        self.adr_b_fc = take(m.sensor("b_fc").adr)
        self.adr_gyro = take(m.sensor("gyro").adr)
        self.adr_wl = m.jnt_dofadr[m.joint("wheel_fl").id]
        self.adr_wr = m.jnt_dofadr[m.joint("wheel_fr").id]
        self.jq_steer = m.jnt_qposadr[m.joint("steer_joint").id]
        self.jq_wl = m.jnt_qposadr[m.joint("wheel_fl").id]
        self.jq_wr = m.jnt_qposadr[m.joint("wheel_fr").id]
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
        x, _y, z, w = self.d.qpos[3], self.d.qpos[4], self.d.qpos[5], self.d.qpos[6]
        return math.atan2(2.0 * (w * z + x * _y), 1.0 - 2.0 * (_y * _y + z * z))

    def car_pose(self):
        x, y = float(self.d.qpos[0]), float(self.d.qpos[1])
        return x, y, self._yaw()

    def step(self, throttle: float, steer: float, ctrl_substeps: int = 1):
        dt = self.drive.timestep
        self._indices()
        if self.sim_cfg.kinematic:
            self.step_kinematic(throttle, steer, dt)
        else:
            self.step_physical(throttle, steer, ctrl_substeps)
        self.sim_steps += 1
        obs = self.observe()
        if self.sim_cfg.kinematic:
            if self._collision_2d():
                self.collisions += 1
        else:
            if any(v < 0.06 for v in obs["ranges"].values()):
                self.collisions += 1
        return obs

    def step_kinematic(self, throttle: float, steer: float, dt: float):
        w_cmd = steer * self.drive.max_omega
        self._w_cmd += dt * max(-dt * 16.0, min(16.0 * dt, w_cmd - self._w_cmd))
        v_cmd = throttle * self.drive.max_speed
        self.speed += dt * max(-8.0 * dt, min(8.0 * dt, v_cmd - self.speed))
        self.heading += self._w_cmd * dt
        self.d.qpos[0] += self.speed * math.cos(self.heading) * dt
        self.d.qpos[1] += self.speed * math.sin(self.heading) * dt
        self._w_cmd_prev = self._w_cmd
        self._sync_kinematic(self.speed / self.drive.wheel_radius, steer)

    def _collision_2d(self):
        cfg = self.sim_cfg
        thr = cfg.obstacle_radius + 0.07
        for (ox, oy), _ in self.obstacles:
            if math.hypot(self.d.qpos[0] - ox, self.d.qpos[1] - oy) < thr:
                return True
        return False

    def _sync_kinematic(self, wheel_spin: float = 0.0, steer: float = 0.0):
        self.d.qpos[2] = 0.05
        self.d.qpos[3], self.d.qpos[4], self.d.qpos[5], self.d.qpos[6] = (
            math.cos(self.heading / 2), 0.0, math.sin(self.heading / 2), math.cos(self.heading / 2))
        self.d.qpos[self.jq_steer] = steer * self.drive.max_steer_rad
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