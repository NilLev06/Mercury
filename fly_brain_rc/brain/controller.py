import math
import numpy as np
from ..config import DriveConfig
from .ring_attractor import RingAttractor
from .optic_flow import OpticFlowBrain, H1Steering
from .looming import LoomingDetector
from .reflex import BristleReflex


class FlyBrainController:
    def __init__(self, drive: DriveConfig, ring: RingAttractor, flow_brain: OpticFlowBrain,
                 h1: H1Steering, loom: LoomingDetector, reflex: BristleReflex):
        self.drive = drive
        self.ring = ring
        self.flow_brain = flow_brain
        self.h1 = h1
        self.loom = loom
        self.reflex = reflex
        self._stuck_t = 0.0
        self._stuck_escape = 0.0
        self._last_xy = None
        self._escape_dir = 0.0
        self._avoid_hold = 0.0
        self._reflex_was_active = False
        self._ref_active_time = 0.0
        self._ref_lock_dir = 0.0
        self._ref_lock_t = 0.0

    def _stuck_check(self, x: float, y: float, throttle: float, dt: float, steer: float = 0.0,
                     goal_bearing_deg: float = 0.0, escape_hint: float = 0.0) -> float:
        pos = (x, y)
        if self._last_xy is None:
            self._last_xy = pos
            return 0.0
        moved = math.hypot(pos[0] - self._last_xy[0], pos[1] - self._last_xy[1])
        self._last_xy = pos
        self._stuck_escape = max(0.0, self._stuck_escape - dt)
        if abs(throttle) > 0.25 and moved < 0.0012:
            self._stuck_t += dt
        else:
            self._stuck_t = max(0.0, self._stuck_t - 2.0 * dt)
        if self._stuck_t > 1.0 and self._stuck_escape <= 0.0:
            self._stuck_t = 0.0
            self._stuck_escape = 0.9
            hint = escape_hint if abs(escape_hint) > 0.1 else (1.0 if goal_bearing_deg <= 0 else -1.0)
            self._escape_dir = hint
        return self._escape_dir if self._stuck_escape > 0 else 0.0

    def step(self, frame_gray: np.ndarray, ranges: dict, goal_bearing_deg: float,
             yaw_rate_deg: float, dt: float, odom: dict | None = None) -> tuple:
        fl = self.flow_brain.step(frame_gray)
        lom = self.loom.step(frame_gray, dt)
        mask = lom["alarm"]
        ring_heading = self.ring.step(goal_bearing_deg, yaw_rate_deg, dt, masked=mask)
        ref = self.reflex.step(ranges, dt)

        steer_goal = float(np.clip(ring_heading / 90.0, -1.0, 1.0)) * self.drive.goal_steer_gain
        steer_flow = self.h1.steer(fl)
        speed_f = self.h1.throttle_factor(fl)

        throttle = self.drive.base_throttle * speed_f
        if odom is not None and self.drive.slow_band > 0.0:
            fb = odom["dist_to_goal"] / self.drive.slow_band
            slow = max(0.12, fb * fb)
            cosb = (math.cos(math.radians(odom["goal_bearing_deg"])) + 1.0) * 0.5
            throttle *= slow * (0.45 + 0.55 * cosb)
        else:
            slow = 1.0
        steer = float(np.clip(steer_goal + steer_flow, -1.0, 1.0))

        escape_steer = 0.0
        if odom is not None:
            escape_steer = self._stuck_check(odom["x"], odom["y"], throttle, dt, steer,
                                             goal_bearing_deg, self.reflex._evade_dir)
        if escape_steer:
            throttle = -0.6
            steer = escape_steer

        if self._ref_lock_t > 0.0:
            self._ref_lock_t -= dt
            if self._ref_lock_t > 2.0:
                throttle = -0.6
            else:
                throttle = 0.6
            steer = self._ref_lock_dir
        elif ref["active"]:
            last_t = self._ref_active_time
            self._ref_active_time += dt
            if last_t > 1.5:
                self._ref_lock_t = 4.0
                self._ref_lock_dir = -self.reflex._evade_dir
                self._ref_active_time = 0.0
                self._avoid_hold = 4.0
            elif ref["retreat"]:
                throttle = -0.6
                steer = 0.9 * self.reflex._evade_dir
            else:
                throttle = -0.2 * ref["steer"]
                steer = ref["steer"]
            self._avoid_hold = 0.9
            self._reflex_was_active = True
        elif lom["alarm"]:
            throttle = -0.25
            steer = lom["side"]
            self._avoid_hold = max(self._avoid_hold, 0.6)
        elif self._reflex_was_active and self._avoid_hold > 0.0:
            self._avoid_hold -= dt
            throttle = 0.45 * speed_f
            steer = 0.8 * self.reflex._evade_dir
        else:
            self._reflex_was_active = False
            self._ref_active_time = max(0.0, self._ref_active_time - 3.0 * dt)

        return throttle, steer