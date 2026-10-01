import numpy as np

from ..config import DriveConfig
from .goal_vision import GoalDetector
from .looming import LoomingDetector
from .optic_flow import H1Steering, OpticFlowBrain
from .reflex import BristleReflex
from .ring_attractor import RingAttractor


class FlyBrainController:
    """Deterministic visual search controller with insect-like escape reflexes."""

    def __init__(
        self,
        drive: DriveConfig,
        ring: RingAttractor,
        flow_brain: OpticFlowBrain,
        h1: H1Steering,
        loom: LoomingDetector,
        reflex: BristleReflex,
    ):
        self.drive = drive
        self.ring = ring
        self.flow_brain = flow_brain
        self.h1 = h1
        self.loom = loom
        self.reflex = reflex
        self.goal_detector = GoalDetector()
        self.mode = "search"
        self.goal_observation = self.goal_detector.step(np.zeros((1, 1, 3), dtype=np.uint8))
        self._search_time = 0.0
        self._search_direction = 1.0
        self._avoid_time = 0.0
        self._avoid_direction = 1.0
        self._target_memory = 0.0
        self._last_target_bearing = 0.0
        self._arrival_time = 0.0

    def _search_command(self, dt: float) -> tuple[float, float]:
        self._search_time += dt
        period = 16.0
        if self._search_time >= period:
            self._search_time -= period
            self._search_direction *= -1.0
        if self._search_time < 11.0:
            return 0.70, self._search_direction
        return 0.70, 0.08 * self._search_direction

    def step(
        self,
        frame_rgb: np.ndarray,
        frame_gray: np.ndarray,
        ranges: dict,
        yaw_rate_deg: float,
        dt: float,
    ) -> tuple[float, float]:
        del yaw_rate_deg
        flow = self.flow_brain.step(frame_gray)
        looming = self.loom.step(frame_gray, dt)
        reflex = self.reflex.step(ranges, dt)
        goal = self.goal_detector.step(frame_rgb)
        self.goal_observation = goal
        self._arrival_time = max(0.0, self._arrival_time - dt)
        close_goal_in_view = goal.visible and goal.area > 0.10 and abs(goal.bearing) < 0.55
        if close_goal_in_view:
            self._arrival_time = 3.0

        if ranges.get("contact", False):
            self.mode = "avoid"
            left = ranges.get("front_left", 1000.0)
            right = ranges.get("front_right", 1000.0)
            if left < right:
                self._avoid_direction = -1.0
            elif right < left:
                self._avoid_direction = 1.0
            else:
                self._avoid_direction *= -1.0
            self._avoid_time = 4.0
            return -0.70, 0.95 * self._avoid_direction

        if self._avoid_time > 0.0:
            self.mode = "avoid"
            self._avoid_time = max(0.0, self._avoid_time - dt)
            if self._avoid_time > 2.4:
                return -0.62, 0.9 * self._avoid_direction
            return 0.48, 0.95 * self._avoid_direction

        validated_looming = (
            looming["alarm"]
            and flow["divergence"] > self.loom.cfg.min_divergence
            and not goal.visible
        )
        if validated_looming:
            self.mode = "avoid"
            self._avoid_direction = -1.0 if looming["side"] > 0.0 else 1.0
            self._avoid_time = 3.0
            return -0.45, self._avoid_direction

        front_left = ranges.get("front_left", 1000.0)
        front_right = ranges.get("front_right", 1000.0)
        front = min(front_left, ranges.get("center", 1000.0), front_right)
        goal_is_centered = (
            (goal.visible and abs(goal.bearing) < 0.55)
            or (self._target_memory > 0.0 and abs(self._last_target_bearing) < 0.55)
        )
        center_range = ranges.get("center", 1000.0)
        side_range = min(front_left, front_right)
        goal_range_signature = side_range < 0.25 and center_range > side_range + 0.25
        isolated_center_obstacle = (
            center_range < 0.06
            and min(front_left, front_right) > center_range + 0.15
        )
        goal_envelops_sensors = (
            self._arrival_time > 0.0
            and goal.visible
            and abs(goal.bearing) < 0.55
            and max(front_left, center_range, front_right) < 0.08
        )
        target_is_close_ahead = goal_is_centered and (
            goal_range_signature
            or (close_goal_in_view and not isolated_center_obstacle)
            or goal_envelops_sensors
            or (self._arrival_time > 0.0 and center_range > 0.06)
        )
        scanning_from_safe_start = not goal.visible and self._target_memory <= 0.0 and self._search_time < 11.0
        avoid_threshold = 0.12 if scanning_from_safe_start else 0.38
        reflex_trigger = reflex["active"] and not scanning_from_safe_start
        if (front < avoid_threshold or reflex_trigger) and not target_is_close_ahead:
            self.mode = "avoid"
            self._avoid_direction = reflex["steer"] or (-1.0 if front_left < front_right else 1.0)
            self._avoid_time = 3.0
            return -0.62, 0.9 * self._avoid_direction

        if goal.visible:
            self.mode = "approach"
            self._last_target_bearing = goal.bearing
            self._target_memory = 2.0
            steer = float(np.clip(1.8 * goal.bearing + 0.12 * self.h1.steer(flow), -1.0, 1.0))
            throttle = 0.62 * (1.0 - 0.55 * abs(steer))
            if goal.area > 0.08:
                throttle *= 0.45
            return max(0.16, throttle), steer

        if self._target_memory > 0.0:
            self.mode = "approach"
            self._target_memory = max(0.0, self._target_memory - dt)
            return 0.30, float(np.clip(1.5 * self._last_target_bearing, -1.0, 1.0))

        self.mode = "search"
        throttle, steer = self._search_command(dt)
        if self._search_time >= 11.0:
            flow_bias = self.h1.steer(flow)
            if abs(flow_bias) > 0.25:
                steer += 0.20 * flow_bias
        return throttle, float(np.clip(steer, -1.0, 1.0))
