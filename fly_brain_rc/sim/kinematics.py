import math
from dataclasses import dataclass

import numpy as np

from ..config import DriveConfig


@dataclass(frozen=True)
class BicycleState:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    speed: float = 0.0
    yaw_rate: float = 0.0


def yaw_quaternion(yaw: float) -> np.ndarray:
    """Return MuJoCo's (w, x, y, z) quaternion for planar yaw."""
    return np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)])


def _move_toward(value: float, target: float, amount: float) -> float:
    if value < target:
        return min(value + amount, target)
    return max(value - amount, target)


def step_bicycle(
    state: BicycleState,
    throttle: float,
    steer: float,
    drive: DriveConfig,
    dt: float,
) -> BicycleState:
    throttle = max(-1.0, min(1.0, float(throttle)))
    steer = max(-1.0, min(1.0, float(steer)))
    target_speed = throttle * drive.max_speed
    accelerating = abs(target_speed) > abs(state.speed) and target_speed * state.speed >= 0.0
    rate = drive.linear_accel if accelerating else drive.linear_decel
    speed = _move_toward(state.speed, target_speed, rate * dt)

    delta = steer * drive.max_steer_rad
    target_yaw_rate = speed * math.tan(delta) / drive.wheelbase
    yaw_rate = _move_toward(state.yaw_rate, target_yaw_rate, drive.angular_accel * dt)
    next_yaw = state.yaw + yaw_rate * dt

    if abs(yaw_rate) < 1e-9:
        x = state.x + speed * math.cos(state.yaw) * dt
        y = state.y + speed * math.sin(state.yaw) * dt
    else:
        radius = speed / yaw_rate
        x = state.x + radius * (math.sin(next_yaw) - math.sin(state.yaw))
        y = state.y - radius * (math.cos(next_yaw) - math.cos(state.yaw))
    yaw = (next_yaw + math.pi) % (2.0 * math.pi) - math.pi
    return BicycleState(x=x, y=y, yaw=yaw, speed=speed, yaw_rate=yaw_rate)
