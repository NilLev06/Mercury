"""Sensor-only reactive bicycle controller. No simulator or teacher imports."""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class Config:
    goal_gain: float = 1.6
    front_margin: float = .55
    brake_area: float = .7
    avoid_gain: float = .8
    cruise: float = .62
    stop_range: float = .13
    bypass_hold: float = .6
    bypass_distance: float = .65
    retreat_duration: float = .7
    rear_margin: float = .20


class ReactiveController:
    def __init__(self, config=None):
        self.config = config or Config()
        self.reset()

    def reset(self):
        self.mode = 'seek'
        self.side = 0
        self.elapsed = 0.
        self.last_bearing = 0.
        self.stalled = 0.
        self.bypass_travel = 0.

    def action(self, features, dt=.1):
        x = np.asarray(features, dtype=float)
        if x.shape != (31,) or not np.isfinite(x).all() or not math.isfinite(dt) or dt <= 0:
            raise ValueError('finite 31-vector and positive dt required')
        c = self.config
        target, bearing, area = x[:3]
        obs, obs_bearing, obs_area = x[3:6]
        front, left, right, side_left, side_right, rear = x[6:12]*1.2
        speed = x[12]*.35
        contact = x[18] > .5
        # Image right is world left for this camera; larger bin index is left.
        right_occupancy = float(np.mean(x[19:25]))
        left_occupancy = float(np.mean(x[25:31]))
        center_occupancy = float(np.mean(x[23:27]))
        if target > .5:
            self.last_bearing = bearing
        goal_front = (target > .5 and abs(bearing) < .20 and area >= .35
                      and not (obs > .5 and abs(obs_bearing) < .25 and obs_area > .12))
        # Area saturates outside the arrival radius; range is necessary here.
        if self.mode == 'stop' or (goal_front and area >= c.brake_area and front <= c.stop_range):
            self.mode = 'stop'
            return 0., 0.
        if self.mode == 'retreat':
            self.elapsed += dt
            if self.elapsed < c.retreat_duration and rear > c.rear_margin:
                return -.3, -float(self.side)
            self.mode, self.elapsed = 'bypass', 0.
            self.bypass_travel = 0.
        if contact and self.mode != 'retreat' and rear > c.rear_margin:
            if not self.side:
                self.side = 1 if left+side_left >= right+side_right else -1
            self.mode, self.elapsed = 'retreat', 0.
            return -.3, -float(self.side)
        front_risk = front < c.front_margin and not goal_front
        visual_risk = (obs > .5 and abs(obs_bearing) < .35 and obs_area > .12) or center_occupancy > .08
        diagonal_risk = min(left,right) < .70*c.front_margin
        risk = front_risk or visual_risk or diagonal_risk
        if self.mode == 'seek' and risk:
            clearance = left+side_left-right-side_right + 2*(right_occupancy-left_occupancy)
            self.side = (1 if clearance > 0 else -1) if abs(clearance) > .04 else (-1 if obs_bearing >= 0 else 1)
            self.mode, self.elapsed = 'bypass', 0.
            self.bypass_travel = 0.
        steer = c.goal_gain*(bearing if target > .5 else self.last_bearing)
        if self.mode == 'bypass':
            self.elapsed += dt
            self.bypass_travel += abs(speed)*dt
            if self.elapsed >= c.bypass_hold and self.bypass_travel >= c.bypass_distance and not risk:
                self.mode, self.side, self.elapsed = 'seek', 0, 0.
            else:
                steer = self.side*c.avoid_gain + .20*steer
        # Repulsion from a close side; distances are in metres.
        steer += .35*(max(0,.28-side_right)-max(0,.28-side_left))/.28
        if target < .5 and self.mode == 'seek':
            steer = .9*(1 if self.last_bearing >= 0 else -1)
        steer = float(np.clip(steer,-1,1))
        throttle = c.cruise*(1-.45*abs(steer))
        if target < .5: throttle = min(throttle,.30)
        if goal_front:
            throttle = min(throttle, max(.08,1.5*(front-c.stop_range)))
        elif front < .20:
            throttle = min(throttle,.12)
        if contact and rear <= c.rear_margin:
            throttle = 0.
        self.stalled = self.stalled+dt if abs(speed)<.015 and throttle>.1 else 0.
        if self.stalled > 1.5 and rear > c.rear_margin:
            self.mode, self.elapsed, self.stalled = 'retreat', 0., 0.
            if not self.side: self.side = 1 if left >= right else -1
            return -.3,-float(self.side)
        return float(np.clip(throttle,-1,1)),steer
