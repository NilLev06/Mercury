"""Evaluator-only arrival, never supplied to the student."""
import math


def body_goal_distance(env):
    x,y,yaw = env.car_pose()
    offset = env.drive.wheelbase/2
    return math.hypot(env.goal[0]-x-offset*math.cos(yaw),
                      env.goal[1]-y-offset*math.sin(yaw))


class GoalStop:
    def __init__(self):
        self.elapsed = 0.0

    def update(self, distance, speed, dt):
        if not all(math.isfinite(v) for v in (distance,speed,dt)) or distance < 0 or dt <= 0:
            raise ValueError('invalid arrival measurement')
        self.elapsed = self.elapsed+dt if distance <= .5 and abs(speed) <= .03 else 0.0
        return self.elapsed >= .5-1e-9
