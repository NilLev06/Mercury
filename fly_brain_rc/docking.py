"""Simulator-calibrated target/range fusion, shared by learning and viewer.

Known sphere size and camera calibration are priors, not world-state inputs.
Uncertain/occluded estimates never authorize ignoring a short range reading.
"""
from dataclasses import dataclass
import math
from .config import DRIVE


@dataclass(frozen=True)
class TargetView:
    distance: float | None = None
    stage: str = 'unseen'
    rays: tuple[bool, bool, bool] = (False,False,False)
    parking: bool = False


def ranges(sensor):
    return (sensor.range_left,sensor.tof,sensor.range_right,
            sensor.range_side_left,sensor.range_side_right,sensor.range_rear)


def target_view(sensor, target):
    if target is None or target.confidence < .5:
        return TargetView()
    left,_,right,_ = target.bbox
    if not (0.01 < left < right < .99) or sensor.rgb.shape[0] == 0:
        return TargetView(stage='uncertain')
    focal = math.tan(math.radians(75)/2)*sensor.rgb.shape[1]/sensor.rgb.shape[0]
    a,b = math.atan((2*left-1)*focal),math.atan((2*right-1)*focal)
    half = (b-a)/2
    if half <= .005:
        return TargetView(stage='uncertain')
    radius = .22
    length = radius/math.sin(half)
    angle = (a+b)/2  # eye camera x axis is the car's positive lateral axis
    x = DRIVE.wheelbase*.7 + length*math.cos(angle)
    y = length*math.sin(angle)
    distance = math.hypot(x,y)
    edge = DRIVE.track/2+.022
    # Ray plane z=.17, ball centre z=.22; use its horizontal cross section.
    cross_radius = math.sqrt(radius**2-.05**2)
    matches = []
    for value,origin,theta in zip(ranges(sensor)[:3],
            ((DRIVE.wheelbase+.04,edge),(DRIVE.wheelbase+.04,0),(DRIVE.wheelbase+.04,-edge)),
            (math.pi/4,0,-math.pi/4)):
        dx,dy = x-origin[0],y-origin[1]
        projection = dx*math.cos(theta)+dy*math.sin(theta)
        perpendicular2 = max(0,dx*dx+dy*dy-projection*projection)
        expected = (projection-math.sqrt(cross_radius**2-perpendicular2)
                    if projection > 0 and perpendicular2 < cross_radius**2 else None)
        # Tight asymmetric tolerance: nearer occluders must not inherit the
        # target's exemption. Strongly clipped views were rejected above.
        matches.append(expected is not None and expected >= 0 and value < 1.2
                       and expected-.025 <= value <= expected+.08)
    safe = not sensor.contact and min(ranges(sensor)) >= .08
    parking = .60 <= distance <= .85 and abs(target.bearing) <= .18 and matches[1] and safe
    stage = ('park' if parking else 'too_close' if distance < .60 else
             'near' if distance <= 1.2 else 'far')
    return TargetView(distance,stage,tuple(matches),parking)


def prepare_docking(agent):
    """In-memory upgrade, leaving the original policy file and Q rows intact."""
    if agent.task != 'bypass-v1' or agent.sensor_schema >= 3:
        return
    for action in ((.12,0.0),(0.0,0.0)):
        if action not in agent.actions:
            agent.actions += (action,)
            for row in agent.table.values():
                row.append(0.0)
    agent.sensor_schema = 3


class DockingTracker:
    duration = 1.0

    def __init__(self):
        self.reset()

    def reset(self):
        self.elapsed = 0.0
        self.last_reason = 'docking_waiting'

    def update(self, sensor, target, command, dt):
        valid = (target_view(sensor,target).parking and abs(sensor.speed) <= .025
                 and abs(command.traction) <= .05 and abs(command.steering) <= .05)
        self.elapsed = min(self.duration,self.elapsed+dt) if valid else 0.0
        self.last_reason = 'docking_stable' if valid else 'docking_in_progress'
        return self.elapsed >= self.duration-1e-9
