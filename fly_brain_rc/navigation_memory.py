"""Short egocentric goal recall and committed obstacle bypass, without a map."""
import math
from dataclasses import replace


class NavigationMemory:
    def __init__(self, ttl=12.0):
        self.ttl = ttl
        self.goal = None
        self.vector = None
        self.seen_at = self.last_time = None
        self.phase = 'drive'
        self.side = 0
        self.distance = self.rotation = 0.0
        self.started = None
        self.cooldown_until = -float('inf')

    @property
    def goal_bearing(self):
        return None if self.vector is None else math.atan2(self.vector[1],self.vector[0])

    @property
    def confidence(self):
        if self.goal is None:
            return 0.0
        return self.goal.confidence*math.exp(-(self.last_time-self.seen_at)/6)

    def _phase(self, phase, time):
        self.phase, self.started = phase,time
        self.distance = self.rotation = 0.0

    def observe(self, sensor, target, obstacle, *, recovering=False, recovered=False):
        dt = 0 if self.last_time is None else max(0,sensor.timestamp-self.last_time)
        self.last_time = sensor.timestamp
        if self.vector is not None:
            x,y = self.vector
            x -= sensor.encoder_delta
            angle = sensor.yaw_rate*dt
            c,s = math.cos(angle),math.sin(angle)
            self.vector = (c*x+s*y,-s*x+c*y)
        if target is not None:
            # Approximate monocular size prior, not a depth measurement. The
            # camera frame is ~90deg horizontal; normalized bearing is not radians.
            angle = math.atan(target.bearing)
            distance = max(.3,min(6.0,.18/math.sqrt(max(target.apparent_area,.001))))
            self.vector = (distance*math.cos(angle),distance*math.sin(angle))
            self.goal,self.seen_at = target,sensor.timestamp
        elif self.seen_at is not None and sensor.timestamp-self.seen_at > self.ttl:
            self.goal = self.vector = None

        if recovered and self.phase != 'drive':
            self._phase('along',sensor.timestamp)
        if self.phase == 'drive':
            close = (obstacle is not None and abs(obstacle.bearing) < .5
                     and obstacle.apparent_area > .025 and sensor.tof < .8)
            if close and sensor.timestamp >= self.cooldown_until:
                if abs(sensor.range_left-sensor.range_right) > .1:
                    self.side = 1 if sensor.range_left > sensor.range_right else -1
                else:
                    self.side = -1 if obstacle.bearing > 0 else 1
                self._phase('outward',sensor.timestamp)
        elif not recovering:
            self.distance += max(0.0,sensor.encoder_delta)
            self.rotation += self.side*sensor.yaw_rate*dt
            elapsed = sensor.timestamp-self.started
            if self.phase == 'outward' and ((self.rotation >= .9 and self.distance >= .35) or elapsed >= 4):
                self._phase('along',sensor.timestamp)
            elif self.phase == 'along' and (self.distance >= .55 or elapsed >= 3.5):
                self._phase('drive',sensor.timestamp)
                self.cooldown_until = sensor.timestamp+1.0

    def recalled_target(self, sensor):
        if self.goal is None or self.vector is None:
            return None
        angle = self.goal_bearing
        bearing = max(-1.0,min(1.0,math.tan(angle))) if abs(angle) < math.pi/2 else math.copysign(1.0,angle)
        confidence = self.goal.confidence*math.exp(-(sensor.timestamp-self.seen_at)/6)
        return replace(self.goal,bearing=bearing,confidence=confidence,timestamp=sensor.timestamp)

    def command(self):
        if self.phase == 'outward':
            return .7,self.side*.75
        if self.phase == 'along':
            return .7,0.0
        return None
