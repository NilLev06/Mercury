"""Sensor-only, committed bicycle escape; no world pose or target coordinates."""


class RecoveryManoeuvre:
    def __init__(self):
        self.phase = 'drive'
        self.turn = 1
        self.distance = self.rotation = 0.0
        self.started = self.last_time = None
        self.last_impact = -float('inf')
        self.was_contact = False
        self.retreat_distance = .85
        self.arc_angle = 1.7

    def _phase(self, phase, time):
        self.phase, self.started = phase, time
        self.distance = self.rotation = 0.0

    def observe(self, sensor, previous_traction, obstacle=None, target=None):
        dt = 0 if self.last_time is None else max(0, sensor.timestamp-self.last_time)
        self.last_time = sensor.timestamp
        if self.phase != 'drive':
            self.distance += abs(sensor.encoder_delta)
            self.rotation += self.turn * sensor.yaw_rate * dt
        impact = sensor.contact and not self.was_contact
        imminent = (self.phase == 'drive' and sensor.tof < .30
                    and (previous_traction is None or previous_traction > 0)
                    and not (target is not None and target.bbox[0] <= .5 <= target.bbox[2] and obstacle is None))
        if impact or imminent:
            repeated = sensor.timestamp-self.last_impact < 8
            if not repeated:
                self.turn = 1 if sensor.range_left >= sensor.range_right else -1
            self.last_impact = sensor.timestamp
            self.retreat_distance = .85 if obstacle is None else .35
            self.arc_angle = 1.7 if obstacle is None else 1.1
            rear = sensor.contact_rear or (not sensor.contact_front and previous_traction is not None and previous_traction < 0)
            if rear and sensor.contact_right:
                self.turn = -1
            elif rear and sensor.contact_left:
                self.turn = 1
            self._phase('arc' if rear else 'retreat',sensor.timestamp)
        elif self.phase != 'drive':
            elapsed = sensor.timestamp-self.started
            if self.phase == 'retreat' and (self.distance >= self.retreat_distance or elapsed >= 4.5):
                self._phase('arc',sensor.timestamp)
            elif self.phase == 'arc' and (self.rotation >= self.arc_angle and self.distance >= .35 or elapsed >= 6):
                self._phase('clear',sensor.timestamp)
            elif self.phase == 'clear' and (self.distance >= .35 or elapsed >= 3):
                self._phase('drive',sensor.timestamp)
        self.was_contact = sensor.contact

    def command(self, sensor):
        if self.phase == 'drive':
            return None
        if self.phase == 'retreat':
            # Straight backing creates real clearance before the forward arc.
            if sensor.contact_rear:
                self._phase('arc',sensor.timestamp)
            else:
                return -.7, 0.0
        if sensor.contact_front:
            self._phase('retreat',sensor.timestamp)
            return -.7, 0.0
        return (.7, self.turn*.75) if self.phase == 'arc' else (.3, 0.0)
