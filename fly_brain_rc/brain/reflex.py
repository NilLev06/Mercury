from ..config import ReflexConfig


class BristleReflex:
    def __init__(self, cfg: ReflexConfig):
        self.cfg = cfg
        self.retreat_left = 0.0
        self.evade_left = 0.0
        self._evade_dir = 0.0

    def step(self, ranges: dict, dt: float) -> dict:
        cfg = self.cfg
        self.retreat_left = max(0.0, self.retreat_left - dt)
        self.evade_left = max(0.0, self.evade_left - dt)
        front = min(ranges.get("center", 9.0), ranges.get("front_left", 9.0), ranges.get("front_right", 9.0))
        fl = min(ranges.get("front_left", 9.0), 9.0)
        fr = min(ranges.get("front_right", 9.0), 9.0)
        if front < cfg.front_threshold:
            self.retreat_left = cfg.retreat_duration
            self._evade_dir = 1.0 if fr < fl else -1.0
        elif fl < cfg.side_threshold and fl < fr:
            self.evade_left = cfg.evade_duration
            self._evade_dir = -1.0
        elif fr < cfg.side_threshold:
            self.evade_left = cfg.evade_duration
            self._evade_dir = 1.0
        active = self.retreat_left > 0 or self.evade_left > 0
        steer = 0.0
        if self.retreat_left > 0:
            steer = 0.8 * self._evade_dir
        elif self.evade_left > 0:
            steer = self._evade_dir
        return dict(active=active, retreat=self.retreat_left > 0, evade=self.evade_left > 0, steer=steer)
