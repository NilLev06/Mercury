import numpy as np
from ..config import RingConfig


class RingAttractor:
    def __init__(self, cfg: RingConfig):
        self.cfg = cfg
        self.N = cfg.n
        self.drive = np.zeros(self.N)
        n = self.N
        idx = np.arange(n)
        d = np.abs(idx[:, None] - idx[None, :])
        d = np.minimum(d, n - d)
        w = np.exp(-0.5 * (d / cfg.exc_sigma) ** 2)
        self.W = cfg.excit_gain * w - cfg.inhib_gain * np.ones((n, n))
        np.fill_diagonal(self.W, 0.0)
        self.r = np.ones(n) / n
        self.heading = 0.0

    def _gauss(self, center_deg: float, width_bins: float = 3.0) -> np.ndarray:
        rel = np.abs((np.arange(self.N) - center_deg / (360.0 / self.N)) % self.N)
        rel = np.minimum(rel, self.N - rel)
        return np.exp(-0.5 * (rel / width_bins) ** 2)

    def step(self, goal_bearing_deg: float, yaw_rate_deg: float, dt: float, masked: bool = False) -> float:
        if masked:
            drive = np.zeros(self.N)
        else:
            drive = self.cfg.input_gain * self._gauss(goal_bearing_deg)
        r_sum = self.r.sum() or 1e-9
        h = self.cfg.bias + self.W @ (self.r / r_sum)
        r_act = np.maximum(0.0, h)
        r_in = np.maximum(0.0, drive)
        self.r = (1.0 - self.cfg.leak) * r_act + r_in
        s = self.r.sum()
        if s > 0:
            self.r = self.r / s
        shift_bins = -int(round(yaw_rate_deg * dt * self.N / 360.0))
        if shift_bins % self.N != 0:
            self.r = np.roll(self.r, shift_bins % self.N)
        self.heading = self._circular_mean()
        return self.heading

    def _circular_mean(self) -> float:
        ang = (np.arange(self.N) * 2.0 * np.pi / self.N)
        angles = ang
        x = np.sum(self.r * np.cos(angles))
        y = np.sum(self.r * np.sin(angles))
        deg = np.degrees(np.arctan2(y, x))
        return deg