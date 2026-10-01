"""Simultaneous leaky updates for populations of rate neurons."""

import numpy as np


class RateLayer:
    """Integrate ``tau * da/dt = -a + sigmoid(Wa + x + bias)``."""

    def __init__(self, tau: float | np.ndarray = 1.0,
                 bias: float | np.ndarray = 0.0):
        self.tau = np.asarray(tau, dtype=float)
        self.bias = np.asarray(bias, dtype=float)
        if (self.tau.ndim > 1 or not np.isfinite(self.tau).all()
                or np.any(self.tau <= 0.0)):
            raise ValueError("tau must contain finite positive values")
        if self.bias.ndim > 1 or not np.isfinite(self.bias).all():
            raise ValueError("bias must contain finite values")

    def step(self, previous: np.ndarray, recurrent: np.ndarray,
             external: np.ndarray, dt: float) -> np.ndarray:
        """Return all next activities, computed only from ``previous``."""
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        previous = np.asarray(previous, dtype=float)
        recurrent = np.asarray(recurrent, dtype=float)
        external = np.asarray(external, dtype=float)
        if previous.ndim != 1:
            raise ValueError("previous must be a vector")
        count = previous.size
        if (recurrent.shape != (count, count) or external.shape != (count,)
                or self.tau.shape not in ((), (count,))
                or self.bias.shape not in ((), (count,))):
            raise ValueError("neural inputs and parameters must have matching shapes")
        if not (np.isfinite(previous).all() and np.isfinite(recurrent).all()
                and np.isfinite(external).all()):
            raise ValueError("neural inputs must contain finite values")

        drive = recurrent @ previous + external + self.bias
        sigmoid = np.empty_like(drive)
        positive = drive >= 0.0
        sigmoid[positive] = 1.0 / (1.0 + np.exp(-drive[positive]))
        exp_drive = np.exp(drive[~positive])
        sigmoid[~positive] = exp_drive / (1.0 + exp_drive)
        next_activity = previous + (dt / self.tau) * (sigmoid - previous)
        return np.clip(next_activity, 0.0, 1.0)
