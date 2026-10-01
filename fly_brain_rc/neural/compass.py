"""Compass population and encoder-based path integration."""

import math

import numpy as np


class CompassState:
    """Track heading and the return vector using only relative motion."""

    RING_SIZE = 72
    _BUMP_CONCENTRATION = 8.0
    _OPTIC_GAIN = 0.25

    def __init__(self):
        self._preferred = np.linspace(-math.pi, math.pi, self.RING_SIZE, endpoint=False)
        self.reset()

    @property
    def activity(self) -> np.ndarray:
        return self._activity.copy()

    @property
    def heading(self) -> float:
        x = float(np.dot(self._activity, np.cos(self._preferred)))
        y = float(np.dot(self._activity, np.sin(self._preferred)))
        return math.atan2(y, x)

    @property
    def home_vector(self) -> np.ndarray:
        """The estimated displacement from the current place back to reset origin."""
        return self._home_vector.copy()

    def reset(self, heading: float = 0.0) -> None:
        if not math.isfinite(heading):
            raise ValueError("initial heading must be finite")
        self._activity = self._encode(heading)
        self._home_vector = np.zeros(2, dtype=float)

    def update(self, yaw_rate: float, optic_rotation: float,
               encoder_delta: float, dt: float) -> None:
        if (not all(map(math.isfinite, (yaw_rate, optic_rotation, encoder_delta, dt)))
                or dt <= 0.0):
            raise ValueError("motion inputs must be finite and dt positive")
        optic_correction = self._OPTIC_GAIN * float(np.clip(optic_rotation, -1.0, 1.0))
        next_heading = self.heading + (yaw_rate + optic_correction) * dt
        if not math.isfinite(next_heading):
            raise ValueError("integrated heading must remain finite")
        next_activity = self._encode(next_heading)
        if not np.isfinite(next_activity).all():
            raise ValueError("compass activity must remain finite")
        decoded_heading = math.atan2(
            float(np.dot(next_activity, np.sin(self._preferred))),
            float(np.dot(next_activity, np.cos(self._preferred))),
        )
        with np.errstate(over="ignore", invalid="ignore"):
            next_home_vector = self._home_vector - encoder_delta * np.array(
                [math.cos(decoded_heading), math.sin(decoded_heading)]
            )
        if not np.isfinite(next_home_vector).all():
            raise ValueError("home vector must remain finite")
        self._home_vector = next_home_vector
        self._activity = next_activity

    def _encode(self, heading: float) -> np.ndarray:
        bump = np.exp(self._BUMP_CONCENTRATION *
                      (np.cos(self._preferred - heading) - 1.0))
        return bump / bump.sum()
