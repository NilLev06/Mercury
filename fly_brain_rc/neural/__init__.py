"""Rate-based neural dynamics and embodied navigation state."""

from .compass import CompassState
from .dynamics import RateLayer

__all__ = ["CompassState", "RateLayer"]
