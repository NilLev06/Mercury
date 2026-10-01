"""Replaceable visual perception producing backend-neutral observations."""

from .familiarity import VisualDescriptor
from .motion import MotionEstimator
from .target import MockRedBallDetector, TargetDetector

__all__ = ["MockRedBallDetector", "MotionEstimator", "TargetDetector", "VisualDescriptor"]
