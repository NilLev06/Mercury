"""Typed data exchanged by sensing, perception, the brain, and actuation."""

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class SensorFrame:
    rgb: np.ndarray
    gray: np.ndarray
    yaw_rate: float
    encoder_delta: float
    tof: float
    contact: bool
    timestamp: float
    contact_left: bool = False
    contact_right: bool = False
    speed: float = 0.0
    range_left: float = 1.2
    range_right: float = 1.2
    contact_front: bool = False
    contact_rear: bool = False


@dataclass(frozen=True)
class ObjectObservation:
    class_name: str
    confidence: float
    bbox: tuple[float, float, float, float]
    bearing: float
    apparent_area: float
    timestamp: float


@dataclass(frozen=True)
class MotionObservation:
    flow_left: float
    flow_center: float
    flow_right: float
    optic_rotation: float
    looming: float
    timestamp: float


@dataclass(frozen=True)
class ActuatorCommand:
    traction: float
    steering: float
    timestamp: float
    sequence: int

    def __post_init__(self):
        object.__setattr__(self, "traction", max(-1.0, min(1.0, self.traction)))
        object.__setattr__(self, "steering", max(-1.0, min(1.0, self.steering)))


@dataclass(frozen=True)
class BrainTelemetry:
    populations: Mapping[str, float] = field(default_factory=dict)
    compass_ring: tuple[float, ...] = ()
    target_memory: tuple[float, ...] = ()
    path_integration: tuple[float, float] = (0.0, 0.0)
    familiarity: float = 0.0
    habituation: float = 0.0
    threat: float = 0.0
    motor_left: float = 0.0
    motor_right: float = 0.0
    motor_forward: float = 0.0
    motor_reverse: float = 0.0
    traction: float = 0.0
    steering: float = 0.0


@dataclass(frozen=True)
class RuntimeTelemetry:
    timestamp: float
    sequence: int
    sensor: SensorFrame
    target: ObjectObservation | None
    motion: MotionObservation
    brain: BrainTelemetry
    command: ActuatorCommand
