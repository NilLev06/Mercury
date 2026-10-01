"""Versioned magnitudes for a fixed, signed neural topology."""

from dataclasses import dataclass
import json
from types import MappingProxyType

import numpy as np


@dataclass(frozen=True)
class FlyGenome:
    tau: float = 0.12
    sensory_gain: float = 6.0
    target_gain: float = 12.0
    threat_gain: float = 12.0
    exploration_gain: float = 6.0
    exploration_turn_gain: float = 15.0
    looming_gain: float = 0.2
    memory_decay: float = 0.8
    habituation_rate: float = 0.5
    visit_decay: float = 0.03
    proximity_distance: float = 0.35
    arrival_area: float = 0.15
    VERSION = 2
    BOUNDS = MappingProxyType({
        'tau': (0.02, 2.0), 'sensory_gain': (0.1, 12.0),
        'target_gain': (0.1, 12.0), 'threat_gain': (6.0, 20.0),
        'exploration_gain': (0.1, 6.0), 'memory_decay': (0.01, 5.0),
        'exploration_turn_gain': (1.0, 30.0),
        'looming_gain': (0.05, 1.0),
        'habituation_rate': (0.01, 3.0), 'visit_decay': (0.001, 1.0),
        'proximity_distance': (0.1, 1.0), 'arrival_area': (0.05, 0.3),
    })

    def __post_init__(self):
        for name, (lower, upper) in self.BOUNDS.items():
            value = getattr(self, name)
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number))
                    or not np.isfinite(value) or not lower <= value <= upper):
                raise ValueError(f'{name} must be finite and in [{lower}, {upper}]')

    @classmethod
    def default(cls):
        return cls()

    def vector(self):
        return np.array([getattr(self, name) for name in self.BOUNDS], dtype=float)

    def with_vector(self, vector):
        try:
            values = np.asarray(vector, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError('invalid genome vector') from exc
        if values.shape != (len(self.BOUNDS),):
            raise ValueError('genome vector has incorrect shape')
        return type(self)(**dict(zip(self.BOUNDS, values)))

    def to_json(self):
        return json.dumps({'version': self.VERSION, 'parameters': {
            name: float(getattr(self, name)) for name in self.BOUNDS}}, allow_nan=False, sort_keys=True)

    @classmethod
    def from_json(cls, source):
        try:
            payload = json.loads(source)
        except (TypeError, ValueError) as exc:
            raise ValueError('invalid genome JSON') from exc
        if (not isinstance(payload, dict) or set(payload) != {'version', 'parameters'}
                or type(payload['version']) is not int or payload['version'] != cls.VERSION):
            raise ValueError('unsupported genome schema or version')
        parameters = payload['parameters']
        if not isinstance(parameters, dict) or set(parameters) != set(cls.BOUNDS):
            raise ValueError('invalid genome parameters')
        return cls(**parameters)
