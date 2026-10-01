"""Sensor-only runtime; ground truth is confined to episode evaluation."""
from dataclasses import dataclass
from pathlib import Path
import math

from .contracts import RuntimeTelemetry
from .memory import FamiliarityMemory
from .neural.brain import NeuralFlyBrain, ArrivalTracker
from .neural.genome import FlyGenome
from .perception.target import MockRedBallDetector
from .perception.motion import MotionEstimator
from .perception.familiarity import VisualDescriptor


def load_genome(path=None):
    return FlyGenome.from_json(Path(path).read_text()) if path else FlyGenome.default()


class FlyRuntime:
    def __init__(self, env, brain=None, dt=None, target='red_ball', *, detector=None,
                 motion=None, descriptor=None, memory=None, emit=None):
        self.env = env
        self.brain = brain if brain is not None else NeuralFlyBrain()
        self.dt = dt if dt is not None else env.drive.timestep
        if not math.isfinite(self.dt) or self.dt <= 0:
            raise ValueError('dt must be finite and positive')
        if target != 'red_ball':
            raise ValueError('unsupported target')
        self.target = target
        self.detector = detector if detector is not None else MockRedBallDetector()
        self.motion = motion if motion is not None else MotionEstimator()
        self.descriptor = descriptor if descriptor is not None else VisualDescriptor()
        self.memory = memory if memory is not None else FamiliarityMemory()
        self.emit = emit

    def tick(self):
        sensor = self.env.sensor_frame()
        target = self.detector.detect(sensor, self.target)
        motion = self.motion.update(sensor, self.dt)
        familiarity = self.memory.observe(self.descriptor.encode(sensor.gray), sensor.timestamp)
        command, brain = self.brain.step(sensor, target, motion, familiarity, self.dt)
        self.env.apply(command)
        record = RuntimeTelemetry(sensor.timestamp, command.sequence, sensor, target, motion, brain, command)
        if self.emit is not None:
            self.emit(record)
        return record


@dataclass(frozen=True)
class EpisodeResult:
    success: bool
    steps: int
    acquisition_time: float | None
    arrival_time: float | None
    stable_stop_time: float | None
    collisions: int
    penetrations: int
    reverse_distance: float
    steering_variation: float
    familiarity_novelty: float
    min_goal_distance: float


def run_episode(runtime, steps=1000, *, arrival=None, on_tick=None):
    if type(steps) is not int or steps <= 0:
        raise ValueError('steps must be positive')
    tracker = arrival if arrival is not None else ArrivalTracker()
    acquisition = reached = stopped = None
    reverse = variation = novelty = 0.0
    previous_steering = None
    env = runtime.env
    minimum = env.goal_distance()
    for index in range(steps):
        record = runtime.tick()
        now = record.timestamp
        if record.target is not None and acquisition is None:
            acquisition = now
        distance = env.goal_distance()
        minimum = min(minimum, distance)
        if distance < env.sim_cfg.goal_radius and reached is None:
            reached = now + runtime.dt
        reverse += max(0.0, -record.sensor.encoder_delta)
        if previous_steering is not None:
            variation += abs(record.command.steering - previous_steering)
        previous_steering = record.command.steering
        novelty += 1 - record.brain.familiarity
        # Arrival uses the image and odometry from the same pre-actuation instant.
        stable = tracker.update(record.target, record.command,
                                record.sensor.encoder_delta / runtime.dt,
                                record.sensor.tof, runtime.dt)
        if on_tick is not None:
            on_tick(record)
        if stable:
            stopped = now
            break
    success = stopped is not None and reached is not None and env.collisions == 0 and env.penetrations == 0
    return EpisodeResult(success, index + 1,
                         acquisition, reached, stopped, env.collisions, env.penetrations,
                         reverse, variation, novelty / (index + 1), minimum)
