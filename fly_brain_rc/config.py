from dataclasses import dataclass


@dataclass
class RingConfig:
    n: int = 72
    exc_sigma: float = 2.2
    excit_gain: float = 1.6
    inhib_gain: float = 0.4
    input_gain: float = 1.3
    leak: float = 0.08
    bias: float = 0.02


@dataclass
class OpticFlowConfig:
    width: int = 64
    height: int = 32
    pyr_scale: float = 0.5
    levels: int = 3
    winsize: int = 15
    iterations: int = 3
    poly_n: int = 5
    poly_sigma: float = 1.1
    steer_gain: float = 1.6
    velocity_gain: float = 1.0


@dataclass
class LoomingConfig:
    tau_threshold: float = 1.0
    min_divergence: float = 0.6
    gmix: float = 0.4
    cooldown: float = 1.2


@dataclass
class ReflexConfig:
    front_threshold: float = 0.36
    side_threshold: float = 0.22
    retreat_duration: float = 0.8
    evade_duration: float = 0.5


@dataclass
class DriveConfig:
    base_throttle: float = 0.6
    goal_steer_gain: float = 1.6
    max_steer: float = 0.9
    max_steer_rad: float = 0.35
    max_speed: float = 0.35
    max_omega: float = 0.85
    slow_band: float = 0.6
    wheel_gain: float = 0.3
    wheel_friction: float = 0.6
    track: float = 0.36
    wheel_radius: float = 0.035
    timestep: float = 0.02


@dataclass
class SimConfig:
    arena_half: float = 3.0
    n_obstacles: int = 8
    obstacle_radius: float = 0.28
    obstacle_height: float = 0.5
    goal_radius: float = 0.25
    cam_width: int = 128
    cam_height: int = 96
    episode_steps: int = 1200
    kinematic: bool = True


DRIVE = DriveConfig()
RING = RingConfig()
FLOW = OpticFlowConfig()
LOOM = LoomingConfig()
REFLEX = ReflexConfig()
SIM = SimConfig()