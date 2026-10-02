"""Q-learning training loop for the MuJoCo fly-car simulation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import DRIVE, SIM
from .contracts import ActuatorCommand, BrainTelemetry, MotionObservation, RuntimeTelemetry
from .perception.target import MockBlueObstacleDetector, MockRedBallDetector
from .rl import ExplorationReward, QLearningAgent, observation_state
from .sim.env import FlyCarEnv
from .sim.suites import training_cases

@dataclass(frozen=True)
class TrainingSummary:
    episodes: int
    successes: int
    mean_reward: float
    explored_cells: float


def load_training_agent(output: str | Path, *, seed: int, epsilon: float) -> QLearningAgent:
    """Resume the persisted policy when it exists; otherwise create its seed."""
    output = Path(output)
    if output.exists():
        return QLearningAgent.load(output, epsilon=epsilon)
    return QLearningAgent(epsilon=epsilon, seed=seed)


def train_q_learning(*, episodes: int, steps: int, output: str | Path,
                     seed: int = 0, epsilon: float = 0.25, emit=None, progress=None) -> TrainingSummary:
    if type(episodes) is not int or episodes <= 0 or type(steps) is not int or steps <= 0:
        raise ValueError("episodes and steps must be positive integers")
    output = Path(output)
    agent = load_training_agent(output, seed=seed, epsilon=epsilon)
    target_detector, obstacle_detector = MockRedBallDetector(), MockBlueObstacleDetector()
    rewards, cells, successes = [], [], 0
    cases = training_cases(seed, episodes, occluded=True)

    for index, case in enumerate(cases):
        env = FlyCarEnv(SIM, DRIVE, seed=case.seed, scenario=case.scenario, render=True)
        reward_model = ExplorationReward()
        reward_model.reset(env.car_pose()[:2])
        total = 0.0
        try:
            for sequence in range(1, steps + 1):
                sensor = env.sensor_frame()
                target = target_detector.detect(sensor, "red_ball")
                obstacle = obstacle_detector.detect(sensor, "obstacle")
                state = observation_state(sensor, target, obstacle)
                action = agent.choose(state)
                traction, steering = agent.actions[action]
                env.apply(ActuatorCommand(traction, steering, sensor.timestamp, sequence))

                next_sensor = env.sensor_frame()
                next_target = target_detector.detect(next_sensor, "red_ball")
                next_obstacle = obstacle_detector.detect(next_sensor, "obstacle")
                next_state = observation_state(next_sensor, next_target, next_obstacle)
                reached = env.goal_distance() <= env.sim_cfg.goal_radius
                moved = abs(next_sensor.encoder_delta) >= 0.001
                reward = reward_model.step(env.car_pose()[:2], moved=moved,
                                           contact=next_sensor.contact, reached=reached,
                                           target=next_target)
                agent.update(state, action, reward, next_state, terminal=reached)
                total += reward
                if emit is not None:
                    telemetry = BrainTelemetry(
                        populations={
                            'rl_episode_progress': (index + sequence / steps) / episodes,
                            'rl_exploration': min(1.0, len(reward_model.visited) / 20.0),
                            'rl_idle': min(1.0, reward_model.stationary_steps / 20.0),
                        },
                        threat=float(next_sensor.contact),
                        motor_left=max(0.0, -steering), motor_right=max(0.0, steering),
                        motor_forward=max(0.0, traction), motor_reverse=max(0.0, -traction),
                        traction=traction, steering=steering,
                    )
                    emit(RuntimeTelemetry(next_sensor.timestamp, sequence, next_sensor, next_target,
                                          MotionObservation(0.0, 0.0, 0.0, 0.0, 0.0, next_sensor.timestamp),
                                          telemetry, ActuatorCommand(traction, steering, next_sensor.timestamp, sequence)))
                if progress is not None:
                    progress(index + 1, episodes, total, len(reward_model.visited))
                if reached:
                    successes += 1
                    break
        finally:
            if env.renderer is not None:
                env.renderer.close()
        rewards.append(total)
        cells.append(len(reward_model.visited))
        # Gradually reduce random exploration while keeping occasional novelty.
        agent.epsilon = max(0.04, epsilon * (1.0 - (index + 1) / (episodes * 1.15)))

    output.parent.mkdir(parents=True, exist_ok=True)
    agent.save(output)
    return TrainingSummary(episodes, successes, sum(rewards) / episodes, sum(cells) / episodes)
