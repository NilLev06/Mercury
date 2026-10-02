"""Q-learning training loop for the MuJoCo fly-car simulation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import DRIVE, SIM
from .contracts import ActuatorCommand
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


def train_q_learning(*, episodes: int, steps: int, output: str | Path,
                     seed: int = 0, epsilon: float = 0.25) -> TrainingSummary:
    if type(episodes) is not int or episodes <= 0 or type(steps) is not int or steps <= 0:
        raise ValueError("episodes and steps must be positive integers")
    agent = QLearningAgent(epsilon=epsilon, seed=seed)
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
                                           contact=next_sensor.contact, reached=reached)
                agent.update(state, action, reward, next_state, terminal=reached)
                total += reward
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

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    agent.save(output)
    return TrainingSummary(episodes, successes, sum(rewards) / episodes, sum(cells) / episodes)
