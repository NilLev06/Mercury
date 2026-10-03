"""Q-learning training loop for the MuJoCo fly-car simulation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import shutil

from .config import DRIVE, SIM
from .contracts import ActuatorCommand, BrainTelemetry, MotionObservation, RuntimeTelemetry
from .perception.target import MockBlueObstacleDetector, MockRedBallDetector
from .rl import ACTION_DURATION, POLICY_VERSION, ExplorationReward, QLearningAgent, RLController, observation_state
from .sim.env import FlyCarEnv
from .sim.suites import training_cases, recovery_cases
from .sim.kinematics import BicycleState

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
        payload = json.loads(output.read_text())
        if payload.get('task','navigation') != 'navigation':
            raise ValueError('general training requires a navigation policy, not the bypass experiment')
        version = payload.get('version')
        if version != POLICY_VERSION:
            backup = output.with_name(output.name + f'.v{version}.bak')
            index = 1
            while backup.exists():
                backup = output.with_name(output.name + f'.v{version}.{index}.bak')
                index += 1
            shutil.copy2(output, backup)
            if version in (5,6):
                print(f'Old policy preserved: {backup}. Migrating compact observations.',flush=True)
                return QLearningAgent.load(output,epsilon=epsilon,seed=seed)
            print(f'Old policy preserved: {backup}. Starting new observation schema.', flush=True)
            return QLearningAgent(epsilon=epsilon, seed=seed)
        return QLearningAgent.load(output, epsilon=epsilon, seed=seed)
    return QLearningAgent(epsilon=epsilon, seed=seed)


def train_q_learning(*, episodes: int, steps: int, output: str | Path,
                     seed: int = 0, epsilon: float = 0.25, emit=None, progress=None, cancel=None) -> TrainingSummary:
    if type(episodes) is not int or episodes <= 0 or type(steps) is not int or steps <= 0:
        raise ValueError("episodes and steps must be positive integers")
    output = Path(output)
    agent = load_training_agent(output, seed=seed, epsilon=epsilon)
    output.parent.mkdir(parents=True, exist_ok=True)
    run_number = agent.training_runs + 1
    if output.exists():
        snapshot = output.with_name(f'{output.stem}.before-run-{run_number}.json')
        if not snapshot.exists():
            shutil.copy2(output, snapshot)
    run_seed = seed + agent.training_runs * 10_007
    agent._rng.seed(run_seed)
    target_detector, obstacle_detector = MockRedBallDetector(), MockBlueObstacleDetector()
    rewards, cells, successes = [], [], 0
    open_cases = training_cases(run_seed, episodes)
    occluded_cases = training_cases(run_seed + 1, episodes, occluded=True)
    recover_cases = recovery_cases(run_seed, episodes)

    for index in range(episodes):
        if cancel and cancel():
            break
        # Every launch expands the map experience.  Within a launch, advance
        # from near-contact recovery to pursuit, one occluder and clutter.
        phase = index * 4 // episodes
        if phase == 0:
            case = recover_cases[index]
            env = FlyCarEnv(SIM, DRIVE, seed=case.seed, scenario=case.scenario, render=True)
            env.state = BicycleState(*case.start_pose, 0.0)
            env.start_pose = case.start_pose
            env._sync_kinematic()
        elif phase == 1:
            case = open_cases[index]
            env = FlyCarEnv(SIM, DRIVE, seed=case.seed, scenario=case.scenario, render=True)
        elif phase == 2:
            case = occluded_cases[index]
            env = FlyCarEnv(SIM, DRIVE, seed=case.seed, scenario=case.scenario, render=True)
        else:
            env = FlyCarEnv(SIM, DRIVE, seed=run_seed + index, scenario=None, render=True)
        reward_model = ExplorationReward()
        controller = RLController(agent)
        reward_model.reset(env.car_pose()[:2])
        total = 0.0
        reached, sequence, cancelled = False,0,False
        applied_steps = 0
        contact_seconds = reverse_seconds = reverse_distance = stalled_seconds = 0.0
        try:
            sensor = env.sensor_frame()
            state = controller.observe(sensor, target_detector.detect(sensor, 'red_ball'),
                                      obstacle_detector.detect(sensor, 'obstacle'))
            remaining = 0
            window_reward = 0.0
            for sequence in range(1, steps + 1):
                if cancel and cancel():
                    cancelled = True
                    break
                if remaining == 0:
                    action = controller.choose(state, explore=True)
                    window_learnable = not controller.overriding
                    window_state = state
                    window_reward = 0.0
                    remaining = max(1, round(ACTION_DURATION / env.drive.timestep))
                traction, steering = agent.actions[action]
                env.apply(ActuatorCommand(traction, steering, sensor.timestamp, sequence))
                applied_steps += 1

                next_sensor = env.sensor_frame()
                next_target = target_detector.detect(next_sensor, "red_ball")
                next_obstacle = obstacle_detector.detect(next_sensor, "obstacle")
                old_mode = controller.mode
                next_state = controller.observe(next_sensor, next_target, next_obstacle)
                phase_changed = old_mode != controller.mode
                reached = env.goal_distance() <= env.sim_cfg.goal_radius
                moved = abs(next_sensor.encoder_delta) >= 0.001
                reward = reward_model.step(env.car_pose()[:2], moved=moved,
                                           contact=next_sensor.contact, reached=reached,
                                           target=next_target, speed=next_sensor.speed,
                                           dt=env.drive.timestep)
                contact_seconds += float(next_sensor.contact) * env.drive.timestep
                reverse_seconds += float(next_sensor.speed < -.025) * env.drive.timestep
                reverse_distance += max(0.0, -next_sensor.encoder_delta)
                stalled_seconds += float(not moved) * env.drive.timestep
                window_reward += reward
                remaining -= 1
                if remaining == 0 or reached or sequence == steps or phase_changed:
                    if window_learnable:
                        agent.update(window_state, action, window_reward, next_state,
                                     terminal=reached or controller.mode != 'drive')
                    remaining = 0
                sensor, state = next_sensor, next_state
                total += reward
                if emit is not None:
                    telemetry = BrainTelemetry(
                        populations={
                            'rl_episode_progress': (index + sequence / steps) / episodes,
                            'rl_exploration': min(1.0, len(reward_model.visited) / 20.0),
                            'rl_idle': min(1.0, reward_model.stationary_steps / 20.0),
                            'rl_target_memory': float(controller.navigation.vector is not None),
                            'rl_memory_confidence': controller.navigation.confidence,
                            'rl_bypass_active': float(controller.navigation.phase != 'drive'),
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
            if cancelled and remaining > 0 and applied_steps and window_learnable:
                agent.update(window_state,action,window_reward,state,terminal=controller.mode!='drive')
        finally:
            if env.renderer is not None:
                env.renderer.close()
        rewards.append(total)
        cells.append(len(reward_model.visited))
        report = dict(policy_version=POLICY_VERSION, run=run_number, episode=index + 1, phase=('recovery', 'open', 'occluded', 'clutter')[phase],
                      seed=case.seed if phase < 3 else run_seed + index,
                      success=bool(reached), steps=applied_steps, cancelled=cancelled,
                      duration=applied_steps * env.drive.timestep, reward=total,
                      explored_cells=len(reward_model.visited), contacts=env.collisions,
                      contact_seconds=contact_seconds, reverse_seconds=reverse_seconds,
                      reverse_distance=reverse_distance, stalled_seconds=stalled_seconds)
        with output.with_suffix('.history.jsonl').open('a') as history:
            history.write(json.dumps(report, allow_nan=False) + '\n')
        print(json.dumps(report,allow_nan=False),flush=True)
        # Gradually reduce random exploration while keeping occasional novelty.
        agent.epsilon = max(0.04, epsilon * (1.0 - (index + 1) / (episodes * 1.15)))
        agent.training_runs = run_number
        agent.save(output)
        if cancelled:
            break

    output.parent.mkdir(parents=True, exist_ok=True)
    return TrainingSummary(len(rewards), successes, sum(rewards) / len(rewards) if rewards else 0,
                           sum(cells) / len(cells) if cells else 0)
