"""Isolated single-cylinder experiment. Coordinates are trainer-only rewards."""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import shutil

from .config import DRIVE, SIM
from .contracts import ActuatorCommand, BrainTelemetry, MotionObservation, RuntimeTelemetry
from .perception.target import MockBlueObstacleDetector, MockRedBallDetector
from .rl import ACTION_DURATION, QLearningAgent, RLController
from .sim.env import FlyCarEnv
from .sim.kinematics import BicycleState
from .sim.scenarios import BLUE, Scenario
from .sim.suites import SuiteCase
from .docking import DockingTracker, prepare_docking, target_view, ranges


def bypass_cases(seed, count, *, held_out=False):
    if type(count) is not int or count < 0:
        raise ValueError('count must be nonnegative')
    base = seed + (1_000_000 if held_out else 0)
    rng = random.Random(base)
    result = []
    for i in range(count):
        # Offset cases first, then increasingly occluded paths. Random starts
        # ensure the learner cannot memorize one steering sequence.
        offset = rng.choice((-1,1))*rng.uniform(.35,.65) if i < count//4 else rng.uniform(-.4,.4)
        name = f'bypass-{"test" if held_out else "train"}-{base}-{i}'
        scenario = Scenario(name,(((rng.uniform(.9,1.35),offset),BLUE),),
                            (rng.uniform(2.15,2.55),rng.uniform(-.3,.3)))
        result.append(SuiteCase(name,base+i,scenario,
                               (rng.uniform(-.15,.05),rng.uniform(-.25,.25),rng.uniform(-.18,.18))))
    return tuple(result)


class BypassReward:
    def __init__(self, distance):
        self.distance = distance
        self.contact = False
        self.steering = 0.0
        self.risk = None
        self.alignment = None

    def step(self, distance, *, contact, steering, reached, dt,
             sensor=None,target=None,command=None):
        # Dense trainer-only distance potential: a detour can cost temporarily,
        # but progress and a successful pass outweigh time/contact costs.
        reward = 8*(max(0,self.distance-.70)-max(0,distance-.70))-.15*dt
        reward -= 4*max(0,.60-distance)*dt
        if sensor is not None:
            view = target_view(sensor,target)
            # Target rays still have a hard stand-off: exempt only the safe
            # approach band, never contact or driving through the sphere.
            risk = sum(max(0,(.12 if i<3 and view.rays[i] else .20)-gap)
                       for i,gap in enumerate(ranges(sensor)))
            if self.risk is not None:
                reward -= 5*max(0,risk-self.risk)
            self.risk = risk
            alignment = abs(target.bearing) if target is not None and view.stage != 'far' else None
            if alignment is not None and self.alignment is not None:
                reward += .5*(self.alignment-alignment)
            self.alignment = alignment
            parked = (view.parking and abs(sensor.speed)<=.025 and command is not None
                      and abs(command.traction)<=.05 and abs(command.steering)<=.05)
            if parked:
                reward += 2*dt
            elif abs(sensor.speed)<.025:
                reward -= .5*dt
        if contact:
            reward -= .8*dt + (2 if not self.contact else 0)
        reward -= .03*abs(steering-self.steering)
        if reached:
            reward += 40
        self.distance,self.contact,self.steering = distance,contact,steering
        return reward


@dataclass(frozen=True)
class BypassSummary:
    episodes: int
    successes: int
    mean_reward: float
    cancelled: bool
    evaluation: bool


def train_bypass(*, episodes, steps, output, seed=0, epsilon=.3,
                 emit=None, progress=None, cancel=None, evaluate=False):
    if type(episodes) is not int or episodes <= 0 or type(steps) is not int or steps <= 0:
        raise ValueError('episodes and steps must be positive')
    output = Path(output)
    if output.exists():
        agent = QLearningAgent.load(output,epsilon=0 if evaluate else epsilon,seed=seed)
        if agent.task != 'bypass-v1':
            raise ValueError('expected a separate bypass-v1 policy; general policy is not overwritten')
    elif evaluate:
        raise ValueError('train the bypass policy first')
    else:
        agent = QLearningAgent(epsilon=epsilon,seed=seed)
        agent.task = 'bypass-v1'
    output.parent.mkdir(parents=True,exist_ok=True)
    run = agent.training_runs+1
    if not evaluate and output.exists():
        backup = output.with_name(f'{output.stem}.before-run-{run}.json')
        if not backup.exists():
            shutil.copy2(output,backup)
    cases = bypass_cases(seed if evaluate else seed+agent.training_runs*10_007,
                         episodes,held_out=evaluate)
    prepare_docking(agent)
    detector, obstacles = MockRedBallDetector(),MockBlueObstacleDetector()
    totals,successes = [],0
    cancelled = False
    for index,case in enumerate(cases):
        if cancel and cancel():
            cancelled = True
            break
        env = FlyCarEnv(SIM,DRIVE,seed=case.seed,scenario=case.scenario,render=True)
        env.state = BicycleState(*case.start_pose,0)
        env.start_pose = case.start_pose
        env._sync_kinematic()
        env._last_encoder_position = env._encoder_position()
        controller = RLController(agent)
        arrival = DockingTracker()
        reward_model = BypassReward(env.goal_distance())
        total,contacts,remaining = 0.0,0,0
        reached = False
        sequence = applied_steps = 0
        try:
            sensor = env.sensor_frame()
            state = controller.observe(sensor,detector.detect(sensor,'red_ball'),obstacles.detect(sensor,'obstacle'))
            for sequence in range(1,steps+1):
                if cancel and cancel():
                    cancelled = True
                    break
                if remaining == 0:
                    action = controller.choose(state,explore=not evaluate)
                    window_state,window_reward = state,0.0
                    remaining = max(1,round(ACTION_DURATION/DRIVE.timestep))
                traction,steering = agent.actions[action]
                command = ActuatorCommand(traction,steering,sensor.timestamp,sequence)
                # Same pre-application sensor/command predicate as the viewer.
                reached = arrival.update(sensor,detector.detect(sensor,'red_ball'),command,DRIVE.timestep)
                env.apply(command)
                applied_steps += 1
                next_sensor = env.sensor_frame()
                target = detector.detect(next_sensor,'red_ball')
                next_state = controller.observe(next_sensor,target,obstacles.detect(next_sensor,'obstacle'))
                reward = reward_model.step(env.goal_distance(),contact=next_sensor.contact,
                                           steering=steering,reached=reached,dt=DRIVE.timestep,
                                           sensor=next_sensor,target=target,command=command)
                total += reward
                contacts += int(next_sensor.contact)
                window_reward += reward
                remaining -= 1
                unsafe = action not in agent.admissible_actions(next_state)
                if remaining == 0 or reached or sequence == steps or unsafe:
                    if not evaluate:
                        agent.update(window_state,action,window_reward,next_state,terminal=reached or sequence==steps)
                    remaining = 0
                sensor,state = next_sensor,next_state
                if emit:
                    brain = BrainTelemetry(populations={'rl_memory_confidence':controller.navigation.confidence},
                                           traction=traction,steering=steering,
                                           motor_forward=max(0,traction),motor_reverse=max(0,-traction),
                                           motor_left=max(0,-steering),motor_right=max(0,steering),threat=float(sensor.contact))
                    emit(RuntimeTelemetry(sensor.timestamp,sequence,sensor,target,
                                          MotionObservation(0,0,0,0,0,sensor.timestamp),brain,
                                          ActuatorCommand(traction,steering,sensor.timestamp,sequence)))
                if progress:
                    progress(index+1,episodes,total,0)
                if reached:
                    break
            if cancelled and remaining > 0 and applied_steps and not evaluate:
                agent.update(window_state,action,window_reward,state,terminal=False)
        finally:
            env.renderer.close()
        totals.append(total)
        successes += int(reached)
        report = dict(task='bypass-v1',sensor_schema=3,arrival_criterion='stable-docking-v1',
                      stable_stop=arrival.elapsed,run=run,episode=index+1,seed=case.seed,
                      evaluation=evaluate,success=reached,cancelled=cancelled,
                      steps=applied_steps,reward=total,contacts=env.collisions,
                      contact_seconds=contacts*DRIVE.timestep,minimum_final_distance=env.goal_distance())
        suffix = '.evaluation.jsonl' if evaluate else '.history.jsonl'
        with output.with_suffix(suffix).open('a') as handle:
            handle.write(json.dumps(report,allow_nan=False)+'\n')
        if not evaluate:
            agent.training_runs = run
            agent.epsilon = max(.05,epsilon*(1-(index+1)/(episodes*1.15)))
            agent.save(output)
        print(json.dumps(report,allow_nan=False),flush=True)
        if cancelled:
            break
    return BypassSummary(len(totals),successes,sum(totals)/len(totals) if totals else 0,cancelled,evaluate)
