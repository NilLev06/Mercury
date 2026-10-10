"""Temporary observation-only A/B. Never writes the user's policy."""
import sys, json, math, shutil, hashlib
from pathlib import Path
from contextlib import nullcontext
from unittest.mock import patch
sys.path.insert(0, '/Users/leonard/Desktop/DigitalFlyCar')
from fly_brain_rc.config import SIM, DRIVE
from fly_brain_rc.bypass_learning import train_bypass, bypass_cases
from fly_brain_rc.sim.env import FlyCarEnv
from fly_brain_rc.sim.scenarios import Scenario, get_scenario
from fly_brain_rc.sim.suites import SuiteCase
from fly_brain_rc.sim.kinematics import BicycleState
from fly_brain_rc.rl import QLearningAgent, QLearningBrain, RLController
from fly_brain_rc.contracts import MotionObservation
from fly_brain_rc.perception.target import MockRedBallDetector, MockBlueObstacleDetector

root = Path(__file__).parent
source = Path('/Users/leonard/Desktop/DigitalFlyCar/artifacts/rl/bypass-policy.json')
original_state = RLController._policy_state
original_values = QLearningAgent.values

def detailed_state(controller, sensor, target, obstacle):
    state = original_state(controller, sensor, target, obstacle)
    if controller.agent.task != 'bypass-v1':
        return state
    effective = target if target is not None else controller.navigation.recalled_target(sensor)
    if effective is None:
        detail = 'goal_none'
    else:
        b = effective.bearing
        detail = ('goal_far_left' if b < -.55 else 'goal_left' if b < -.18 else
                  'goal_center' if b <= .18 else 'goal_right' if b <= .55 else 'goal_far_right')
    return state + (detail,)

def detailed_values(agent, state):
    state = tuple(state)
    if agent.task == 'bypass-v1' and len(state) == 11:
        candidates = [state]
        if state[0].startswith('target_memory_'):
            candidates.append((state[0].replace('target_memory_', 'target_', 1),)+state[1:])
        for key in candidates:
            if key in agent.table:
                return agent.table[key]
        return original_values(agent, state[:10])
    return original_values(agent, state)

class DetailContext:
    def __enter__(self):
        self.a = patch.object(RLController, '_policy_state', detailed_state)
        self.b = patch.object(QLearningAgent, 'values', detailed_values)
        self.a.start(); self.b.start()
    def __exit__(self, *args):
        self.b.stop(); self.a.stop()

def cases():
    generated = bypass_cases(42, 8, held_out=True)
    fixed = (SuiteCase('straight', 0, Scenario('straight', (), (2.3, 0)), (0,0,0)),
             SuiteCase('single_cylinder', 0, get_scenario('single_cylinder'), (0,0,0)))
    return fixed + (generated[0], generated[3])

def evaluate(name, path, suite, detailed=False):
    results = []
    with DetailContext() if detailed else nullcontext():
        for case in suite:
            agent = QLearningAgent.load(path, epsilon=0)
            brain = QLearningBrain(agent)
            env = FlyCarEnv(SIM, DRIVE, seed=case.seed, scenario=case.scenario)
            env.state = BicycleState(*case.start_pose, 0)
            env._sync_kinematic()
            env._last_encoder_position = env._encoder_position()
            td, od = MockRedBallDetector(), MockBlueObstacleDetector()
            minimum, reverse, stalled, seen = 100., 0, 0, 0
            trace = []
            try:
                for tick in range(2400):
                    sensor = env.sensor_frame()
                    target = td.detect(sensor, 'red_ball')
                    obstacle = od.detect(sensor, 'obstacle')
                    command, _ = brain.step(sensor, target, MotionObservation(0,0,0,0,0,sensor.timestamp),
                                            0, DRIVE.timestep, obstacle=obstacle)
                    if tick % 10 == 0:
                        trace.append(dict(time=sensor.timestamp, state=brain.controller._policy_state(sensor,target,obstacle),
                                          bearing=None if target is None else target.bearing,
                                          area=None if target is None else target.apparent_area,
                                          traction=command.traction, steering=command.steering,
                                          speed=sensor.speed, front=sensor.tof, left=sensor.range_left,
                                          right=sensor.range_right, contact=sensor.contact,
                                          pose=env.car_pose()))
                    reverse += sensor.speed < -.025
                    stalled += abs(sensor.speed) <= .025
                    seen += target is not None
                    env.apply(command)
                    x,y,yaw = env.car_pose()
                    minimum = min(minimum, math.hypot(env.goal[0]-x-.09*math.cos(yaw),
                                                     env.goal[1]-y-.09*math.sin(yaw)))
                    if env.goal_center_reached():
                        break
                result = dict(policy=name, case=case.identifier, success=env.goal_center_reached(),
                              minimum_center=minimum, contacts=env.collisions, steps=tick+1,
                              reverse_seconds=reverse*.02, stalled_seconds=stalled*.02,
                              seen_fraction=seen/(tick+1))
                results.append(result)
                print('EVAL', json.dumps(result), flush=True)
                (root/f'{name}-{case.identifier}.trace.json').write_text(json.dumps(trace))
            finally:
                env.renderer.close()
    (root/f'{name}.evaluation.json').write_text(json.dumps(results, indent=2))
    return results

if __name__ == '__main__':
    fingerprint = hashlib.sha256(source.read_bytes()).hexdigest()
    suite = cases()
    train_cases = bypass_cases(73, 24)
    held_out = bypass_cases(73, 6, held_out=True)
    evaluate('before', source, suite + held_out)
    for name, detailed in (('control',False), ('bearing',True)):
        output = root/f'{name}.json'
        if output.exists():
            raise RuntimeError('refusing to overwrite experiment output')
        shutil.copy2(source, output)
        with DetailContext() if detailed else nullcontext():
            with patch('fly_brain_rc.bypass_learning.bypass_cases', return_value=train_cases):
                result = train_bypass(episodes=24, steps=2400, output=output, seed=42)
                print('TRAIN_SUMMARY', name, result, flush=True)
        evaluate(name, output, suite + held_out, detailed)
    after = hashlib.sha256(source.read_bytes()).hexdigest()
    if after != fingerprint:
        raise RuntimeError('source policy changed during experiment; investigate external activity')
    print('SOURCE_UNCHANGED', fingerprint, flush=True)
