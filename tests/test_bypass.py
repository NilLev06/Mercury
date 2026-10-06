import tempfile
import unittest
from pathlib import Path
from dataclasses import replace
import numpy as np
from fly_brain_rc.contracts import ObjectObservation, SensorFrame
from fly_brain_rc.rl import QLearningAgent, RLController


class BypassTests(unittest.TestCase):
    def test_task_tag_survives_save_and_load(self):
        a = QLearningAgent()
        a.task = 'bypass-v1'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'policy.json'
            a.save(path)
            self.assertEqual(QLearningAgent.load(path).task,'bypass-v1')

    def test_bypass_policy_drives_without_scripted_arc_or_retreat(self):
        a = QLearningAgent()
        a.task = 'bypass-v1'
        c = RLController(a)
        s = SensorFrame(np.zeros((2,2,3),dtype=np.uint8),np.zeros((2,2)),0,0,.2,False,0)
        obstacle = ObjectObservation('obstacle',1,(.3,.3,.7,.8),0,.1,0)
        state = c.observe(s,None,obstacle)
        a.table[state] = [0,9]+[0]*9
        action = c.choose(state,explore=False)
        self.assertEqual(action,1)
        self.assertFalse(c.overriding)
        state = c.observe(replace(s,timestamp=.02,contact=True,contact_front=True),None,obstacle)
        action = c.choose(state,explore=False)
        self.assertLess(a.actions[action][0],0)
        self.assertFalse(c.overriding)

    def test_seeded_single_cylinder_cases_and_holdout_are_separate(self):
        from fly_brain_rc.bypass_learning import bypass_cases
        train = bypass_cases(0,12)
        self.assertEqual(train,bypass_cases(0,12))
        self.assertTrue(all(len(c.scenario.obstacles)==1 for c in train))
        self.assertGreater(len({c.start_pose for c in train}),1)
        self.assertTrue(set(c.scenario for c in train).isdisjoint(c.scenario for c in bypass_cases(0,12,held_out=True)))

    def test_reward_prefers_progress_and_penalizes_sustained_contact(self):
        from fly_brain_rc.bypass_learning import BypassReward
        r = BypassReward(2)
        progress = r.step(1.8,contact=False,steering=0,reached=False,dt=.2)
        stall = r.step(1.8,contact=False,steering=0,reached=False,dt=.2)
        bump = r.step(1.8,contact=True,steering=0,reached=False,dt=.2)
        held = r.step(1.8,contact=True,steering=0,reached=False,dt=.2)
        self.assertGreater(progress,stall)
        self.assertLess(bump,stall)
        self.assertLess(held,stall)
        self.assertGreater(r.step(.4,contact=False,steering=0,reached=True,dt=.2),20)

    def test_cancelled_training_checkpoints_and_evaluation_does_not_rewrite(self):
        from unittest.mock import patch
        from tests.test_environment import HeadlessRenderer
        from fly_brain_rc.bypass_learning import train_bypass
        with tempfile.TemporaryDirectory() as directory, patch('fly_brain_rc.sim.env.mujoco.Renderer',HeadlessRenderer):
            path = Path(directory)/'bypass.json'
            ticks = []
            summary = train_bypass(episodes=2,steps=30,output=path,
                                   emit=ticks.append,cancel=lambda:len(ticks)>=1)
            self.assertTrue(summary.cancelled)
            self.assertTrue(QLearningAgent.load(path).table)
            before = path.read_bytes()
            train_bypass(episodes=1,steps=10,output=path,evaluate=True)
            self.assertEqual(path.read_bytes(),before)

    def test_viewer_bypass_reach_finishes_without_claiming_learned_stop(self):
        from unittest.mock import patch
        from tests.test_environment import HeadlessRenderer
        from fly_brain_rc.config import SIM, DRIVE
        from fly_brain_rc.sim.env import FlyCarEnv
        from fly_brain_rc.sim.scenarios import Scenario
        from fly_brain_rc.runtime import FlyRuntime
        from fly_brain_rc.rl import QLearningBrain
        from fly_brain_rc.dashboard.server import MissionController
        a = QLearningAgent()
        a.task = 'bypass-v1'
        with patch('fly_brain_rc.sim.env.mujoco.Renderer',HeadlessRenderer):
            env = FlyCarEnv(SIM,DRIVE,scenario=Scenario('reach',(),(.09,0)))
            from fly_brain_rc.sim.kinematics import BicycleState
            env.state = BicycleState(0,0,0,.2,.1)
            env._sync_kinematic()
            mission = MissionController(running=True)
            runtime = FlyRuntime(env,QLearningBrain(a),mission=mission,bypass_evaluation=True)
            record = runtime.tick()
            self.assertEqual(mission.state()['status'],'finished')
            self.assertTrue(mission.state()['result']['goal_reached'])
            self.assertFalse(mission.state()['result']['stable_arrival'])
            self.assertEqual(record.command.traction,0)
            self.assertEqual(env.state.speed,0)
            self.assertEqual(env.state.x,0)
            env.renderer.close()

    def test_viewer_keeps_driving_when_only_near_the_ball(self):
        from unittest.mock import patch
        from tests.test_environment import HeadlessRenderer
        from fly_brain_rc.config import SIM, DRIVE
        from fly_brain_rc.sim.env import FlyCarEnv
        from fly_brain_rc.sim.scenarios import Scenario
        from fly_brain_rc.runtime import FlyRuntime
        from fly_brain_rc.rl import QLearningBrain
        from fly_brain_rc.dashboard.server import MissionController
        a = QLearningAgent()
        a.task = 'bypass-v1'
        with patch('fly_brain_rc.sim.env.mujoco.Renderer',HeadlessRenderer):
            env = FlyCarEnv(SIM,DRIVE,scenario=Scenario('near',(),(.45,0)))
            mission = MissionController(running=True)
            record = FlyRuntime(env,QLearningBrain(a),mission=mission,bypass_evaluation=True).tick()
            self.assertEqual(mission.state()['status'],'running')
            self.assertNotEqual(record.command.traction,0)
            env.renderer.close()

    def test_training_scores_center_entry_not_old_half_meter_zone(self):
        from unittest.mock import patch
        from tests.test_environment import HeadlessRenderer
        from fly_brain_rc.bypass_learning import train_bypass
        from fly_brain_rc.sim.suites import SuiteCase
        from fly_brain_rc.sim.scenarios import Scenario
        with tempfile.TemporaryDirectory() as directory, patch('fly_brain_rc.sim.env.mujoco.Renderer',HeadlessRenderer):
            for x,want in ((.45,0),(.09,1)):
                case = SuiteCase('center',0,Scenario('center',(),(x,0)),(0,0,0))
                with patch('fly_brain_rc.bypass_learning.bypass_cases',return_value=(case,)):
                    result = train_bypass(episodes=1,steps=1,epsilon=0,output=Path(directory)/f'{x}.json')
                self.assertEqual(result.successes,want)
