import tempfile
import unittest
from pathlib import Path
from dataclasses import replace
import numpy as np

from fly_brain_rc.contracts import SensorFrame, ObjectObservation, ActuatorCommand
from fly_brain_rc.rl import QLearningAgent, RLController


def frame(**kwargs):
    return replace(SensorFrame(np.zeros((96,128,3),dtype=np.uint8),np.zeros((96,128)),
                               0,0,.266,False,0),**kwargs)


def ball():
    # Known 22 cm sphere, camera 12.6 cm ahead, centre 70 cm from rear axle.
    return ObjectObservation('red_ball',.9,(.297,.3,.703,.7),0,.12,0)


class DockingTests(unittest.TestCase):
    def test_visual_depth_and_center_ray_match_known_ball(self):
        from fly_brain_rc.docking import target_view
        view = target_view(frame(),ball())
        self.assertAlmostEqual(view.distance,.7,delta=.02)
        self.assertTrue(view.rays[1])
        self.assertTrue(view.parking)

    def test_visible_ball_does_not_excuse_nearer_occluder_or_side_hazard(self):
        from fly_brain_rc.docking import target_view
        view = target_view(frame(tof=.02),ball())
        self.assertFalse(view.rays[1])
        self.assertFalse(view.parking)
        self.assertFalse(target_view(frame(range_side_left=.01),ball()).parking)

    def test_moderately_nearer_occluder_is_not_mistaken_for_target(self):
        from fly_brain_rc.docking import target_view
        target = replace(ball(),bbox=(.251,.2,.749,.8))  # ball centre ~.61 m
        view = target_view(frame(tof=.12),target)
        self.assertFalse(view.rays[1])
        self.assertFalse(view.parking)

    def test_image_right_maps_to_positive_car_lateral_diagonal(self):
        from fly_brain_rc.docking import target_view
        # Wide camera: sphere centre (.8,+.6), matching +45 degree ray .52 m.
        sensor = frame(rgb=np.zeros((96,256,3),dtype=np.uint8),tof=1.2,range_left=.52)
        target = ObjectObservation('red_ball',.9,(.628,.3,.859,.6),.487,.04,0)
        self.assertEqual(target_view(sensor,target).rays,(True,False,False))

    def test_stable_arrival_requires_stop_and_resets_on_contact(self):
        from fly_brain_rc.docking import DockingTracker
        tracker = DockingTracker()
        stopped = ActuatorCommand(0,0,0,0)
        for _ in range(4):
            self.assertFalse(tracker.update(frame(),ball(),stopped,.2))
        self.assertTrue(tracker.update(frame(),ball(),stopped,.2))
        self.assertFalse(tracker.update(frame(contact=True),ball(),stopped,.2))
        self.assertEqual(tracker.elapsed,0)
        self.assertFalse(tracker.update(frame(),ball(),ActuatorCommand(.3,0,0,0),1))
        self.assertFalse(tracker.update(frame(),replace(ball(),bearing=.8),stopped,1))

    def test_upgrade_keeps_old_estimates_and_stop_is_available_only_at_target(self):
        from fly_brain_rc.docking import prepare_docking
        a = QLearningAgent()
        a.task = 'bypass-v1'
        old = RLController(a).observe(frame(tof=1.2),None,None)
        a.table[old] = [2.0]*11
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'policy.json'
            a.save(p)
            original = p.read_bytes()
            a = QLearningAgent.load(p)
            prepare_docking(a)
            self.assertEqual(a.table[old][:11],[2.0]*11)
            self.assertEqual(len(a.table[old]),len(a.actions))
            c = RLController(a)
            state = c.observe(frame(),ball(),None)
            stop = a.actions.index((0,0))
            self.assertIn(stop,a.admissible_actions(state))
            self.assertEqual(a.choose(state,explore=False),stop)
            far = RLController(a).observe(frame(tof=1.2),None,None)
            self.assertNotIn(stop,a.admissible_actions(far))
            self.assertEqual(a.values(far)[:11],[2.0]*11)
            self.assertEqual(p.read_bytes(),original)

    def test_critical_ray_is_distinct_and_training_updates_new_docking_rows(self):
        from fly_brain_rc.docking import prepare_docking
        a = QLearningAgent()
        a.task = 'bypass-v1'
        prepare_docking(a)
        c = RLController(a)
        state = c.observe(frame(range_left=.01),ball(),None)
        self.assertEqual(state[3],'critical')
        clear = c.observe(frame(),ball(),None)
        stop = a.actions.index((0,0))
        a.update(clear,stop,2,clear,terminal=True)
        self.assertGreater(a.values(clear)[stop],0)

    def test_dangerous_closing_costs_more_but_retreat_does_not_earn_reward(self):
        from fly_brain_rc.bypass_learning import BypassReward
        r = BypassReward(2)
        args = dict(contact=False,steering=0,reached=False,dt=.2,target=None,
                    command=ActuatorCommand(.3,0,0,0))
        r.step(2,sensor=frame(range_left=.3),**args)
        closing = r.step(2,sensor=frame(range_left=.01),**args)
        retreat = r.step(2,sensor=frame(range_left=.3),**args)
        self.assertLess(closing,retreat)
        self.assertLessEqual(retreat,0)

    def test_initial_proximity_alone_does_not_add_a_hazard_penalty(self):
        from fly_brain_rc.bypass_learning import BypassReward
        args = dict(contact=False,steering=0,reached=False,dt=.2,target=None,
                    command=ActuatorCommand(0,0,0,0))
        near = BypassReward(2).step(2,sensor=frame(range_left=.01),**args)
        clear = BypassReward(2).step(2,sensor=frame(range_left=1.2),**args)
        self.assertAlmostEqual(near,clear)

    def test_shaping_does_not_reward_driving_through_target_or_waiting_elsewhere(self):
        from fly_brain_rc.bypass_learning import BypassReward
        r = BypassReward(.7)
        closer = r.step(.4,contact=False,steering=0,reached=False,dt=.2,
                        sensor=frame(tof=.01),target=ball(),command=ActuatorCommand(.3,0,0,0))
        self.assertLess(closer,0)
        parked = BypassReward(.7).step(.7,contact=False,steering=0,reached=False,dt=.2,
                        sensor=frame(),target=ball(),command=ActuatorCommand(0,0,0,0))
        idle = BypassReward(2).step(2,contact=False,steering=0,reached=False,dt=.2,
                        sensor=frame(tof=1.2),target=None,command=ActuatorCommand(0,0,0,0))
        self.assertGreater(parked,0)
        self.assertLess(idle,0)

    def test_runtime_does_not_auto_finish_when_ball_is_beside_car(self):
        from unittest.mock import patch
        from tests.test_environment import HeadlessRenderer
        from fly_brain_rc.config import SIM,DRIVE
        from fly_brain_rc.runtime import FlyRuntime
        from fly_brain_rc.rl import QLearningBrain
        from fly_brain_rc.sim.env import FlyCarEnv
        from fly_brain_rc.sim.scenarios import Scenario
        from fly_brain_rc.dashboard.server import MissionController
        a = QLearningAgent()
        a.task = 'bypass-v1'
        with patch('fly_brain_rc.sim.env.mujoco.Renderer',HeadlessRenderer):
            env = FlyCarEnv(SIM,DRIVE,scenario=Scenario('beside',(),(0,.45)))
            mission = MissionController(running=True)
            FlyRuntime(env,QLearningBrain(a),mission=mission,bypass_evaluation=True).tick()
            self.assertEqual(mission.state()['status'],'running')
            env.renderer.close()
