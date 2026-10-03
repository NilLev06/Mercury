from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from fly_brain_rc.config import SIM, DRIVE
from fly_brain_rc.sim.env import FlyCarEnv
from fly_brain_rc.sim.scenarios import Scenario, BLUE
from fly_brain_rc.sim.kinematics import BicycleState
from fly_brain_rc.rl import QLearningAgent, RLController
from tests.test_environment import HeadlessRenderer


class SurroundRangesTests(unittest.TestCase):
    def test_front_diagonals_use_outer_edges_and_true_45_degree_directions(self):
        import math
        env = FlyCarEnv(SIM,DRIVE,scenario=Scenario('geometry',(),(2,0)),render=False)
        for name,y,sign in (('sF_L',.202,1),('sF_R',-.202,-1)):
            site = env.m.site(name).id
            np.testing.assert_allclose(env.d.site_xpos[site][:2],[.22,y],atol=1e-6)
            direction = env.d.site_xmat[site].reshape(3,3)[:,2]
            np.testing.assert_allclose(direction,[1/math.sqrt(2),sign/math.sqrt(2),0],atol=1e-6)
        for address in (env.adr_b_fl,env.adr_b_fr,env.adr_b_sl,env.adr_b_sr,env.adr_b_rc):
            self.assertEqual(env._range_reading(address),1.2)

    def test_new_side_distances_reach_browser_telemetry(self):
        from fly_brain_rc.contracts import RuntimeTelemetry,BrainTelemetry,MotionObservation,ActuatorCommand
        from fly_brain_rc.telemetry import TelemetryBus
        with patch('fly_brain_rc.sim.env.mujoco.Renderer',HeadlessRenderer):
            env = FlyCarEnv(SIM,DRIVE,scenario=Scenario('side',(((.09,.55),BLUE),),(2,0)))
            sensor = env.sensor_frame()
            record = RuntimeTelemetry(0,1,sensor,None,MotionObservation(0,0,0,0,0,0),BrainTelemetry(),ActuatorCommand(0,0,0,1))
            payload = TelemetryBus().serialize(record)
            self.assertAlmostEqual(payload['sensor']['range_side_left'],.068,places=3)
            self.assertEqual(payload['sensor']['range_rear'],1.2)
            env.renderer.close()

    def test_side_cylinder_detected_before_contact_on_both_sides(self):
        for y,attribute in ((.55,'range_side_left'),(-.55,'range_side_right')):
            with self.subTest(y=y), patch('fly_brain_rc.sim.env.mujoco.Renderer',HeadlessRenderer):
                env = FlyCarEnv(SIM,DRIVE,scenario=Scenario('side',(((.09,y),BLUE),),(2,0)))
                sensor = env.sensor_frame()
                # .55 cylinder center minus radius .28 and wheel edge .202.
                self.assertAlmostEqual(getattr(sensor,attribute),.068,places=3)
                self.assertFalse(sensor.contact)
                self.assertEqual(sensor.tof,1.2)
                env.renderer.close()

    def test_side_and_rear_wall_clearances(self):
        for pose,attribute,gap in (((0,2.65,0),'range_side_left',.148),
                                    ((0,-2.65,0),'range_side_right',.148),
                                    ((-2.7,0,0),'range_rear',.26)):
            with self.subTest(attribute=attribute), patch('fly_brain_rc.sim.env.mujoco.Renderer',HeadlessRenderer):
                env = FlyCarEnv(SIM,DRIVE,scenario=Scenario('wall',(),(2,0)))
                env.state = BicycleState(*pose,0)
                env._sync_kinematic()
                self.assertAlmostEqual(getattr(env.sensor_frame(),attribute),gap,places=3)
                env.renderer.close()

    def test_zero_clearance_is_not_reported_as_miss(self):
        env = FlyCarEnv(SIM,DRIVE,scenario=Scenario('zero',(),(2,0)),render=False)
        env.d.sensordata[env.adr_b_fc] = 0
        self.assertEqual(env._range_reading(env.adr_b_fc),0)
        env.d.sensordata[env.adr_b_fc] = -1
        self.assertEqual(env._range_reading(env.adr_b_fc),1.2)

    def test_bypass_can_distinguish_side_and_rear_danger(self):
        with patch('fly_brain_rc.sim.env.mujoco.Renderer',HeadlessRenderer):
            env = FlyCarEnv(SIM,DRIVE,scenario=Scenario('clear',(),(2,0)))
            sensor = env.sensor_frame()
            a = QLearningAgent()
            a.task = 'bypass-v1'
            c = RLController(a)
            clear = c.observe(sensor,None,None)
            left = c.observe(replace(sensor,range_side_left=.1),None,None)
            rear = c.observe(replace(sensor,range_rear=.1),None,None)
            self.assertNotEqual(clear,left)
            self.assertNotEqual(clear,rear)
            self.assertNotEqual(left,rear)
            self.assertEqual(len(clear),10)
            # Same extra sensors must not change the general policy schema.
            general = RLController(QLearningAgent())
            self.assertEqual(len(general.observe(sensor,None,None)),9)
            env.renderer.close()

    def test_old_bypass_estimates_warm_start_but_new_context_learns_independently(self):
        import json
        oldstate = ('target_center','clear','small','clear','clear','clear','forward','no_contact','no_contact')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'policy.json'
            a = QLearningAgent()
            a.task = 'bypass-v1'
            a.training_runs = 3
            a.table[oldstate] = [2.0]*len(a.actions)
            a.save(path)
            data = json.loads(path.read_text())
            data.pop('sensor_schema',None)
            path.write_text(json.dumps(data))
            original = path.read_bytes()
            loaded = QLearningAgent.load(path)
            newstate = oldstate+('side_left:rear_clear',)
            self.assertEqual(loaded.values(newstate),[2.0]*11)
            self.assertEqual(loaded.training_runs,3)
            loaded.update(newstate,0,-3,newstate,terminal=True)
            self.assertEqual(loaded.table[oldstate],[2.0]*11)
            self.assertLess(loaded.values(newstate)[0],2)
            self.assertEqual(path.read_bytes(),original)
