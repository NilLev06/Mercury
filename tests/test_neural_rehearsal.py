import unittest
from unittest.mock import patch
import torch,numpy as np
from fly_brain_rc.contracts import SensorFrame
from fly_brain_rc.imitation.model import RecurrentPolicy,upgrade_sector_inputs
from fly_brain_rc.sim.scenarios import Scenario,BLUE
class NeuralRehearsalTests(unittest.TestCase):
    def test_layout_roundtrip_keeps_pose_goal_and_colours_separate_from_inputs(self):
        from scripts.prepare_neural_rehearsal import case_spec,restore_case
        c=Scenario('test',(((1.,.2),BLUE),),(2.4,-.1));p=(0.,.02,.03)
        restored,pose,tier=restore_case(case_spec(c,p,'complex'))
        self.assertEqual(restored,c);self.assertEqual(pose,p);self.assertEqual(tier,'complex')

    def test_original_19_actor_controls_while_actual_31_features_are_recorded(self):
        from scripts.prepare_neural_rehearsal import collect_neural_teacher
        graph=dict(ids=['a','b','c'],edges=[[0,1,2],[1,2,3],[2,0,4]])
        torch.manual_seed(31);actor=RecurrentPolicy(graph,motor_head='residual32');student=upgrade_sector_inputs(actor)
        class Env:
            speed=0.;collisions=0
            def __init__(self):self.commands=[]
            def sensor_frame(self):
                rgb=np.zeros((30,120,3),np.uint8);rgb[5:25,5:15]=(0,0,255)
                return SensorFrame(rgb,np.zeros((30,120),np.uint8),0.,0.,1.2,False,0.)
            def step(self,*action,**kwargs):self.commands.append(action)
        env=Env()
        with patch('scripts.prepare_neural_rehearsal.body_goal_distance',return_value=2.):data,report=collect_neural_teacher(env,actor,student,steps=10)
        self.assertEqual(data['x'].shape,(2,31));self.assertGreater(float(data['x'][0,19]),0.)
        with torch.no_grad():expected,_=actor(data['x'][None,:,:19]);_,state=student(data['x'][None,:1])
        torch.testing.assert_close(data['y'],expected[0]);torch.testing.assert_close(data['h'][0],state[0])
        self.assertEqual(env.commands[0],tuple(expected[0,0].tolist()));self.assertEqual(env.commands[5],tuple(expected[0,1].tolist()))

    def test_neural_correction_labels_share_frame_but_student_alone_drives(self):
        from scripts.prepare_neural_rehearsal import NeuralLabelTeacher
        from scripts.train_precontact_dagger import collect_precontact
        graph=dict(ids=['a','b','c'],edges=[[0,1,2],[1,2,3]])
        torch.manual_seed(17);actor=RecurrentPolicy(graph,motor_head='residual32');student=upgrade_sector_inputs(actor)
        with torch.no_grad():student.readout.residual[-1].bias.add_(.2)
        class Env:
            speed=.1;collisions=0;_contact_active=False
            def __init__(self):self.commands=[];self.frames=0
            def sensor_frame(self):
                self.frames+=1
                return SensorFrame(np.zeros((30,120,3),np.uint8),np.zeros((30,120),np.uint8),0.,0.,1.2,False,0.)
            def step(self,*a,**kw):self.commands.append(a)
        env=Env();teacher=NeuralLabelTeacher(actor)
        with patch('scripts.train_precontact_dagger.body_goal_distance',return_value=2.):data,r=collect_precontact(env,student,teacher,steps=10)
        with torch.no_grad():labels,_=actor(data['x'][None,:,:19]);commands,_=student(data['x'][None])
        self.assertEqual(env.frames,2);torch.testing.assert_close(data['y'],labels[0])
        self.assertEqual(env.commands[0],tuple(commands[0,0].tolist()))
        self.assertFalse(torch.equal(labels,commands))

    def test_core_fingerprint_allows_motor_updates_but_rejects_sensory_change(self):
        from scripts.prepare_neural_rehearsal import core_fingerprint
        model=RecurrentPolicy(dict(ids=['a','b'],edges=[[0,1,2]]),motor_head='residual32')
        initial=core_fingerprint(model)
        with torch.no_grad():model.readout.residual[-1].bias.add_(.1)
        self.assertEqual(core_fingerprint(model),initial)
        with torch.no_grad():model.encoder.bias.add_(.01)
        self.assertNotEqual(core_fingerprint(model),initial)

    def test_fallback_only_runs_after_failed_neural_demo_and_keeps_both_reports(self):
        from scripts.prepare_neural_rehearsal import demonstration_attempts
        calls=[]
        def expert():calls.append('expert');return {'y':'expert'},dict(success=True,contacts=0)
        good=lambda:({'y':'neural'},dict(success=True,contacts=0))
        self.assertEqual(len(demonstration_attempts(good,expert)),1);self.assertEqual(calls,[])
        failed=lambda:({'y':'neural'},dict(success=True,contacts=2))
        attempts=demonstration_attempts(failed,expert)
        self.assertEqual(len(attempts),2);self.assertEqual(calls,['expert']);self.assertEqual(attempts[0][1]['contacts'],2)
        self.assertEqual(attempts[1][0]['y'],'expert')
