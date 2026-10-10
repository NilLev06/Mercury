import unittest
from unittest.mock import patch
import torch,numpy as np
from fly_brain_rc.contracts import SensorFrame
from fly_brain_rc.imitation.model import RecurrentPolicy
class PrecontactDaggerTests(unittest.TestCase):
    def test_collection_stops_at_contact_and_executes_only_neural_commands(self):
        from scripts.train_precontact_dagger import collect_precontact
        graph=dict(ids=['a','b','c'],edges=[[0,1,2],[1,2,3],[2,0,4]])
        model=RecurrentPolicy(graph,motor_head='residual32')
        class Env:
            speed=0.;_contact_active=False;collisions=0
            def __init__(self):self.steps=0;self.commands=[]
            def sensor_frame(self):return SensorFrame(np.zeros((4,4,3),np.uint8),np.zeros((4,4),np.uint8),0.,0.,1.2,False,0.)
            def step(self,*action,**kwargs):
                self.commands.append(action);self.steps+=1
                if self.steps==3:self._contact_active=True;self.collisions=1
        class Teacher:
            def action(self):return (.987,.876)
        env=Env()
        with patch('scripts.train_precontact_dagger.body_goal_distance',return_value=2.):data,report=collect_precontact(env,model,Teacher(),steps=100)
        self.assertEqual(report['steps'],3);self.assertEqual(report['collection_cutoff'],'contact');self.assertEqual(len(data['x']),1)
        torch.testing.assert_close(data['y'][0],torch.tensor([.987,.876]))
        self.assertNotEqual(env.commands[0],(.987,.876));self.assertEqual(env.commands[0],env.commands[-1])

    def test_stationary_cutoff_does_not_override_motor_or_claim_success(self):
        from scripts.train_precontact_dagger import collect_precontact
        model=RecurrentPolicy(dict(ids=['a'],edges=[]),motor_head='residual32')
        class Env:
            speed=0.;_contact_active=False;collisions=0
            def sensor_frame(self):return SensorFrame(np.zeros((4,4,3),np.uint8),np.zeros((4,4),np.uint8),0.,0.,1.2,False,0.)
            def step(self,*action,**kwargs):pass
        teacher=type('Teacher',(),{'action':lambda self:(.45,.5)})()
        with patch('scripts.train_precontact_dagger.body_goal_distance',return_value=2.):data,report=collect_precontact(Env(),model,teacher,steps=100,max_idle=.06)
        self.assertEqual(report['steps'],3);self.assertFalse(report['success']);self.assertEqual(report['collection_cutoff'],'stationary')
        self.assertEqual(len(data['x']),len(data['h']))

class RehearsalSelectionTests(unittest.TestCase):
    def test_balanced_rotation_has_old_canonical_plus_disjoint_chunks(self):
        from scripts.train_precontact_dagger import rotating_collection
        from scripts.prepare_neural_rehearsal import case_spec
        from fly_brain_rc.sim.scenarios import Scenario
        pool=[case_spec(Scenario(f'{tier}-{i}',(),(2.,0.)),(0.,float(i),0.),tier)
              for tier in ('simple','complex') for i in range(20)]
        a=rotating_collection(pool,0,8);b=rotating_collection(pool,1,8)
        self.assertEqual(len(a),23);self.assertEqual(len(b),23)
        self.assertEqual(len(set(c.name for c,p,t in a[7:])&set(c.name for c,p,t in b[7:])),0)
        self.assertTrue(all(p==(0.,0.,0.) for c,p,t in a[:7]))
        self.assertEqual([p[1] for c,p,t in b[7:15]],list(range(8,16)))

    def test_large_gate_rejects_contact_arrivals_and_simple_regression(self):
        from scripts.train_precontact_dagger import expanded_gate
        baseline=dict(reports=[dict(family='simple',success=True,contacts=0) for _ in range(16)])
        old=dict(successes=7,contacts=0)
        previous=dict(reports=[dict(family='complex',success=True,contacts=0) for _ in range(6)])
        clean=dict(reports=baseline['reports']+[dict(family='complex',success=i<29,contacts=0) for i in range(32)])
        self.assertTrue(expanded_gate(old,previous,clean,baseline))
        clean['reports'][-1]['contacts']=1
        self.assertFalse(expanded_gate(old,previous,clean,baseline))
        clean['reports'][-1]['contacts']=0;clean['reports'][0]=dict(family='simple',success=False,contacts=0)
        self.assertFalse(expanded_gate(old,previous,clean,baseline))

    def test_expanded_gate_supports_predeclared_larger_64_complex_pool(self):
        from scripts.train_precontact_dagger import expanded_gate
        baseline=dict(reports=[dict(family='simple',success=i<31,contacts=0) for i in range(32)])
        old=dict(successes=7,contacts=0);previous=dict(reports=[dict(family='complex',success=True,contacts=0) for _ in range(6)])
        new=dict(reports=[dict(r) for r in baseline['reports']]+[dict(family='complex',success=i<58,contacts=0) for i in range(64)])
        self.assertTrue(expanded_gate(old,previous,new,baseline))
        new['reports'][32+57]['success']=False
        self.assertFalse(expanded_gate(old,previous,new,baseline))

class SelfCollectionTests(unittest.TestCase):
    def test_absent_teacher_records_actual_student_commands(self):
        from scripts.train_precontact_dagger import collect_precontact
        from unittest.mock import patch
        from fly_brain_rc.imitation.model import RecurrentPolicy
        from fly_brain_rc.contracts import SensorFrame
        import numpy as np,torch
        model=RecurrentPolicy(dict(ids=['a','b'],edges=[[0,1,2]]),motor_head='residual32')
        class Env:
            speed=.1;collisions=0;_contact_active=False
            def __init__(self):self.commands=[]
            def sensor_frame(self):return SensorFrame(np.zeros((30,120,3),np.uint8),np.zeros((30,120),np.uint8),0.,0.,1.2,False,0.)
            def step(self,*a,**kw):self.commands.append(a)
        env=Env()
        with patch('scripts.train_precontact_dagger.body_goal_distance',return_value=2.):data,r=collect_precontact(env,model,None,steps=10)
        torch.testing.assert_close(data['y'],torch.tensor([env.commands[0],env.commands[5]]))
