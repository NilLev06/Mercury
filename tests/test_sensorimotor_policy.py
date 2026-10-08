import unittest
import torch

from fly_brain_rc.imitation.sensorimotor_policy import SensorimotorPolicy, observable_drive


class SensorimotorPolicyTests(unittest.TestCase):
    def test_motor_memory_preserves_chunked_inference_and_zero_residual_baseline(self):
        g=dict(ids=['0','1','2'],edges=[[0,2,5],[1,2,5]],neurons=[{}]*3,
            inputs={'LC10a:left':[0],'LC10a:right':[1]},outputs=[2])
        old=SensorimotorPolicy(g);model=SensorimotorPolicy(g,motor_memory=True)
        model.head.load_state_dict(old.head.state_dict())
        x=torch.rand(2,12,19)
        full,state=model(x)
        a,s=model(x[:,:5]);b,s=model(x[:,5:],s)
        self.assertTrue(torch.allclose(full,torch.cat((a,b),1),atol=1e-7))
        self.assertTrue(torch.allclose(state,s,atol=1e-7))
        self.assertTrue(torch.equal(old(x)[0],full))
        self.assertEqual(state.shape,(2,19))

    def test_motor_memory_cannot_receive_observations_without_anatomical_edges(self):
        g=dict(ids=['0','1','2'],edges=[],neurons=[{}]*3,
            inputs={'LC10a:left':[0],'LC10a:right':[1]},outputs=[2])
        model=SensorimotorPolicy(g,motor_memory=True)
        with torch.no_grad():model.memory_readout.weight.fill_(.2)
        x=torch.zeros(2,30,19);x[1,:,0]=1
        y,s=model(x)
        self.assertTrue(torch.equal(y[0],y[1]))
        self.assertTrue(torch.equal(s[0,3:],s[1,3:]))

    def test_motor_memory_distinguishes_neural_history_after_common_input(self):
        g=dict(ids=['0','1','2'],edges=[[0,2,5],[1,2,5]],neurons=[{}]*3,
            inputs={'LC10a:left':[0],'LC10a:right':[1]},outputs=[2])
        torch.manual_seed(4);model=SensorimotorPolicy(g,motor_memory=True)
        x=torch.zeros(2,20,19);x[1,:,0]=1
        _,s=model(x)
        _,s=model(torch.zeros(2,5,19),s)
        self.assertGreater((s[0,3:]-s[1,3:]).abs().sum().item(),1e-5)

    def test_target_direction_routes_to_separate_sides_without_hidden_goal(self):
        x=torch.zeros(2,19);x[:,0]=1;x[0,1]=-1;x[1,1]=1
        names=('LC10a:left','LC10a:right')
        y=observable_drive(x,names)
        self.assertGreater(y[0,0],y[0,1])
        self.assertGreater(y[1,1],y[1,0])

    def test_without_connections_raw_camera_features_cannot_bypass_core(self):
        g=dict(ids=['0','1','2'],edges=[],neurons=[{}]*3,
            inputs={'LC10a:left':[0],'LC10a:right':[1]},outputs=[2])
        policy=SensorimotorPolicy(g)
        x=torch.zeros(2,5,19);x[1,:,0]=1;x[1,:,1]=1
        y,state=policy(x)
        self.assertTrue(torch.equal(y[0],y[1]))
        self.assertEqual(tuple(state.shape),(2,3))
        self.assertEqual(tuple(y.shape),(2,5,2))

    def test_rejects_unknown_channel_and_wrong_observation_size(self):
        with self.assertRaises(ValueError):observable_drive(torch.zeros(1,19),('invented',))
        with self.assertRaises(ValueError):observable_drive(torch.zeros(1,20),('LC10a:left',))

    def test_visible_target_size_reaches_population_drive_for_arrival_learning(self):
        x=torch.zeros(2,19);x[:,0]=1;x[1,2]=1
        y=observable_drive(x,('LC10a:left','LC10a:right'))
        self.assertGreater(y[1].sum().item(),y[0].sum().item())

    def test_learned_input_receives_distinct_geometry_and_ego_motion(self):
        g=dict(ids=['0','1','2'],edges=[[0,2,5],[1,2,5]],neurons=[{}]*3,
            inputs={'LC10a:left':[0],'LC10a:right':[1]},outputs=[2])
        model=SensorimotorPolicy(g,learned_input=True)
        x=torch.zeros(1,4,19);x[:,:,6:12]=1
        # Calibrate one adapter route, independently for each formerly ignored
        # signal. Removing that observation from the encoder breaks this test.
        for feature in (7,9,11,12,13):
            with torch.no_grad():
                for p in model.encoder.parameters():p.zero_()
                model.encoder[0].weight[0,feature]=1
                model.encoder[2].weight[0,0]=1
            changed=x.clone();changed[:,:,feature]=.1 if feature<12 else 1
            _,a=model(x);_,b=model(changed)
            self.assertFalse(torch.equal(model.core.readout(a),model.core.readout(b)))

    def test_learned_input_gradients_cross_frozen_core_without_motor_bypass(self):
        g=dict(ids=['0','1','2'],edges=[[0,2,5],[1,2,5]],neurons=[{}]*3,
            inputs={'LC10a:left':[0],'LC10a:right':[1]},outputs=[2])
        model=SensorimotorPolicy(g,learned_input=True)
        x=torch.zeros(1,5,19);x[:,:,12]=1
        model(x)[0].sum().backward()
        self.assertGreater(model.encoder[2].weight.grad.abs().sum().item(),0)
        self.assertIsNone(model.core.weights.grad)
        g['edges']=[];model=SensorimotorPolicy(g,learned_input=True)
        y,_=model(torch.cat((x,x*0),0))
        self.assertTrue(torch.equal(y[0],y[1]))

    def test_zero_initialized_input_adapter_preserves_existing_policy(self):
        g=dict(ids=['0','1','2'],edges=[[0,2,5],[1,2,5]],neurons=[{}]*3,
            inputs={'LC10a:left':[0],'LC10a:right':[1]},outputs=[2])
        baseline=SensorimotorPolicy(g);model=SensorimotorPolicy(g,learned_input=True)
        model.head.load_state_dict(baseline.head.state_dict())
        x=torch.rand(2,5,19)
        self.assertTrue(torch.equal(baseline(x)[0],model(x)[0]))
