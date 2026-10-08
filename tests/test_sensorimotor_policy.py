import unittest
import torch

from fly_brain_rc.imitation.sensorimotor_policy import SensorimotorPolicy, observable_drive


class SensorimotorPolicyTests(unittest.TestCase):
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
