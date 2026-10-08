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
