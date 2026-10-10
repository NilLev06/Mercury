import unittest
import torch
from fly_brain_rc.imitation.model import RecurrentPolicy

class ThrottleCalibrationTests(unittest.TestCase):
    def test_only_final_throttle_row_changes_and_steering_predictions_are_exact(self):
        from scripts.calibrate_neural_throttle import fit_throttle
        torch.manual_seed(9)
        model=RecurrentPolicy(dict(ids=['a','b','c'],edges=[[0,1,2],[1,2,3]]),motor_head='residual32')
        h=torch.randn(50,3);y=torch.zeros(50,2);y[:,0]=.4
        before={k:v.clone() for k,v in model.state_dict().items()}
        with torch.no_grad():steer=model.readout(h)[:,1].clone();error=(torch.tanh(model.readout(h)[:,0])-.4).square().mean()
        fit_throttle(model,[dict(h=h,y=y)],(h,y),updates=30,anchor_weight=1.)
        with torch.no_grad():
            torch.testing.assert_close(model.readout(h)[:,1],steer,rtol=0,atol=0)
            self.assertLess((torch.tanh(model.readout(h)[:,0])-.4).square().mean(),error)
        for k,v in model.state_dict().items():
            if k not in ('readout.residual.2.weight','readout.residual.2.bias'):torch.testing.assert_close(v,before[k],rtol=0,atol=0)
            else:torch.testing.assert_close(v[1],before[k][1],rtol=0,atol=0)
