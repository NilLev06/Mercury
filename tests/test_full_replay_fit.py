import unittest
import torch
from fly_brain_rc.imitation.model import RecurrentPolicy

class FullReplayFitTests(unittest.TestCase):
    def test_full_batch_fitting_learns_both_tiers_with_masked_recurrence(self):
        from scripts.fit_full_replay import FullReplayFitter
        torch.set_num_threads(2);torch.manual_seed(21)
        graph=dict(ids=['a','b','c'],edges=[[0,1,2],[1,2,3],[2,0,4]])
        model=RecurrentPolicy(graph,motor_head='residual32',trainable_core=True)
        x=torch.rand(12,19);y=torch.tensor([[.45,-.3]]).repeat(12,1)
        episodes=[dict(x=x,y=y,tier='simple'),dict(x=x[:8],y=y[:8],tier='complex')]
        before=float((model(x[None])[0][0]-y).square().mean().detach())
        mask=model.mask.clone();fitter=FullReplayFitter(model,episodes,opening_weight=1.)
        for _ in range(8):fitter.step()
        after=float((model(x[None])[0][0]-y).square().mean().detach())
        self.assertLess(after,before/20)
        self.assertTrue(torch.equal(model.mask,mask))
        self.assertTrue(torch.all(model.effective_weights()[mask==0]==0))
