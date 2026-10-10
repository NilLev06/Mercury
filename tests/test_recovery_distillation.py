import unittest
import torch


class RecoveryDistillationTests(unittest.TestCase):
    def test_distills_full_history_without_mutating_legacy_labels(self):
        from scripts.train_recovery_distillation import distilled_replay
        class Model:
            def __call__(self,x):
                return torch.cumsum(x[:,:,:2],1),None
        fixed=[dict(x=torch.ones(3,31),y=torch.zeros(3,2),h=torch.zeros(3,4),tier=t) for t in ('simple','complex')]
        recoveries=[dict(x=torch.zeros(2,31),y=torch.full((2,2),-.3),tier=t,supervision_start=1) for t in ('simple','complex')]
        replay,summary=distilled_replay(Model(),fixed,recoveries)
        torch.testing.assert_close(replay[0]['y'],torch.tensor([[1.,1.],[2.,2.],[3.,3.]]))
        self.assertFalse(replay[0]['y'].requires_grad);self.assertNotIn('h',replay[0])
        self.assertTrue(torch.all(fixed[0]['y']==0));self.assertEqual(replay[2]['supervision_start'],1)
        torch.testing.assert_close(replay[2]['y'],recoveries[0]['y'])
        self.assertEqual(summary['simple']['recovery_fraction'],.5)

    def test_balance_each_tier_and_reject_missing_recoveries(self):
        from scripts.train_recovery_distillation import distilled_replay
        class Model:
            def __call__(self,x):return x[:,:,:2],None
        fixed=[dict(x=torch.ones(1,31),y=torch.zeros(1,2),tier=t) for t in ('simple','complex') for _ in range(5)]
        rec=[dict(x=torch.ones(1,31),y=torch.zeros(1,2),tier=t) for t in ('simple','complex') for _ in range(2)]
        replay,summary=distilled_replay(Model(),fixed,rec)
        self.assertEqual(len(replay),22)
        self.assertAlmostEqual(summary['complex']['recovery_fraction'],6/11)
        with self.assertRaises(ValueError):distilled_replay(Model(),fixed,rec[:2])
