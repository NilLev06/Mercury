import unittest
import torch
class RecoveryReplayTests(unittest.TestCase):
    def test_prefix_replays_partial_last_decision_exactly(self):
        from scripts.audit_neural_recovery import replay_prefix
        class Env:
            def __init__(self):self.commands=[]
            def step(self,*a,**kw):self.commands.append(a)
        e=Env();replay_prefix(e,torch.tensor([[.1,.2],[.3,.4]]),7)
        self.assertEqual(len(e.commands),7);self.assertEqual(e.commands[:5],[tuple(torch.tensor([.1,.2]).tolist())]*5)
        self.assertEqual(e.commands[5:],[tuple(torch.tensor([.3,.4]).tolist())]*2)
        with self.assertRaises(ValueError):replay_prefix(e,torch.zeros(3,2),7)
