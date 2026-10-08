import tempfile
from pathlib import Path
import unittest
import torch

from fly_brain_rc.imitation.sensorimotor_policy import SensorimotorPolicy
from fly_brain_rc.imitation.sensorimotor_training import cache_demonstrations,fit_cached,save_trial,load_trial,fit_interfaces


def policy():
    return SensorimotorPolicy(dict(ids=['0','1','2'],edges=[[0,2,5],[1,2,5]],
        neurons=[{}]*3,inputs={'LC10a:left':[0],'LC10a:right':[1]},outputs=[2]))


class SensorimotorTrainingTests(unittest.TestCase):
    def test_interface_fit_updates_encoder_without_changing_anatomy(self):
        torch.manual_seed(8)
        model=SensorimotorPolicy(policy().graph,learned_input=True)
        optimizer=torch.optim.Adam(model.parameters(),lr=.002)
        x=torch.zeros(8,19);x[:,12]=1
        data=[(x,torch.ones(8,2)*.4)]
        before=model.core.weights.to_dense().clone()
        fit_interfaces(model,optimizer,data,updates=3,batch_size=2,sequence_length=4)
        self.assertGreater(model.encoder[-1].weight.abs().sum().item(),0)
        self.assertTrue(torch.equal(before,model.core.weights.to_dense()))
        with self.assertRaises(ValueError):
            fit_cached(model,optimizer,cache_demonstrations(model,data),updates=1)

    def test_learned_interface_resume_preserves_both_adapters(self):
        torch.manual_seed(4)
        model=SensorimotorPolicy(policy().graph,learned_input=True)
        optimizer=torch.optim.Adam(model.parameters(),lr=.002)
        data=[(torch.rand(8,19),torch.zeros(8,2))]
        fit_interfaces(model,optimizer,data,updates=2,batch_size=2,sequence_length=4)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'trial.pt'
            save_trial(path,model,optimizer,data,rounds=1)
            fit_interfaces(model,optimizer,data,updates=2,batch_size=2,sequence_length=4)
            restored,opt,replay,_=load_trial(path)
            self.assertTrue(restored.learned_input)
            fit_interfaces(restored,opt,replay,updates=2,batch_size=2,sequence_length=4)
            for actual,expected in zip(restored.parameters(),model.parameters()):
                self.assertTrue(torch.equal(actual,expected))

    def test_resume_restores_named_optimizer_groups_and_parameter_order(self):
        torch.manual_seed(7)
        model=SensorimotorPolicy(policy().graph,learned_input=True)
        optimizer=torch.optim.Adam([
            {'params':list(model.head.parameters()),'lr':.001},
            {'params':list(model.encoder.parameters()),'lr':.003}])
        data=[(torch.rand(8,19),torch.zeros(8,2))]
        fit_interfaces(model,optimizer,data,updates=2,batch_size=2,sequence_length=4)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'trial.pt'
            save_trial(path,model,optimizer,data,rounds=1)
            fit_interfaces(model,optimizer,data,updates=2,batch_size=2,sequence_length=4)
            restored,opt,replay,_=load_trial(path)
            fit_interfaces(restored,opt,replay,updates=2,batch_size=2,sequence_length=4)
            self.assertEqual([g['lr'] for g in opt.param_groups],[.001,.003])
            for actual,expected in zip(restored.parameters(),model.parameters()):
                self.assertTrue(torch.equal(actual,expected))

    def test_interface_loss_excludes_padding_for_unequal_episodes(self):
        torch.manual_seed(21)
        model=SensorimotorPolicy(policy().graph,learned_input=True)
        with torch.no_grad():
            for p in model.head.parameters():p.zero_()
        optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        data=[(torch.zeros(n,19),torch.full((n,2),.2)) for n in (1,80)]
        loss=fit_interfaces(model,optimizer,data,updates=1,batch_size=16,sequence_length=8)
        self.assertAlmostEqual(loss,.04,places=6)

    def test_cached_features_match_live_forward_and_reset_each_episode(self):
        model=policy();x=torch.zeros(12,19);x[:,0]=1;x[:,2]=.5
        data=[(x,torch.zeros(12,2)),(x,torch.zeros(12,2))]
        cache=cache_demonstrations(model,data)
        self.assertTrue(torch.equal(cache[0][0],cache[1][0]))
        predicted=torch.tanh(model.head((cache[0][0]-model.feature_mean)/model.feature_scale))
        self.assertTrue(torch.allclose(predicted,model(x[None])[0][0],atol=1e-7))

    def test_fit_changes_only_motor_head_and_learns_teacher_labels(self):
        torch.manual_seed(1);model=policy()
        x=torch.zeros(24,19);x[:,0]=1
        cache=cache_demonstrations(model,[(x,torch.zeros(24,2))])
        before={k:v.clone() for k,v in model.core.state_dict().items()}
        optimizer=torch.optim.Adam(model.head.parameters(),lr=.02)
        loss=fit_cached(model,optimizer,cache,updates=80,batch_size=16)
        self.assertLess(loss,.005)
        for key,value in before.items():
            actual=model.core.state_dict()[key]
            if value.is_sparse:
                self.assertTrue(torch.equal(value.to_dense(),actual.to_dense()))
            else:self.assertTrue(torch.equal(value,actual))

    def test_checkpoint_roundtrip_retains_optimizer_and_data_without_overwrite(self):
        model=policy();optimizer=torch.optim.Adam(model.head.parameters(),lr=.01)
        data=[(torch.zeros(4,19),torch.zeros(4,2))]
        fit_cached(model,optimizer,cache_demonstrations(model,data),updates=2)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'trial.pt'
            save_trial(path,model,optimizer,data,rounds=2)
            restored,opt,replay,rounds=load_trial(path)
            self.assertEqual(rounds,2);self.assertEqual(len(replay),1)
            self.assertTrue(opt.state_dict()['state'])
            self.assertTrue(torch.equal(model(data[0][0][None])[0],restored(data[0][0][None])[0]))
            with self.assertRaises(FileExistsError):save_trial(path,model,optimizer,data,rounds=3)

    def test_invalid_or_misaligned_demonstrations_are_rejected(self):
        model=policy()
        for data in ([],[(torch.zeros(4,19),torch.zeros(3,2))],
                     [(torch.full((4,19),float('nan')),torch.zeros(4,2))]):
            with self.assertRaises(ValueError):cache_demonstrations(model,data)

    def test_unsupported_optimizer_is_rejected_before_saving(self):
        model=policy();optimizer=torch.optim.SGD(model.head.parameters(),lr=.01)
        data=[(torch.zeros(4,19),torch.zeros(4,2))]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'trial.pt'
            with self.assertRaises(ValueError):save_trial(path,model,optimizer,data,rounds=0)
            self.assertFalse(path.exists())

    def test_resume_reproduces_continued_training_exactly(self):
        torch.manual_seed(9);model=policy()
        optimizer=torch.optim.Adam(model.head.parameters(),lr=.01)
        data=[(torch.zeros(4,19),torch.zeros(4,2))]
        cache=cache_demonstrations(model,data)
        fit_cached(model,optimizer,cache,updates=2)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'trial.pt'
            save_trial(path,model,optimizer,data,rounds=1)
            fit_cached(model,optimizer,cache,updates=3)
            restored,opt,replay,_=load_trial(path)
            fit_cached(restored,opt,cache_demonstrations(restored,replay),updates=3)
            for actual,expected in zip(restored.head.parameters(),model.head.parameters()):
                self.assertTrue(torch.equal(actual,expected))
