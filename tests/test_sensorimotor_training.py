import tempfile
from pathlib import Path
import unittest
import torch

from fly_brain_rc.imitation.sensorimotor_policy import SensorimotorPolicy
from fly_brain_rc.imitation.sensorimotor_training import cache_demonstrations,fit_cached,save_trial,load_trial


def policy():
    return SensorimotorPolicy(dict(ids=['0','1','2'],edges=[[0,2,5],[1,2,5]],
        neurons=[{}]*3,inputs={'LC10a:left':[0],'LC10a:right':[1]},outputs=[2]))


class SensorimotorTrainingTests(unittest.TestCase):
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
