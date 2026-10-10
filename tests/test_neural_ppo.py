import math,tempfile,unittest
from pathlib import Path
import torch
from fly_brain_rc.imitation.model import RecurrentPolicy,save_checkpoint,load_checkpoint

class NeuralPPOTests(unittest.TestCase):
    def test_transformed_density_is_finite_and_unchanged_policy_ratio_is_one(self):
        from scripts.train_neural_ppo import squashed_log_prob
        z=torch.tensor([[0.,.8],[10.,-10.]])
        mean=torch.tensor([[.1,-.2],[9.,-9.]])
        std=torch.tensor([.1,.3])
        logp=squashed_log_prob(z,mean,std)
        self.assertTrue(torch.isfinite(logp).all())
        self.assertTrue(torch.equal((logp-logp).exp(),torch.ones(2)))
        expected=(torch.distributions.Normal(mean[0],std).log_prob(z[0])-torch.log1p(-z[0].tanh().square())).sum()
        self.assertAlmostEqual(float(logp[0]),float(expected),places=5)

    def test_clipping_handles_both_advantage_signs(self):
        from scripts.train_neural_ppo import clipped_loss
        new=torch.tensor([math.log(2.),math.log(.5)])
        old=torch.zeros(2);adv=torch.tensor([1.,-1.])
        loss=clipped_loss(new,old,adv)
        torch.testing.assert_close(loss,torch.tensor([-1.2,.8]))

    def test_gae_terminal_and_truncated_end_do_not_cross_episode_boundary(self):
        from scripts.train_neural_ppo import advantages
        reward=torch.tensor([1.,2.]);value=torch.tensor([.5,.7])
        adv,returns=advantages(reward,value,bootstrap=10.,terminal=True,gamma=.9,lam=1.)
        torch.testing.assert_close(returns,torch.tensor([2.8,2.]))
        adv2,returns2=advantages(reward,value,bootstrap=10.,terminal=False,gamma=.9,lam=1.)
        torch.testing.assert_close(returns2,torch.tensor([10.9,11.]))

    def test_reward_counts_contact_seconds_and_terminal_potential(self):
        from scripts.train_neural_ppo import transition_reward
        reward,parts=transition_reward(2.,1.9,reached=False,contact=False,contact_seconds=.04,elapsed=.1,gamma=.995)
        self.assertAlmostEqual(parts['contact_duration'],-.008)
        self.assertAlmostEqual(parts['potential'],5*(2.-.995*1.9))
        self.assertAlmostEqual(reward,sum(parts.values()))
        win,wp=transition_reward(.44,.43,reached=True,contact=False,contact_seconds=0.,elapsed=.1,gamma=.995)
        hit,hp=transition_reward(2.,2.,reached=False,contact=True,contact_seconds=.02,elapsed=.02,gamma=.995)
        self.assertEqual(wp['arrival'],25.)
        self.assertEqual(hp['collision'],-10.)
        self.assertAlmostEqual(wp['potential'],2.2)
        self.assertAlmostEqual(hp['potential'],10.)

    def test_actor_update_uses_frozen_cached_brain_states_and_roundtrips(self):
        from scripts.train_neural_ppo import freeze_actor,squashed_log_prob,clipped_loss
        torch.set_num_threads(2);torch.manual_seed(42)
        graph=dict(ids=['a','b','c'],edges=[[0,1,2],[1,2,3],[2,0,4]])
        model=RecurrentPolicy(graph,motor_head='residual32',trainable_core=True)
        freeze_actor(model)
        x=torch.rand(1,10,19);before={n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
        states=[];h=None;outputs=[]
        with torch.no_grad():
            full,_=model(x)
            for frame in x.unbind(1):
                y,h=model(frame[:,None],h);states.append(h[0].clone());outputs.append(y[0,0])
        torch.testing.assert_close(full[0],torch.stack(outputs))
        hs=torch.stack(states);std=torch.tensor([.1,.3]);z=torch.atanh(full[0].clamp(-.99,.99))+.05
        old=squashed_log_prob(z,model.readout(hs),std).detach()
        opt=torch.optim.Adam(model.parameters(),lr=.001)
        opt.zero_grad();clipped_loss(squashed_log_prob(z,model.readout(hs),std),old,torch.ones(10)).mean().backward();opt.step()
        for n,p in model.named_parameters():
            if n in before:self.assertTrue(torch.equal(p,before[n]),n)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'actor.pt';save_checkpoint(path,model,opt,[],144)
            restored,_,_,_=load_checkpoint(path,graph)
        self.assertFalse(restored.trainable_core)
        self.assertTrue(torch.equal(restored(x)[0],model(x)[0]))

    def test_actual_collector_bootstraps_cutoff_and_stops_on_contact(self):
        from unittest.mock import patch
        from scripts.train_neural_ppo import rollout
        class Env:
            renderer=None
            def __init__(self,*args,**kwargs):self.used=0;self.collisions=0;self._contact_active=False;self.speed=0.
            def step(self,*args,**kwargs):
                self.used+=1
                if hit and self.used==3:self._contact_active=True;self.collisions=1
        class Value(torch.nn.Module):
            def forward(self,h):return h.new_tensor(2.)
        graph=dict(ids=['a','b','c'],edges=[[0,1,2],[1,2,3],[2,0,4]])
        model=RecurrentPolicy(graph,motor_head='residual32');case=type('Case',(),{'name':'mock'})()
        for hit in (False,True):
            with patch('scripts.train_neural_ppo.FlyCarEnv',Env),patch('scripts.train_neural_ppo.place_start'),patch('scripts.train_neural_ppo.body_goal_distance',return_value=2.),patch('scripts.train_neural_ppo.brain_step',return_value=torch.zeros(1,3)):
                data,report=rollout(model,Value(),case,(0,0,0),torch.tensor([.1,.3]),steps=6)
            if hit:
                self.assertEqual(report['steps'],3);self.assertEqual(len(data['h']),1)
                self.assertTrue(report['terminal']);self.assertAlmostEqual(float(data['returns'][-1]),-.0052,places=5)
            else:
                self.assertEqual(report['steps'],6);self.assertEqual(len(data['h']),2)
                self.assertTrue(report['truncated']);self.assertAlmostEqual(float(data['returns'][-1]),2.0396,places=5)
