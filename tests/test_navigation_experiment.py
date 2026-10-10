import tempfile
import unittest
from pathlib import Path
import torch

from fly_brain_rc.imitation.model import RecurrentPolicy, load_checkpoint, save_checkpoint


class NavigationExperimentTests(unittest.TestCase):
    def test_opening_loss_prioritizes_initial_decisions_and_ignores_padding(self):
        from scripts.train_live_core import sequence_errors
        predicted=torch.zeros(2,20,2)
        labels=torch.zeros_like(predicted);labels[:,:,0]=.5
        predicted[:]=labels
        predicted[0,0,1]=1.;predicted[1,10,1]=1.
        lengths=torch.tensor([12,12])
        original=sequence_errors(predicted,labels,lengths)
        focused=sequence_errors(predicted,labels,lengths,opening_weight=10.)
        self.assertEqual(float(original[0]),float(original[1]))
        self.assertGreater(float(focused[0]),float(focused[1])*5)
        predicted[:,12:]=100.
        self.assertTrue(torch.equal(focused,sequence_errors(predicted,labels,lengths,opening_weight=10.)))

    def test_experimental_optimizer_roundtrips_without_changing_frozen_core(self):
        from scripts.improve_navigation import build_optimizer
        graph = dict(ids=['a','b'], edges=[[0,1,2],[1,0,3]])
        model = RecurrentPolicy(graph, motor_head='residual32')
        before = {n:p.detach().clone() for n,p in model.named_parameters() if not p.requires_grad}
        optimizer = build_optimizer(model, 1e-4)
        x = torch.rand(1,4,19)
        y, _ = model(x)
        optimizer.zero_grad(); (y-.4).square().mean().backward(); optimizer.step()
        for n,p in model.named_parameters():
            if n in before:
                self.assertTrue(torch.equal(p,before[n]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'candidate.pt'
            save_checkpoint(path, model, optimizer, [], 144)
            restored, opt, _, _ = load_checkpoint(path,graph)
        self.assertTrue(torch.equal(model(x)[0],restored(x)[0]))
        self.assertEqual(len(opt.state),len(optimizer.state))

    def test_residual_blending_preserves_core_and_rejects_changed_anatomical_weights(self):
        from scripts.improve_navigation import blend_residual
        graph = dict(ids=['a','b'], edges=[[0,1,2],[1,0,3]])
        model = RecurrentPolicy(graph, motor_head='residual32')
        base = {k:v.clone() for k,v in model.state_dict().items()}
        changed = {k:v.clone() for k,v in base.items()}
        changed['readout.residual.2.bias'] += 2
        mixed = blend_residual(base, changed, .25)
        self.assertTrue(torch.equal(mixed['readout.residual.2.bias'], base['readout.residual.2.bias']+.5))
        self.assertTrue(torch.equal(mixed['encoder.weight'],base['encoder.weight']))
        changed['recurrent'] += .01
        with self.assertRaises(ValueError):
            blend_residual(base, changed, .25)

    def test_mixed_fitting_respects_tier_weight_and_keeps_core_frozen(self):
        from scripts.train_mixed_curriculum import fit_mixture
        torch.set_num_threads(2)
        graph = dict(ids=['a','b'],edges=[[0,1,2],[1,0,3]])
        signs=[]
        for fraction in (.2,.8):
            torch.manual_seed(7)
            model=RecurrentPolicy(graph,motor_head='residual32')
            frozen={n:p.clone() for n,p in model.named_parameters() if not p.requires_grad}
            h=torch.tensor([[1.,0.]]).repeat(8,1)
            replay=[dict(tier='simple',h=h,y=torch.tensor([[.6,0.]]).repeat(8,1)),
                    dict(tier='complex',h=h,y=torch.tensor([[-.6,0.]]).repeat(8,1))]
            anchor=torch.zeros(8,2)
            anchors=(anchor,torch.tanh(model.readout(anchor)).detach())
            optimizer=torch.optim.Adam(model.parameters(),lr=.01)
            fit_mixture(model,optimizer,replay,anchors,complex_fraction=fraction,updates=150)
            signs.append(float(torch.tanh(model.readout(h))[0,0].detach()))
            for n,p in model.named_parameters():
                if n in frozen:self.assertTrue(torch.equal(p,frozen[n]))
        self.assertGreater(signs[0],.15)
        self.assertLess(signs[1],-.15)

    def test_full_core_training_scope_roundtrips_and_preserves_connectivity_mask(self):
        graph=dict(ids=['a','b','c'],edges=[[0,1,2],[1,2,3],[2,0,4]])
        model=RecurrentPolicy(graph,motor_head='residual32',trainable_core=True)
        self.assertTrue(model.encoder.weight.requires_grad)
        self.assertTrue(model.recurrent.requires_grad)
        mask=model.mask.clone()
        optimizer=torch.optim.Adam(model.parameters(),lr=.01)
        x=torch.rand(2,8,19)
        before=model.encoder.weight.detach().clone()
        optimizer.zero_grad();(model(x)[0]-.5).square().mean().backward();optimizer.step()
        self.assertFalse(torch.equal(before,model.encoder.weight))
        self.assertTrue(torch.equal(mask,model.mask))
        self.assertTrue(torch.equal(model.effective_weights()[mask==0],torch.zeros_like(model.effective_weights()[mask==0])))
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'all.pt';save_checkpoint(path,model,optimizer,[],144)
            restored,_,_,_=load_checkpoint(path,graph)
        self.assertTrue(restored.trainable_core)
        self.assertTrue(restored.recurrent.requires_grad)
        self.assertTrue(torch.equal(restored(x)[0],model(x)[0]))
        legacy=RecurrentPolicy(graph,motor_head='residual32')
        self.assertFalse(legacy.recurrent.requires_grad)

    def test_live_full_sequence_fitting_learns_without_creating_connections(self):
        from scripts.train_live_core import fit_live_sequences
        graph=dict(ids=['a','b','c'],edges=[[0,1,2],[1,2,3],[2,0,4]])
        torch.manual_seed(42)
        model=RecurrentPolicy(graph,motor_head='residual32',trainable_core=True)
        x=torch.zeros(12,19);x[:,0]=1;x[:,6:12]=1
        y=torch.tensor([[.5,-.3]]).repeat(12,1)
        episodes=[dict(x=x,y=y,tier='simple'),dict(x=x,y=y,tier='complex')]
        anchors=[(x,model(x[None])[0][0].detach())]
        before=float((model(x[None])[0][0]-y).square().mean().detach())
        encoder=model.encoder.weight.detach().clone();mask=model.mask.clone()
        optimizer=torch.optim.Adam(model.parameters(),lr=.01)
        fit_live_sequences(model,optimizer,episodes,anchors,updates=30,anchor_weight=.01)
        after=float((model(x[None])[0][0]-y).square().mean().detach())
        self.assertLess(after,before/3)
        self.assertFalse(torch.equal(encoder,model.encoder.weight))
        self.assertTrue(torch.equal(mask,model.mask))
        self.assertTrue(torch.all(model.effective_weights()[mask==0]==0))

    def test_live_source_rejects_dense_and_tampered_masks(self):
        from scripts.train_live_core import load_live_source
        graph=dict(ids=['a','b'],edges=[[0,1,2],[1,0,3]])
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'source.pt'
            model=RecurrentPolicy(graph,baseline=True,motor_head='residual32')
            save_checkpoint(path,model,torch.optim.Adam(model.parameters()),[],144)
            with self.assertRaises(ValueError):load_live_source(path,graph)
            model=RecurrentPolicy(graph,motor_head='residual32')
            model.mask.fill_(1)
            save_checkpoint(path,model,torch.optim.Adam(model.parameters()),[],144)
            with self.assertRaises(ValueError):load_live_source(path,graph)
            model=RecurrentPolicy(graph,motor_head='residual32')
            save_checkpoint(path,model,torch.optim.Adam(model.parameters()),[],144)
            restored=load_live_source(path,graph)
            self.assertTrue(restored.trainable_core)
            self.assertTrue(all(p.requires_grad for p in restored.parameters()))

    def test_stronger_cached_source_anchor_reduces_old_output_drift(self):
        import copy
        from scripts.train_mixed_curriculum import fit_mixture
        torch.manual_seed(45);torch.set_num_threads(2)
        graph=dict(ids=['a','b','c'],edges=[[0,1,2],[1,2,3],[2,0,4]])
        source=RecurrentPolicy(graph,motor_head='residual32')
        h=torch.randn(16,3)
        with torch.no_grad():target=source.readout(h).tanh()
        replay=[dict(h=h,y=torch.full((16,2),.8),tier=t) for t in ('simple','complex')]
        drift=[]
        for weight in (1.,32.):
            model=copy.deepcopy(source);opt=torch.optim.Adam(model.parameters(),lr=.01);torch.manual_seed(77)
            fit_mixture(model,opt,replay,(h,target),complex_fraction=.5,updates=150,anchor_weight=weight)
            with torch.no_grad():drift.append(float((model.readout(h).tanh()-target).square().mean()))
        self.assertLess(drift[1],drift[0]/5)

class FullNeuralBlendTests(unittest.TestCase):
    def test_full_blend_preserves_mask_and_rejects_topology_changes(self):
        from scripts.improve_navigation import blend_full_policy
        a={'mask':torch.eye(2),'encoder.weight':torch.zeros(2,3),'recurrent':torch.zeros(2,2)}
        b={k:v.clone() for k,v in a.items()};b['encoder.weight'].add_(2.);b['recurrent'].add_(.2)
        c=blend_full_policy(a,b,.25)
        torch.testing.assert_close(c['encoder.weight'],torch.full((2,3),.5));torch.testing.assert_close(c['mask'],a['mask'],rtol=0,atol=0)
        b['mask'][0,1]=1.
        with self.assertRaises(ValueError):blend_full_policy(a,b,.25)

class RecoveryLossMaskTests(unittest.TestCase):
    def test_prefix_is_context_only_and_padding_has_no_loss(self):
        from scripts.train_live_core import sequence_errors
        p=torch.ones(1,5,2,requires_grad=True);y=torch.zeros_like(p)
        loss=sequence_errors(p,y,torch.tensor([4]),supervision_starts=torch.tensor([2])).mean();loss.backward()
        self.assertTrue(torch.equal(p.grad[:,:2],torch.zeros_like(p.grad[:,:2])))
        self.assertTrue(torch.equal(p.grad[:,4:],torch.zeros_like(p.grad[:,4:])))
        self.assertGreater(float(p.grad[:,2:4].abs().sum()),0.)
        with self.assertRaises(ValueError):sequence_errors(p,y,torch.tensor([4]),supervision_starts=torch.tensor([4]))
