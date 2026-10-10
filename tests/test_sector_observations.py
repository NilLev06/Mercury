import unittest
from dataclasses import replace
from pathlib import Path
import tempfile
import numpy as np
import torch
from fly_brain_rc.contracts import SensorFrame
from fly_brain_rc.imitation.observations import SCHEMA,SECTOR_SCHEMA,encode_observation
from fly_brain_rc.imitation.model import RecurrentPolicy,load_checkpoint,save_checkpoint,upgrade_sector_inputs

class SectorObservationTests(unittest.TestCase):
    def test_experimental_resume_inherits_scope_unless_explicitly_overridden(self):
        from scripts.train_live_core import resolve_fit_scope
        model=RecurrentPolicy(dict(ids=['a'],edges=[]))
        self.assertEqual(resolve_fit_scope(model),'all')
        model.training_metadata['fixed_replay_refinement']={'scope':'sectors-residual'}
        self.assertEqual(resolve_fit_scope(model),'sectors-residual')
        self.assertEqual(resolve_fit_scope(model,'all'),'all')
        model.training_metadata['fit_scope']='all'
        self.assertEqual(resolve_fit_scope(model),'all')
        with self.assertRaises(ValueError):resolve_fit_scope(model,'unknown')

    def test_sector_only_fitting_keeps_legacy_parameters_even_with_optimizer_momentum(self):
        from scripts.train_live_core import fit_live_sequences
        graph=dict(ids=['a','b','c'],edges=[[0,1,2],[1,2,3],[2,0,4]])
        torch.manual_seed(13);torch.set_num_threads(2)
        model=upgrade_sector_inputs(RecurrentPolicy(graph,motor_head='residual32'))
        optimizer=torch.optim.Adam(model.parameters(),lr=.01)
        x=torch.rand(12,31);y=torch.tensor([[.4,-.5]]).repeat(12,1)
        optimizer.zero_grad();model(x[None])[0].square().mean().backward();optimizer.step()
        before={n:p.detach().clone() for n,p in model.named_parameters()}
        episodes=[dict(x=x,y=y,tier=t) for t in ('simple','complex')]
        anchors=[(x,model(x[None])[0][0].detach())]
        fit_live_sequences(model,optimizer,episodes,anchors,updates=10,anchor_weight=.01,scope='sectors-residual')
        for n,p in model.named_parameters():
            if n=='encoder.weight':
                self.assertTrue(torch.equal(p[:,:19],before[n][:,:19]))
                self.assertFalse(torch.equal(p[:,19:],before[n][:,19:]))
            elif not n.startswith('readout.residual.'):
                self.assertTrue(torch.equal(p,before[n]),n)

    def frame(self,rgb):
        return SensorFrame(rgb=rgb,gray=np.zeros(rgb.shape[:2],dtype=np.uint8),yaw_rate=0.,encoder_delta=0.,tof=1.,contact=False,timestamp=0.)

    def test_sectors_keep_all_components_and_original_inputs(self):
        rgb=np.zeros((60,120,3),dtype=np.uint8)
        rgb[10:50,2:18]=(0,0,255);rgb[10:40,101:117]=(0,0,255)
        frame=self.frame(rgb)
        original=encode_observation(frame,None,None)
        richer=encode_observation(frame,None,None,schema=SECTOR_SCHEMA)
        np.testing.assert_array_equal(original,richer[:19])
        self.assertEqual(richer.shape,(31,))
        self.assertGreater(richer[19],0)
        self.assertGreater(richer[-1],0)
        self.assertEqual(float(richer[25]),0.)
        self.assertTrue(np.all((richer[19:]>=0)&(richer[19:]<=1)))
        with self.assertRaises(ValueError):encode_observation(frame,None,None,schema='unknown')

    def test_zero_column_upgrade_keeps_outputs_states_and_roundtrips(self):
        graph=dict(ids=['a','b','c'],edges=[[0,1,2],[1,2,3],[2,0,4]])
        torch.manual_seed(8)
        source=RecurrentPolicy(graph,motor_head='residual32')
        source.training_metadata['cases']=['example']
        richer=upgrade_sector_inputs(source)
        x=torch.rand(2,20,19);full=torch.cat((x,torch.rand(2,20,12)),dim=-1)
        a,h=source(x);b,g=richer(full)
        self.assertTrue(torch.equal(a,b));self.assertTrue(torch.equal(h,g))
        self.assertEqual(source.encoder.in_features,19)
        self.assertTrue(torch.equal(source.mask,richer.mask))
        self.assertTrue(richer.trainable_core)
        self.assertEqual(richer.training_metadata['cases'],['example'])
        richer.training_metadata['cases'].append('independent')
        self.assertEqual(source.training_metadata['cases'],['example'])
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'rich.pt'
            save_checkpoint(path,richer,torch.optim.Adam(richer.parameters()),[],144)
            restored,_,_,_=load_checkpoint(path,graph)
        self.assertEqual(restored.observation_schema,SECTOR_SCHEMA)
        self.assertTrue(torch.equal(restored(full)[0],b))
        self.assertEqual(source.observation_schema,SCHEMA)
