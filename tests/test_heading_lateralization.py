import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests.test_anatomical_heading import fixture
from fly_brain_rc.neural.anatomical_heading import AnatomicalHeading,pilot_parameters


class HeadingLateralizationTests(unittest.TestCase):
    def annotations(self,path,*,missing=False,duplicate=False,unresolved=False):
        with path.open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=['root_id','side'],delimiter='\t')
            writer.writeheader()
            for i in range(1,8 if missing else 9):
                side='unknown' if unresolved and i==1 else ('left' if i%2 else 'right')
                writer.writerow(dict(root_id=str(i),side=side))
            if duplicate:writer.writerow(dict(root_id='1',side='right'))

    def test_split_preserves_topology_and_initial_dynamics(self):
        from fly_brain_rc.neural.heading_lateralization import split_hemisphere_parameters
        a,p=fixture();base=pilot_parameters()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'annotations.tsv';self.annotations(path)
            params=split_hemisphere_parameters(a,base,path)
        old=AnatomicalHeading(a,p,base);new=AnatomicalHeading(a,p,params)
        np.testing.assert_array_equal(old.synapse_counts,new.synapse_counts)
        for _ in range(30):old.step(.01,cue_phase=.3);new.step(.01,cue_phase=.3)
        np.testing.assert_array_equal(old.activity,new.activity)
        params['bias']['EPG:left']=0.
        changed=AnatomicalHeading(a,p,params)
        changed.step(.01)
        self.assertGreater(changed.activity[0],changed.activity[1])

    def test_missing_and_duplicate_annotation_rejected(self):
        from fly_brain_rc.neural.heading_lateralization import split_hemisphere_parameters
        a,_=fixture()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'annotations.tsv'
            for kwargs in (dict(missing=True),dict(duplicate=True),dict(unresolved=True)):
                self.annotations(path,**kwargs)
                with self.assertRaises(ValueError):split_hemisphere_parameters(a,pilot_parameters(),path)

    def test_export_keeps_classes_for_numpy_replay(self):
        from fly_brain_rc.neural.heading_lateralization import split_hemisphere_parameters
        from fly_brain_rc.neural.heading_type_fit import TypeParameterFit
        import torch
        a,p=fixture()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'annotations.tsv';self.annotations(path)
            params=split_hemisphere_parameters(a,pilot_parameters(),path)
        model=AnatomicalHeading(a,p,params);trial=TypeParameterFit(model)
        self.assertIn('EPG:left',trial.kinds)
        with torch.no_grad():
            trial.bias[trial.kinds.index('EPG:left')]=-.7
        replay=AnatomicalHeading(a,p,trial.parameters_dict())
        np.testing.assert_array_equal(replay.synapse_counts,model.synapse_counts)
        state=torch.zeros(1,8)
        for _ in range(50):
            state=trial.step(state,.01,cue_phase=torch.tensor([.3]))
            replay.step(.01,cue_phase=.3)
        np.testing.assert_allclose(state.detach().numpy()[0],replay.activity,atol=2e-6)
        exported=trial.parameters_dict()
        exported['tau']['EPG:left']=1.
        exported['cell_classes']['1']='EPG:right'
        fresh=trial.parameters_dict()
        self.assertEqual(fresh['tau']['EPG:left'],.05)
        self.assertEqual(fresh['cell_classes']['1'],'EPG:left')
