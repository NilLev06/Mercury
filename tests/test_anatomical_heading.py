import unittest
import numpy as np


def fixture():
    types = ['EPG']*4 + ['PEN_a/PEN1','PEN_b/PEN2','PEG','Delta7']
    anatomy = {'neurons':[dict(root_id=str(i+1),cell_type=t) for i,t in enumerate(types)],
               'connections':[dict(pre='1',post='5',syn_count=2,neuropil='EB',nt_type='ACH'),
                              dict(pre='1',post='5',syn_count=3,neuropil='PB',nt_type='ACH'),
                              dict(pre='8',post='1',syn_count=7,neuropil='PB',nt_type='ACH')]}
    phases = {'neurons':[dict(root_id=str(i+1),cell_type=t,
                             phase_degrees=angle if t=='EPG' else None,
                             status='phase_hypothesis' if t=='EPG' else 'unresolved')
                        for i,(t,angle) in enumerate(zip(types,[0,90,180,-90,None,None,None,None]))]}
    return anatomy,phases


class AnatomicalHeadingTests(unittest.TestCase):
    def model(self,**kwargs):
        from fly_brain_rc.neural.anatomical_heading import AnatomicalHeading,pilot_parameters
        a,p = fixture()
        return AnatomicalHeading(a,p,pilot_parameters(),**kwargs)

    def test_directed_counts_aggregate_without_adding_edges(self):
        model=self.model()
        self.assertEqual(model.synapse_counts[4,0],5)
        self.assertEqual(model.synapse_counts[0,7],7)
        self.assertEqual(model.synapse_counts[0,4],0)
        self.assertEqual(np.count_nonzero(model.synapse_counts),2)
        view=model.synapse_counts;view[4,0]=99
        self.assertEqual(model.synapse_counts[4,0],5)

    def test_uniform_epg_activity_is_not_a_heading(self):
        model=self.model(recurrence=False)
        for _ in range(100):model.step(.01)
        self.assertIsNone(model.heading)
        self.assertLess(model.contrast,1e-8)

    def test_cue_is_encoded_by_population_and_lesion_erases_it(self):
        model=self.model(recurrence=False)
        for _ in range(100):model.step(.01,cue_phase=.2)
        self.assertIsNotNone(model.heading)
        self.assertLess(abs(model.heading-.2),.25)
        self.assertGreater(model.contrast,.1)
        for _ in range(200):model.step(.01)
        self.assertIsNone(model.heading)

    def test_sign_is_explicit_hypothesis_not_nt_label(self):
        from fly_brain_rc.neural.anatomical_heading import AnatomicalHeading,pilot_parameters
        a,p=fixture()
        inhibitory=AnatomicalHeading(a,p,pilot_parameters(delta7_gain=-10))
        excitatory=AnatomicalHeading(a,p,pilot_parameters(delta7_gain=10))
        for _ in range(100):
            inhibitory.step(.01);excitatory.step(.01)
        self.assertLess(inhibitory.activity[0],excitatory.activity[0])

    def test_invalid_input_rejected_before_state_changes(self):
        model=self.model();before=model.activity
        for dt,cue in ((0,None),(float('nan'),None),(.01,float('nan'))):
            with self.assertRaises(ValueError):model.step(dt,cue_phase=cue)
            np.testing.assert_array_equal(model.activity,before)

    def test_bad_edge_and_phase_identity_rejected(self):
        from fly_brain_rc.neural.anatomical_heading import AnatomicalHeading,pilot_parameters
        a,p=fixture();a['connections'][0]['pre']='99'
        with self.assertRaises(ValueError):AnatomicalHeading(a,p,pilot_parameters())
        a,p=fixture();p['neurons'][0]['cell_type']='PEG'
        with self.assertRaises(ValueError):AnatomicalHeading(a,p,pilot_parameters())

    def test_bounded_finite_and_timestep_consistent(self):
        models=[self.model(),self.model()]
        for model,dt in zip(models,(.01,.02)):
            for _ in range(round(1/dt)):model.step(dt,cue_phase=.37)
            self.assertTrue(np.isfinite(model.activity).all())
            self.assertTrue(((model.activity>=0)&(model.activity<=1)).all())
        np.testing.assert_allclose(models[0].activity,models[1].activity,atol=1e-10)

    def test_circularly_equivalent_phases_do_not_inflate_coverage(self):
        from fly_brain_rc.neural.anatomical_heading import AnatomicalHeading,pilot_parameters
        a,p=fixture();p['neurons'][1]['phase_degrees']=360.
        with self.assertRaises(ValueError):AnatomicalHeading(a,p,pilot_parameters())

    def test_returned_parameter_metadata_cannot_be_mutated(self):
        model=self.model();view=model.parameters
        view['gain']['EPG']['Delta7']=99
        self.assertEqual(model.parameters['gain']['EPG']['Delta7'],-4.)
