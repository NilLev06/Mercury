import tempfile
import unittest
from pathlib import Path


class HeadingExpansionTests(unittest.TestCase):
    def test_input_populations_are_opt_in_and_connections_remain_measured(self):
        from fly_brain_rc.neural.heading_connectome import extract_heading_circuit
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            cells=root/'cells.csv';edges=root/'edges.csv'
            cells.write_text('root_id,primary_type\n1,EPG\n2,PEN_a/PEN1\n3,PEN_b/PEN2\n'
                             '4,PEG\n5,Delta7\n6,GLNO\n7,ER4d\n8,ExR2_1\n9,T4a\n')
            edges.write_text('pre_root_id,post_root_id,neuropil,syn_count,nt_type\n'
                             '1,2,EB,2,ACH\n6,2,NO,3,GABA\n7,1,EB,4,GABA\n'
                             '8,7,EB,5,DA\n9,1,EB,6,ACH\n')
            core=extract_heading_circuit(cells,edges)
            expanded=extract_heading_circuit(cells,edges,include_inputs=True)
        self.assertEqual(len(core['neurons']),5)
        self.assertEqual(len(core['connections']),1)
        self.assertEqual(len(expanded['neurons']),8)
        self.assertEqual(sum(r['syn_count'] for r in expanded['connections']),14)
        self.assertEqual(expanded['connections'][1]['neuropil'],'NO')

    def test_extra_neurons_have_no_invented_phase_and_explicit_signs(self):
        from fly_brain_rc.neural.anatomical_heading import AnatomicalHeading,expanded_parameters,extend_phases
        from tests.test_anatomical_heading import fixture
        a,p=fixture()
        a['neurons'].append(dict(root_id='9',cell_type='ER4d'))
        a['connections'].append(dict(pre='9',post='1',syn_count=3,neuropil='EB',nt_type='GABA'))
        expanded=extend_phases(a,p)
        self.assertIsNone(expanded['neurons'][-1]['phase_degrees'])
        params=expanded_parameters(a)
        self.assertLess(params['gain']['EPG']['ER4d'],0)
        model=AnatomicalHeading(a,expanded,params)
        self.assertEqual(len(model.activity),9)
        self.assertEqual(model.synapse_counts[0,8],3)
