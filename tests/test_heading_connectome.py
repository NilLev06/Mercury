import tempfile
import unittest
from pathlib import Path


class HeadingExtractionTests(unittest.TestCase):
    def inputs(self,root):
        cells=root/'cells.csv';edges=root/'edges.csv'
        cells.write_text('root_id,primary_type,additional_type(s)\n'
                         '1,EPG,\n2,PEN_a/PEN1,\n3,PEN_b/PEN2,\n'
                         '4,PEG,\n5,Delta7,\n6,EPGt,\n7,T4a,\n')
        edges.write_text('pre_root_id,post_root_id,neuropil,syn_count,nt_type\n'
                         '1,2,PB,1,ACH\n1,2,EB,3,ACH\n'
                         '2,1,EB,8,ACH\n5,1,PB,4,GLUT\n'
                         '6,1,EB,10,ACH\n7,1,EB,12,ACH\n')
        return cells,edges

    def test_selects_named_types_without_discarding_weak_or_regional_edges(self):
        from fly_brain_rc.neural.heading_connectome import extract_heading_circuit
        with tempfile.TemporaryDirectory() as directory:
            cells,edges=self.inputs(Path(directory))
            result=extract_heading_circuit(cells,edges)
        self.assertEqual([n['root_id'] for n in result['neurons']],['1','2','3','4','5'])
        self.assertEqual(len(result['connections']),4)
        self.assertEqual(result['connections'][0],dict(pre='1',post='2',neuropil='PB',syn_count=1,nt_type='ACH'))
        self.assertEqual(result['type_counts']['EPG'],1)
        self.assertFalse(result['functional_model'])
        self.assertEqual(len(result['sources']['cells_sha256']),64)

    def test_missing_required_population_is_not_silently_accepted(self):
        from fly_brain_rc.neural.heading_connectome import extract_heading_circuit
        with tempfile.TemporaryDirectory() as directory:
            cells,edges=self.inputs(Path(directory))
            cells.write_text('root_id,primary_type\n1,EPG\n')
            with self.assertRaisesRegex(ValueError,'missing'):
                extract_heading_circuit(cells,edges)

    def test_negative_internal_synapse_count_is_rejected(self):
        from fly_brain_rc.neural.heading_connectome import extract_heading_circuit
        with tempfile.TemporaryDirectory() as directory:
            cells,edges=self.inputs(Path(directory))
            edges.write_text('pre_root_id,post_root_id,neuropil,syn_count,nt_type\n1,2,PB,-1,ACH\n')
            with self.assertRaises(ValueError):
                extract_heading_circuit(cells,edges)
