import unittest
import json
import tempfile
from pathlib import Path


class HeadingBenchTests(unittest.TestCase):
    def test_annotation_join_preserves_large_ids_and_exposes_missing_sectors(self):
        from fly_brain_rc.neural.heading_bench import annotation_audit
        with tempfile.TemporaryDirectory() as directory:
            anatomy=Path(directory)/'anatomy.json';annotations=Path(directory)/'annotations.tsv'
            anatomy.write_text(json.dumps({'neurons':[{'root_id':'720575940624879678'}]}))
            annotations.write_text('root_id\themibrain_type\tside\ttop_nt\n'
                                   '720575940624879678\tEPG\tleft\tacetylcholine\n'
                                   '720575940624879679\tPEG\tright\tacetylcholine\n')
            result=annotation_audit(anatomy,annotations)
            self.assertEqual(result['matched'],1)
            self.assertEqual(result['type_counts'],{'EPG':1})
            self.assertFalse(result['sector_assignment_available'])
            annotations.write_text('root_id\themibrain_type\tside\ttop_nt\n')
            with self.assertRaisesRegex(ValueError,'missing'):
                annotation_audit(anatomy,annotations)

    def test_report_distinguishes_synthetic_function_from_anatomy(self):
        from fly_brain_rc.neural.heading_bench import run_bench
        report=run_bench(phases=(.2,),velocities=(-.2,.2))
        self.assertFalse(report['anatomical_reconstruction'])
        self.assertEqual(len(report['turns']),2)
        self.assertLess(report['dark_hold']['max_error_degrees'],1.)
        self.assertLess(report['reanchor']['max_error_degrees'],2.)
        self.assertLess(report['ablations']['without_recurrence_contrast'],.001)
        self.assertGreater(report['ablations']['without_turn_input_error_degrees'],50.)
