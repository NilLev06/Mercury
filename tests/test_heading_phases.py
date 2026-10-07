import unittest


class HeadingPhaseTests(unittest.TestCase):
    def test_published_epg_map_is_not_naive_linear_pb_order(self):
        from fly_brain_rc.neural.heading_phases import epg_phase
        self.assertEqual(epg_phase('L1'), -11.25)
        self.assertEqual(epg_phase('R1'), 11.25)
        self.assertEqual(epg_phase('L8'), 33.75)
        self.assertEqual(epg_phase('R8'), 56.25)
        self.assertEqual(epg_phase('R2'), -33.75)
        self.assertIsNone(epg_phase('L9'))  # EPGt is outside selected EPG scope.

    def test_only_single_supported_epg_column_gets_hypothetical_phase(self):
        from fly_brain_rc.neural.heading_phases import phase_hypotheses
        neurons = [dict(root_id=str(i), cell_type=kind, status=status,
                        candidate_columns=cols) for i,kind,status,cols in (
            (1,'EPG','morphology_supported_hypothesis',['L3']),
            (2,'EPG','ambiguous',None),
            (3,'PEN_a/PEN1','morphology_supported_hypothesis',['L3']),
            (4,'Delta7','morphology_supported_hypothesis',['L6','R3']))]
        result = phase_hypotheses({'neurons':neurons})
        self.assertEqual(result['neurons'][0]['phase_degrees'], -101.25)
        for neuron in result['neurons'][1:]:
            self.assertIsNone(neuron['phase_degrees'])
        self.assertFalse(result['functional_model'])
        self.assertEqual(result['validated_assignments'], 0)
