import unittest

from fly_brain_rc.imitation.sensorimotor_probe import evaluate_channels


class SensorimotorProbeTests(unittest.TestCase):
    def test_signal_recovery_requires_connections_not_direct_access_to_inputs(self):
        g=dict(ids=['0','1'],edges=[[0,1,5]],inputs={'vision':[0]},
               outputs=[1],neurons=[{'top_nt':'acetylcholine'}]*2)
        result=evaluate_channels(g,samples=32,steps=16)
        self.assertGreater(result['intact']['vision']['r2'],.95)
        self.assertLess(result['disconnected']['vision']['r2'],.1)
        self.assertFalse(result['vehicle_integration_allowed'])

    def test_rejects_invalid_trial_settings(self):
        with self.assertRaises(ValueError):evaluate_channels({},samples=1)

