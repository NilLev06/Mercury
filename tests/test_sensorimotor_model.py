import unittest
import torch

from fly_brain_rc.imitation.sensorimotor_model import SensorimotorCore


def graph():
    return dict(ids=['0','1','2'],edges=[[0,1,5],[1,2,5]],
        neurons=[{'top_nt':'acetylcholine'}]*3,
        inputs={'vision':[0]},outputs=[2])


class SensorimotorModelTests(unittest.TestCase):
    def test_output_is_not_directly_driven_and_requires_anatomical_propagation(self):
        core=SensorimotorCore(graph())
        x=torch.ones(1,1,dtype=torch.float64)
        first=core.step(x)
        self.assertEqual(core.readout(first).item(),0.)
        second=core.step(x,first)
        self.assertEqual(core.readout(second).item(),0.)
        third=core.step(x,second)
        self.assertGreater(core.readout(third).item(),0.)
        self.assertFalse(any(p.requires_grad for p in core.parameters()))

    def test_removing_connections_removes_output_signal(self):
        g=graph();g['edges']=[]
        core=SensorimotorCore(g)
        state=None
        for _ in range(10):
            state=core.step(torch.ones(2,1,dtype=torch.float64),state)
        self.assertTrue(torch.equal(core.readout(state),torch.zeros(2,1,dtype=torch.float64)))

    def test_transmitter_sign_changes_downstream_signal(self):
        g=graph();g['neurons']=[{'top_nt':'gaba'},{'top_nt':'acetylcholine'},{}]
        core=SensorimotorCore(g)
        state=None
        for _ in range(10):state=core.step(torch.ones(1,1,dtype=torch.float64),state)
        self.assertLess(core.readout(state).item(),0.)

    def test_rejects_nonfinite_or_wrong_input_shape(self):
        core=SensorimotorCore(graph())
        for x in (torch.zeros(1,2),torch.tensor([[float('nan')]])):
            with self.assertRaises(ValueError):core.step(x)

    def test_loaded_graph_cannot_drive_outputs_directly_or_use_invalid_indices(self):
        for inputs,outputs in [({'v':[2]},[2]),({'v':[-1]},[2]),({'v':[0]},[-1]),
                               ({'v':[9]},[2])]:
            g=graph();g.update(inputs=inputs,outputs=outputs)
            with self.assertRaises(ValueError):SensorimotorCore(g)

    def test_rejects_invalid_weights_metadata_and_annotated_non_dn_output(self):
        for edges in ([[0,1,-1]],[[0,1,float('nan')]],[[0,1,1.5]]):
            g=graph();g['edges']=edges
            with self.assertRaises(ValueError):SensorimotorCore(g)
        g=graph();g['neurons']=[]
        with self.assertRaises(ValueError):SensorimotorCore(g)
        g=graph();g['neurons'][2]={'flow':'intrinsic'}
        with self.assertRaises(ValueError):SensorimotorCore(g)
