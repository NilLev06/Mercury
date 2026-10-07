import unittest
import numpy as np
import torch


class HeadingTypeFitTests(unittest.TestCase):
    def model(self):
        from tests.test_anatomical_heading import fixture
        from fly_brain_rc.neural.anatomical_heading import AnatomicalHeading,pilot_parameters
        a,p=fixture()
        return AnatomicalHeading(a,p,pilot_parameters())

    def test_torch_dynamics_match_numpy_and_keep_measured_zeros(self):
        from fly_brain_rc.neural.heading_type_fit import TypeParameterFit
        model=self.model();trial=TypeParameterFit(model)
        state=torch.zeros(1,len(model.activity))
        for _ in range(20):
            state=trial.step(state,.01,cue_phase=torch.tensor([.3]))
            model.step(.01,cue_phase=.3)
        np.testing.assert_allclose(state.detach().numpy()[0],model.activity,atol=2e-6)
        self.assertTrue(torch.all(trial.weights()[torch.tensor(model.synapse_counts==0)]==0))

    def test_only_type_parameters_change_during_fitting(self):
        from fly_brain_rc.neural.heading_type_fit import TypeParameterFit
        model=self.model();trial=TypeParameterFit(model)
        initial=trial.parameters_dict()
        counts=trial.counts.clone()
        optimizer=torch.optim.Adam(trial.parameters(),lr=.02)
        phases=torch.tensor([.2,1.2,2.2])
        before=float(trial.memory_loss(phases,cue_steps=10,dark_steps=20).detach())
        for _ in range(15):
            optimizer.zero_grad()
            loss=trial.memory_loss(phases,cue_steps=10,dark_steps=20)
            loss.backward();optimizer.step()
        after=float(trial.memory_loss(phases,cue_steps=10,dark_steps=20).detach())
        self.assertLess(after,before)
        self.assertNotEqual(trial.parameters_dict(),initial)
        self.assertTrue(torch.equal(trial.counts,counts))
        self.assertFalse(trial.counts.requires_grad)

    def test_exported_parameters_are_accepted_by_numpy_network(self):
        from fly_brain_rc.neural.heading_type_fit import TypeParameterFit
        from fly_brain_rc.neural.anatomical_heading import AnatomicalHeading
        from tests.test_anatomical_heading import fixture
        trial=TypeParameterFit(self.model());a,p=fixture()
        model=AnatomicalHeading(a,p,trial.parameters_dict())
        self.assertEqual(model.synapse_counts.sum(),12)

    def test_recurrence_lesion_matches_numpy(self):
        from fly_brain_rc.neural.heading_type_fit import TypeParameterFit
        model=self.model()
        model.recurrence=False
        trial=TypeParameterFit(model)
        state=torch.zeros(1,len(model.activity))
        for _ in range(50):
            state=trial.step(state,.01,cue_phase=torch.tensor([.3]))
            model.step(.01,cue_phase=.3)
        np.testing.assert_allclose(state.detach().numpy()[0],model.activity,atol=2e-6)

    def test_profile_loss_detects_distortion_hidden_by_first_moment(self):
        from fly_brain_rc.neural.heading_type_fit import TypeParameterFit
        trial=TypeParameterFit(self.model())
        state=torch.full((1,8),.5)
        state[0,:4]=torch.tensor([.8,.5,.2,.5])
        clean=trial.profile_loss(state,torch.tensor([0.]))
        distorted=state.clone()
        distorted[0,:4]+=torch.tensor([.1,-.1,.1,-.1])
        self.assertLess(float(clean),1e-10)
        self.assertGreater(float(trial.profile_loss(distorted,torch.tensor([0.]))),.009)

    def test_equilibrium_residual_matches_observed_rate_change(self):
        from fly_brain_rc.neural.heading_type_fit import TypeParameterFit
        trial=TypeParameterFit(self.model())
        state=torch.full((1,8),.4,requires_grad=True)
        residual=trial.equilibrium_residual(state)
        following=trial.step(state,.001)
        torch.testing.assert_close(residual,(following-state)*trial.tau/.001)
        residual.square().mean().backward()
        self.assertTrue(torch.isfinite(trial.log_gain.grad).all())

    def test_equilibrium_probe_rejects_invalid_activity(self):
        from fly_brain_rc.neural.heading_type_fit import TypeParameterFit
        trial=TypeParameterFit(self.model())
        for value in (-.1,1.1,float('nan')):
            with self.assertRaises(ValueError):
                trial.equilibrium_residual(torch.full((1,8),value))
