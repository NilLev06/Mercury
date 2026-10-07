import unittest
import numpy as np
import torch

from tests.test_heading_yaw import yaw_fixture
from fly_brain_rc.neural.anatomical_heading import AnatomicalHeading


class HeadingManifoldTests(unittest.TestCase):
    def model(self):
        from fly_brain_rc.neural.heading_manifold import HeadingManifoldFit
        a,p,parameters=yaw_fixture()
        return HeadingManifoldFit(AnatomicalHeading(a,p,parameters),harmonics=2)

    def test_curve_is_periodic_and_tangent_matches_finite_difference(self):
        trial=self.model()
        with torch.no_grad():trial.curve_coefficients[1].fill_(.3)
        phase=torch.tensor([.3],dtype=torch.float64)
        trial=trial.double()
        state,tangent=trial.curve(phase)
        repeated,_=trial.curve(phase+2*np.pi)
        plus,_=trial.curve(phase+1e-5);minus,_=trial.curve(phase-1e-5)
        torch.testing.assert_close(state,repeated)
        torch.testing.assert_close(tangent,(plus-minus)/2e-5,atol=1e-9,rtol=1e-7)

    def test_neutrality_objective_backpropagates_to_type_gains(self):
        trial=self.model()
        loss=trial.curve_loss(torch.tensor([-.3,.7,2.]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(trial.log_gain.grad.abs().sum()),0.)
        self.assertFalse(trial.counts.requires_grad)

    def test_curve_is_never_injected_into_exported_runtime_dynamics(self):
        trial=self.model()
        before=trial.parameters_dict()
        with torch.no_grad():trial.curve_coefficients.add_(1.)
        self.assertEqual(before,trial.parameters_dict())
        self.assertNotIn('curve_coefficients',before)

    def test_invalid_phases_and_initializer_rejected(self):
        trial=self.model()
        with self.assertRaises(ValueError):trial.curve(torch.tensor([float('nan')]))
        with self.assertRaises(ValueError):trial.initialize_curve(torch.tensor([0.]),torch.zeros(1,10))

    def test_silent_curve_cannot_claim_noncollapsed_activity(self):
        trial=self.model()
        with torch.no_grad():
            trial.curve_coefficients.zero_();trial.curve_coefficients[0].fill_(-20.)
        terms=trial.curve_terms(torch.tensor([-.3,.7,2.]))
        self.assertAlmostEqual(float(terms['noncollapse'].detach()),.0225,places=6)

    def test_cue_release_loss_requires_dynamics_to_reach_auxiliary_curve(self):
        trial=self.model()
        phases=torch.tensor([-.3,.7,2.])
        loss=trial.release_loss(phases,cue_steps=10,dark_steps=20)
        self.assertGreater(float(loss.detach()),0.)
        loss.backward()
        self.assertGreater(float(trial.bias.grad.abs().sum()),0.)
        self.assertGreater(float(trial.curve_coefficients.grad.abs().sum()),0.)

    def test_drift_term_matches_actual_short_step_and_time_constants(self):
        trial=self.model().double();phases=torch.tensor([-.3,.7,2.],dtype=torch.float64)
        state,_=trial.curve(phases)
        def angle(x):
            rates=x@trial.groups
            moment=(rates-rates.mean(1,keepdim=True))@trial.moments
            return torch.atan2(moment[:,1],moment[:,0])
        before=angle(state);after=angle(trial.step(state,1e-7))
        measured=((after-before)/1e-7).square().mean()
        drift=trial.curve_terms(phases)['drift']
        torch.testing.assert_close(drift,measured,atol=1e-4,rtol=1e-4)
        with torch.no_grad():trial.tau.mul_(2.)
        torch.testing.assert_close(trial.curve_terms(phases)['drift'],drift/4)

    def test_optional_individual_thresholds_export_and_replay_without_rewiring(self):
        from fly_brain_rc.neural.heading_manifold import HeadingManifoldFit
        from fly_brain_rc.neural.heading_yaw import GLNOYawFit
        a,p,parameters=yaw_fixture();original=AnatomicalHeading(a,p,parameters)
        trial=HeadingManifoldFit(original,learn_neuron_bias=True)
        with torch.no_grad():trial.neuron_offsets[0]=.7
        exported=trial.parameters_dict();replay=AnatomicalHeading(a,p,exported)
        state=torch.zeros(1,10)
        for _ in range(20):
            state=trial.step(state,.01,cue_phase=torch.tensor([.2]))
            replay.step(.01,cue_phase=.2)
        np.testing.assert_allclose(state.detach().numpy()[0],replay.activity,atol=2e-6)
        np.testing.assert_array_equal(original.synapse_counts,replay.synapse_counts)
        loaded=GLNOYawFit(AnatomicalHeading(a,p,exported))
        torch.testing.assert_close(loaded.step(torch.zeros(1,10),.01),trial.step(torch.zeros(1,10),.01))

    def test_individual_thresholds_require_exact_identity_join(self):
        a,p,parameters=yaw_fixture();parameters['neuron_bias_offsets']={'1':.7}
        with self.assertRaises(ValueError):AnatomicalHeading(a,p,parameters)
        parameters['neuron_bias_offsets']={str(i):0. for i in range(1,11)}
        parameters['neuron_bias_offsets']['1']=float('nan')
        with self.assertRaises(ValueError):AnatomicalHeading(a,p,parameters)

    def test_release_phase_penalty_detects_error_hidden_by_average_state_loss(self):
        trial=self.model();phases=torch.tensor([-.3,.7,2.])
        state_only=trial.release_loss(phases,cue_steps=10,dark_steps=20)
        phase_aware=trial.release_loss(phases,cue_steps=10,dark_steps=20,phase_weight=1.)
        self.assertGreater(float(phase_aware.detach()),float(state_only.detach()))
        with self.assertRaises(ValueError):trial.release_loss(phases,phase_weight=-1.)


if __name__=='__main__':unittest.main()
