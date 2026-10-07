import unittest
import json
import tempfile
from pathlib import Path
import numpy as np
import torch

from tests.test_anatomical_heading import fixture
from fly_brain_rc.neural.anatomical_heading import AnatomicalHeading,expanded_parameters


def yaw_fixture():
    a,p=fixture()
    for root in ('9','10'):
        a['neurons'].append(dict(root_id=root,cell_type='GLNO'))
        p['neurons'].append(dict(root_id=root,cell_type='GLNO',phase_degrees=None,status='unresolved'))
    params=expanded_parameters(a)
    original=params.copy()
    classes={n['root_id']:f"{n['cell_type']}:{'left' if int(n['root_id'])%2 else 'right'}"
             for n in a['neurons']}
    base={classes[n['root_id']]:n['cell_type'] for n in a['neurons']}
    params['cell_classes']=classes
    params['gain']={post:{pre:original['gain'][base[post]][base[pre]] for pre in base} for post in base}
    for field in ('bias','tau'):
        params[field]={kind:original[field][typ] for kind,typ in base.items()}
    return a,p,params


class HeadingYawTests(unittest.TestCase):
    def model(self):
        a,p,params=yaw_fixture()
        return AnatomicalHeading(a,p,params)

    def test_drive_targets_only_annotated_glno_sides(self):
        from fly_brain_rc.neural.heading_yaw import GLNOYawFit
        trial=GLNOYawFit(self.model(),input_gain=2.)
        drive=trial.drive(torch.tensor([1.,-1.,0.])).detach().numpy()
        expected=np.zeros((3,10));expected[0,8]=2.;expected[1,9]=2.
        np.testing.assert_array_equal(drive,expected)

    def test_numpy_torch_external_input_parity_and_export(self):
        from fly_brain_rc.neural.heading_yaw import GLNOYawFit,glno_drive
        model=self.model();trial=GLNOYawFit(model,input_gain=2.)
        a,p,_=yaw_fixture();replay=AnatomicalHeading(a,p,trial.parameters_dict())
        state=torch.zeros(1,10)
        for _ in range(50):
            state=trial.step_yaw(state,.01,torch.tensor([.4]),cue_phase=torch.tensor([.2]))
            replay.step(.01,cue_phase=.2,external_drive=glno_drive(replay,.4))
        np.testing.assert_allclose(state.detach().numpy()[0],replay.activity,atol=2e-6)
        np.testing.assert_array_equal(model.synapse_counts,replay.synapse_counts)

    def test_invalid_numpy_input_does_not_mutate_activity(self):
        model=self.model();before=model.activity
        for drive in (np.zeros(9),np.full(10,float('nan'))):
            with self.assertRaises(ValueError):model.step(.01,external_drive=drive)
            np.testing.assert_array_equal(model.activity,before)

    def test_yaw_requires_explicit_sides_and_finite_bounded_rates(self):
        from fly_brain_rc.neural.heading_yaw import GLNOYawFit
        a,p,params=yaw_fixture();params.pop('cell_classes')
        params=expanded_parameters(a)
        with self.assertRaises(ValueError):GLNOYawFit(AnatomicalHeading(a,p,params))
        trial=GLNOYawFit(self.model())
        for velocity in (torch.tensor([float('nan')]),torch.tensor([6.])):
            with self.assertRaises(ValueError):trial.drive(velocity)

    def test_velocity_objective_reaches_sensory_gain_through_measured_edges(self):
        from fly_brain_rc.neural.heading_yaw import GLNOYawFit
        a,p,params=yaw_fixture()
        a['connections'].append(dict(pre='9',post='2',syn_count=100,neuropil='EB',nt_type='GABA'))
        trial=GLNOYawFit(AnatomicalHeading(a,p,params))
        loss=trial.velocity_loss(torch.tensor([.2]),torch.tensor([.4]),cue_steps=10,drive_steps=20)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(abs(float(trial.log_input_gain.grad)),1e-7)
        self.assertFalse(trial.counts.requires_grad)

    def test_velocity_objective_rejects_mismatched_batches(self):
        from fly_brain_rc.neural.heading_yaw import GLNOYawFit
        trial=GLNOYawFit(self.model())
        with self.assertRaises(ValueError):
            trial.velocity_loss(torch.tensor([.2,.3]),torch.tensor([.4]))

    def test_saved_gain_and_side_survive_both_checkpoint_routes(self):
        from fly_brain_rc.neural.heading_yaw import GLNOYawFit,glno_drive
        source=GLNOYawFit(self.model(),input_gain=7.,positive_side='right')
        a,p,_=yaw_fixture();replay=AnatomicalHeading(a,p,source.parameters_dict())
        loaded=GLNOYawFit(replay)
        torch.testing.assert_close(loaded.drive(torch.tensor([1.])),source.drive(torch.tensor([1.])))
        default=GLNOYawFit(self.model());default.load_state_dict(source.state_dict())
        replay2=AnatomicalHeading(a,p,default.parameters_dict())
        np.testing.assert_allclose(glno_drive(replay2,1.),source.drive(torch.tensor([1.])).detach().numpy()[0])

    def test_exported_lower_gain_bound_is_replayable(self):
        from fly_brain_rc.neural.heading_yaw import GLNOYawFit,glno_drive
        trial=GLNOYawFit(self.model(),input_gain=.01)
        a,p,_=yaw_fixture();replay=AnatomicalHeading(a,p,trial.parameters_dict())
        self.assertAlmostEqual(glno_drive(replay,1.)[8],.01)

    def test_speed_weight_penalizes_ignoring_rotation(self):
        from fly_brain_rc.neural.heading_yaw import GLNOYawFit
        trial=GLNOYawFit(self.model())
        args=(torch.tensor([.2]),torch.tensor([.4]))
        low=trial.velocity_loss(*args,cue_steps=10,drive_steps=20,speed_weight=.02)
        high=trial.velocity_loss(*args,cue_steps=10,drive_steps=20,speed_weight=2.)
        self.assertGreater(float(high.detach()),float(low.detach()))
        with self.assertRaises(ValueError):trial.velocity_loss(*args,speed_weight=-1.)

    def test_learned_time_constants_export_and_replay_actual_dynamics(self):
        from fly_brain_rc.neural.heading_yaw import GLNOYawFit,glno_drive
        trial=GLNOYawFit(self.model(),learn_tau=True)
        with torch.no_grad():trial.log_tau[trial.kinds.index('GLNO:left')]=np.log(.1)
        a,p,_=yaw_fixture();replay=AnatomicalHeading(a,p,trial.parameters_dict())
        self.assertAlmostEqual(replay.parameters['tau']['GLNO:left'],.1,places=6)
        state=torch.zeros(1,10)
        for _ in range(30):
            state=trial.step_yaw(state,.01,torch.tensor([.4]))
            replay.step(.01,external_drive=glno_drive(replay,.4))
        np.testing.assert_allclose(state.detach().numpy()[0],replay.activity,atol=2e-6)
        state.sum().backward()
        self.assertGreater(abs(float(trial.log_tau.grad[trial.kinds.index('GLNO:left')])),1e-7)

    def test_training_command_resumes_optimizer_without_overwriting_checkpoint(self):
        from fly_brain_rc.neural.heading_yaw import main
        from fly_brain_rc.neural.heading_columns import digest
        a,p,params=yaw_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);ap=root/'a.json';pp=root/'p.json';cp=root/'initial.pt'
            ap.write_text(json.dumps(a));pp.write_text(json.dumps(p))
            torch.save(dict(parameters=params,root_ids=tuple(str(i) for i in range(1,11)),
                            anatomy_sha256=digest(ap),phases_sha256=digest(pp),steps=0),cp)
            original=digest(cp)
            first=root/'first.pt';second=root/'second.pt'
            common=['--anatomy',str(ap),'--phases',str(pp),'--steps','1']
            main(common+['--checkpoint',str(cp),'--out',str(first)])
            main(common+['--checkpoint',str(first),'--out',str(second),'--resume'])
            saved=torch.load(second,weights_only=True)
            self.assertEqual(saved['yaw_training_steps'],2)
            self.assertEqual(saved['optimizer']['state'][0]['step'],2)
            self.assertEqual(digest(cp),original)
            self.assertFalse(saved['vehicle_integration_allowed'])
            self.assertEqual(saved['steps'],2)
            self.assertIn('yaw',saved['steps_scope'])
            for field in ('synapse_scale','cue_strength'):
                altered=torch.load(first,weights_only=True)
                altered['parameters'][field]*=2
                tampered=root/f'{field}.pt';torch.save(altered,tampered)
                with self.assertRaises(ValueError):
                    main(common+['--checkpoint',str(tampered),'--out',str(root/f'{field}-out.pt'),'--resume'])
            bad=torch.load(cp,weights_only=True);bad['anatomy_sha256']='wrong';torch.save(bad,cp)
            rejected=root/'rejected.pt'
            with self.assertRaises(ValueError):
                main(common+['--checkpoint',str(cp),'--out',str(rejected)])
            self.assertFalse(rejected.exists())
            motion=root/'motion.pt';continued=root/'motion-continued.pt'
            main(common+['--checkpoint',str(first),'--out',str(motion),'--objective','motion'])
            main(common+['--checkpoint',str(motion),'--out',str(continued),'--resume'])
            self.assertEqual(torch.load(continued,weights_only=True)['objective'],'motion')
            with self.assertRaises(ValueError):
                main(common+['--checkpoint',str(motion),'--out',str(root/'changed-objective.pt'),
                             '--resume','--objective','moment'])

    def test_differential_input_opposes_glno_sides_and_survives_export(self):
        from fly_brain_rc.neural.heading_yaw import GLNOYawFit,glno_drive
        trial=GLNOYawFit(self.model(),input_gain=2.,input_mode='differential')
        drive=trial.drive(torch.tensor([1.,-1.,0.])).detach().numpy()
        expected=np.zeros((3,10));expected[0,8:]=[2.,-2.];expected[1,8:]=[-2.,2.]
        np.testing.assert_array_equal(drive,expected)
        a,p,_=yaw_fixture();replay=AnatomicalHeading(a,p,trial.parameters_dict())
        np.testing.assert_array_equal(glno_drive(replay,1.),expected[0])

    def test_motion_objective_scores_speed_without_prescribing_bump_amplitude(self):
        from fly_brain_rc.neural.heading_yaw import motion_trace_loss
        phase=torch.tensor([[3.1],[3.14],[3.18]])
        moments=torch.stack((phase.cos(),phase.sin()),dim=-1)*.2
        means=torch.ones(3,1)*.5
        good=motion_trace_loss(moments,means,torch.tensor([.4]),.1)
        scaled=motion_trace_loss(moments*2,means,torch.tensor([.4]),.1)
        pinned=motion_trace_loss(moments[:1].expand_as(moments),means,torch.tensor([.4]),.1)
        self.assertLess(float(good),1e-8)
        self.assertAlmostEqual(float(good),float(scaled),places=7)
        self.assertAlmostEqual(float(pinned),.16,places=5)

    def test_motion_objective_does_not_reward_collapsed_activity(self):
        from fly_brain_rc.neural.heading_yaw import motion_trace_loss
        moments=torch.zeros(3,1,2);means=torch.ones(3,1)*.5
        self.assertGreater(float(motion_trace_loss(moments,means,torch.tensor([0.]),.1)),.9)
        with self.assertRaises(ValueError):
            motion_trace_loss(moments,means,torch.tensor([0.,1.]),.1)

    def test_motion_training_backpropagates_through_the_fixed_neural_graph(self):
        from fly_brain_rc.neural.heading_yaw import GLNOYawFit
        a,p,params=yaw_fixture()
        a['connections'].append(dict(pre='9',post='2',syn_count=100,neuropil='EB',nt_type='GABA'))
        trial=GLNOYawFit(AnatomicalHeading(a,p,params))
        loss=trial.motion_loss(torch.tensor([.2]),torch.tensor([.4]),cue_steps=10,drive_steps=20)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(abs(float(trial.log_input_gain.grad)),1e-7)
        self.assertFalse(trial.counts.requires_grad)
