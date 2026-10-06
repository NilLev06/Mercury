import unittest
import math
from dataclasses import replace
import numpy as np
import tempfile
from pathlib import Path


class ArrivalTests(unittest.TestCase):
    def test_moving_pass_is_not_arrival_and_hold_resets(self):
        from fly_brain_rc.imitation.arrival import GoalStop
        tracker = GoalStop()
        self.assertFalse(tracker.update(.49, .1, .5))
        self.assertFalse(tracker.update(.49, 0, .25))
        self.assertFalse(tracker.update(.51, 0, .25))
        self.assertFalse(tracker.update(.49, 0, .25))
        self.assertTrue(tracker.update(.49, -.02, .25))

    def test_invalid_scoring_input_cannot_succeed(self):
        from fly_brain_rc.imitation.arrival import GoalStop
        for distance, speed, dt in ((float('nan'),0,.02),(-1,0,.02),(.1,0,0)):
            with self.assertRaises(ValueError):
                GoalStop().update(distance,speed,dt)


class ObservationTests(unittest.TestCase):
    def test_ranges_are_retained_without_obstacle_detection_and_goal_size_is_visible(self):
        from fly_brain_rc.imitation.observations import encode_observation
        from fly_brain_rc.contracts import SensorFrame, ObjectObservation
        sensor = SensorFrame(np.zeros((4,4,3),np.uint8),np.zeros((4,4),np.uint8),0,0,
                             .01,False,0,range_left=.02,range_right=.03,
                             range_side_left=.04,range_side_right=.05,range_rear=.06)
        target = ObjectObservation('red_ball',.8,(.4,.4,.6,.6),.1,.02,0)
        x = encode_observation(sensor,target,None)
        y = encode_observation(sensor,replace(target,apparent_area=.2),None)
        np.testing.assert_allclose(x[6:12],[.01/1.2,.02/1.2,.03/1.2,.04/1.2,.05/1.2,.06/1.2])
        self.assertFalse(np.array_equal(x,y))
        self.assertEqual(x.shape,(19,))


class TeacherTests(unittest.TestCase):
    def test_teacher_can_escape_when_only_tracking_margin_touches_cylinder(self):
        from fly_brain_rc.imitation.teacher import BicycleTeacher
        from fly_brain_rc.config import DRIVE,SIM
        from fly_brain_rc.sim.env import FlyCarEnv
        from fly_brain_rc.sim.scenarios import Scenario,BLUE
        env = FlyCarEnv(SIM,DRIVE,scenario=Scenario('tight',(((0,.49),BLUE),),(2.35,0)),render=False)
        self.assertFalse(env._pose_collides(env.state))
        teacher = BicycleTeacher(env)
        for _ in range(100):
            env.step(*teacher.action(),observe=False)
        self.assertGreater(env.car_pose()[0],.25)
        self.assertEqual(env.collisions,0)

    def test_camera_near_plane_does_not_cut_through_contact_surface(self):
        from fly_brain_rc.config import DRIVE,SIM
        from fly_brain_rc.sim.env import FlyCarEnv
        from fly_brain_rc.sim.scenarios import Scenario,BLUE
        env = FlyCarEnv(SIM,DRIVE,scenario=Scenario('occlusion',(((1,0),BLUE),),(2,0)),render=False)
        # Front footprint touches x=.72; eye at x=.626 leaves .094m.
        self.assertLess(env.m.stat.extent*env.m.vis.map.znear,.094)

    def test_dagger_teacher_replans_after_displacement_without_contact(self):
        from fly_brain_rc.imitation.teacher import BicycleTeacher
        from fly_brain_rc.config import DRIVE,SIM
        from fly_brain_rc.sim.env import FlyCarEnv
        from fly_brain_rc.sim.scenarios import Scenario
        from fly_brain_rc.sim.kinematics import BicycleState
        env = FlyCarEnv(SIM,DRIVE,scenario=Scenario('displaced',(),(2.35,0)),render=False)
        teacher = BicycleTeacher(env)
        env.state = BicycleState(0,-.8,0)
        env._sync_kinematic()
        teacher.action()
        self.assertLess(math.hypot(teacher.path[0][0],teacher.path[0][1]+.8),.04)
        teacher.cursor = len(teacher.path)-1
        env.state = BicycleState(2.65,.5,0)
        env._sync_kinematic()
        teacher.action()
        self.assertLess(math.hypot(teacher.path[0][0]-2.65,teacher.path[0][1]-.5),.04)

    def test_teacher_stops_without_contact_on_straight_and_cylinder(self):
        from fly_brain_rc.imitation.teacher import BicycleTeacher
        from fly_brain_rc.imitation.arrival import GoalStop, body_goal_distance
        from fly_brain_rc.config import DRIVE,SIM
        from fly_brain_rc.sim.env import FlyCarEnv
        from fly_brain_rc.sim.scenarios import Scenario,BLUE
        for obstacles in ((),(((1.15,.18),BLUE),)):
            env = FlyCarEnv(SIM,DRIVE,scenario=Scenario('teacher',obstacles,(2.35,0)),render=False)
            try:
                teacher, stop = BicycleTeacher(env), GoalStop()
                success = False
                for i in range(1400):
                    action = teacher.action()
                    env.step(*action,observe=False)
                    if stop.update(body_goal_distance(env),env.speed,DRIVE.timestep):
                        success = True
                        break
                self.assertTrue(success,(obstacles,env.car_pose()))
                self.assertEqual(env.collisions,0)
            finally:
                if env.renderer is not None:
                    env.renderer.close()


class NetworkTests(unittest.TestCase):
    def graph(self):
        from fly_brain_rc.imitation.connectome import extract_graph
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'fixture.csv'
            path.write_text('pre_root_id,post_root_id,syn_count\n720575940625363947,2,5\n2,3,6\n3,720575940625363947,7\n2,3,5\n99,100,9\n')
            return extract_graph(path,limit=3,fixture=True)

    def test_subgraph_preserves_exact_ids_and_anatomical_edges(self):
        graph = self.graph()
        self.assertEqual(set(graph['ids']),{'720575940625363947','2','3'})
        self.assertEqual(len(graph['edges']),3)
        self.assertTrue(graph['manifest']['fixture'])
        self.assertEqual(sorted(e[2] for e in graph['edges']),[5,7,11])


class TrainingTests(unittest.TestCase):
    def test_teacher_help_recurs_after_introductory_stage(self):
        from fly_brain_rc.imitation.training import teacher_probability
        cases = ((0,1),(5,1),(6,.5),(11,.5),(12,.2),(23,.2),
                 (24,1),(25,1),(26,.5),(27,.5),(28,0),(29,0),
                 (42,1),(43,1),(44,.5),(45,.5),(46,0),(47,0),(48,1))
        for iteration,expected in cases:
            with self.subTest(iteration=iteration):
                self.assertEqual(teacher_probability(iteration),expected)

    def test_resumed_training_uses_global_cycle_and_preserves_replay(self):
        import torch
        from unittest.mock import patch
        from fly_brain_rc.imitation.runner import run_experiment
        from fly_brain_rc.imitation.model import RecurrentPolicy,save_checkpoint,load_checkpoint
        from fly_brain_rc.contracts import SensorFrame
        from fly_brain_rc.sim.env import FlyCarEnv
        graph = NetworkTests().graph()
        def env_factory(*args,**kwargs):
            kwargs['render'] = False
            env = FlyCarEnv(*args,**kwargs)
            env.sensor_frame = lambda: SensorFrame(np.zeros((4,4,3),np.uint8),np.zeros((4,4),np.uint8),0,0,1.2,False,0,speed=env.speed)
            return env
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'network.pt'
            model = RecurrentPolicy(graph)
            model.training_metadata = {'seeds':[7],'cases':['previous']}
            old_data = [(torch.ones(2,19),torch.zeros(2,2))]
            save_checkpoint(path,model,torch.optim.Adam(model.parameters()),old_data,43)
            with patch('fly_brain_rc.imitation.runner.prepare_graph',return_value=graph),patch('fly_brain_rc.imitation.runner.FlyCarEnv',side_effect=env_factory):
                result = run_experiment(output=path,episodes=6,steps=1,updates=0,seed=7)
            self.assertEqual([r['teacher_beta'] for r in result['reports']],[1,.5,.5,0,0,1])
            restored,_,data,iteration = load_checkpoint(path,graph)
            self.assertEqual(iteration,49)
            self.assertEqual(len(data),7)
            self.assertTrue(torch.equal(data[0][0],old_data[0][0]))
            self.assertEqual(restored.training_metadata['seeds'],[7])
            self.assertEqual(restored.training_metadata['cases'][0],'previous')

    def test_partial_episode_does_not_reduce_all_training_context_to_one_step(self):
        import torch
        from fly_brain_rc.imitation.model import RecurrentPolicy
        from fly_brain_rc.imitation.training import fit_demonstrations
        model = RecurrentPolicy(NetworkTests().graph())
        optimizer = torch.optim.Adam(model.parameters())
        lengths = []
        handle = model.register_forward_pre_hook(lambda m,args:lengths.append(args[0].shape[1]) if torch.is_grad_enabled() else None)
        data = [(torch.ones(1,19),torch.zeros(1,2)),(torch.ones(16,19),torch.ones(16,2)*.2)]
        torch.manual_seed(1)
        try:
            fit_demonstrations(model,optimizer,data,updates=3,batch_size=8,sequence_length=16)
        finally:
            handle.remove()
        self.assertIn(16,lengths)

    def test_paired_training_uses_identical_replay_without_mutating_source(self):
        import torch
        from fly_brain_rc.imitation.runner import fit_paired_replay
        graph = NetworkTests().graph()
        data = [(torch.ones(16,19),torch.tensor([[.6,-.3]]*16))]
        before = data[0][0].clone()
        models,losses = fit_paired_replay(graph,data,updates=3,seed=11)
        self.assertEqual(set(models),{'connectome','rnn'})
        self.assertFalse(models['connectome'].baseline)
        self.assertTrue(models['rnn'].baseline)
        self.assertTrue(torch.equal(before,data[0][0]))
        self.assertTrue(all(math.isfinite(v) for v in losses.values()))

    def test_test_seed_namespace_cannot_replay_training_seed_plus_million(self):
        from fly_brain_rc.imitation.runner import experiment_case
        train = experiment_case(1_000_000,0)
        heldout = experiment_case(0,0,held_out=True)
        self.assertNotEqual(train.goal,heldout.goal)

    def test_held_out_maps_do_not_change_with_training_age_and_checkpoint_is_read_only(self):
        import torch
        from unittest.mock import patch
        from fly_brain_rc.imitation.runner import run_experiment
        from fly_brain_rc.imitation.model import RecurrentPolicy,save_checkpoint
        from fly_brain_rc.imitation.connectome import checksum
        from fly_brain_rc.contracts import SensorFrame
        from fly_brain_rc.sim.env import FlyCarEnv
        graph = NetworkTests().graph()
        captured = []
        def env_factory(*args,**kwargs):
            captured.append(kwargs['scenario'])
            kwargs['render'] = False
            env = FlyCarEnv(*args,**kwargs)
            env.sensor_frame = lambda: SensorFrame(np.zeros((4,4,3),np.uint8),np.zeros((4,4),np.uint8),0,0,1.2,False,0,speed=env.speed)
            return env
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'network.pt'
            model = RecurrentPolicy(graph)
            opt = torch.optim.Adam(model.parameters())
            with patch('fly_brain_rc.imitation.runner.prepare_graph',return_value=graph),patch('fly_brain_rc.imitation.runner.FlyCarEnv',side_effect=env_factory):
                for age in (3,9):
                    save_checkpoint(path,model,opt,[],age)
                    before = checksum(path)
                    run_experiment(output=path,evaluate=True,episodes=2,steps=1,seed=7)
                    self.assertEqual(checksum(path),before)
        self.assertEqual(captured[:2],captured[2:])

    def test_evaluation_never_instantiates_teacher_and_keeps_weights(self):
        import torch
        from unittest.mock import patch
        from fly_brain_rc.imitation.runner import collect_episode
        from fly_brain_rc.imitation.model import RecurrentPolicy
        from fly_brain_rc.contracts import SensorFrame
        from fly_brain_rc.config import SIM,DRIVE
        from fly_brain_rc.sim.env import FlyCarEnv
        from fly_brain_rc.sim.scenarios import Scenario
        env = FlyCarEnv(SIM,DRIVE,scenario=Scenario('test',(),(2,0)),render=False)
        env.sensor_frame = lambda: SensorFrame(np.zeros((4,4,3),np.uint8),np.zeros((4,4),np.uint8),0,0,1.2,False,env.sim_steps*.02,speed=env.speed)
        model = RecurrentPolicy(NetworkTests().graph())
        before = {k:v.clone() for k,v in model.state_dict().items()}
        with patch('fly_brain_rc.imitation.runner.BicycleTeacher',side_effect=AssertionError('teacher leaked')):
            trajectory,report = collect_episode(env,model,steps=10,beta=None)
        self.assertIsNone(trajectory)
        self.assertEqual(report['steps'],10)
        for k,v in before.items():
            self.assertTrue(torch.equal(v,model.state_dict()[k]))

    def test_recurrent_fit_reduces_demonstration_error(self):
        import torch
        from fly_brain_rc.imitation.model import RecurrentPolicy
        from fly_brain_rc.imitation.training import fit_demonstrations
        torch.manual_seed(5)
        model = RecurrentPolicy(NetworkTests().graph())
        opt = torch.optim.Adam(model.parameters(),lr=.02)
        data = [(torch.ones(16,19),torch.tensor([[.6,-.3]]*16))]
        before = (model(data[0][0][None])[0][0]-data[0][1]).square().mean().item()
        fit_demonstrations(model,opt,data,updates=30,batch_size=2,sequence_length=16)
        after = (model(data[0][0][None])[0][0]-data[0][1]).square().mean().item()
        self.assertLess(after,before*.3)

    def test_mixed_rollout_keeps_teacher_labels_not_student_actions(self):
        from fly_brain_rc.imitation.training import mix_action
        import torch
        torch.manual_seed(7)
        action,label = mix_action((.6,.2),(-.1,-.8),beta=0)
        self.assertEqual(tuple(action),(-.1,-.8))
        self.assertEqual(tuple(label),(.6,.2))
        action,label = mix_action((.6,.2),(-.1,-.8),beta=1)
        self.assertEqual(tuple(action),(.6,.2))

    def test_training_changes_allowed_edges_without_inventing_connections(self):
        import torch
        from fly_brain_rc.imitation.model import RecurrentPolicy
        torch.manual_seed(5)
        model = RecurrentPolicy(NetworkTests().graph())
        original = model.recurrent.detach().clone()
        optimizer = torch.optim.Adam(model.parameters(),lr=.01)
        x = torch.ones((1,6,19))
        for _ in range(4):
            optimizer.zero_grad()
            loss = (model(x)[0]-.7).square().mean()
            loss.backward()
            optimizer.step()
        weights = model.effective_weights().detach()
        self.assertTrue(torch.all(weights[model.mask==0]==0))
        self.assertFalse(torch.equal(weights[model.mask!=0],original[model.mask!=0]))

    def test_checkpoint_restores_optimizer_rng_and_demonstrations(self):
        import torch
        from fly_brain_rc.imitation.model import RecurrentPolicy,save_checkpoint,load_checkpoint
        graph = NetworkTests().graph()
        model = RecurrentPolicy(graph)
        model.training_metadata = {'seeds':[7],'cases':['imitation-train-7-0']}
        optimizer = torch.optim.Adam(model.parameters(),lr=.01)
        loss = model(torch.ones(1,5,19))[0].sum()
        loss.backward()
        optimizer.step()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'student.pt'
            data = [(torch.ones(5,19),torch.zeros(5,2))]
            save_checkpoint(path,model,optimizer,data,3)
            expected_rng = torch.rand(3)
            restored,opt,trajectories,iteration = load_checkpoint(path,graph)
            self.assertTrue(torch.equal(torch.rand(3),expected_rng))
            self.assertEqual(iteration,3)
            self.assertEqual(restored.training_metadata,{'seeds':[7],'cases':['imitation-train-7-0']})
            self.assertEqual(len(opt.state),len(optimizer.state))
            self.assertTrue(torch.equal(trajectories[0][0],data[0][0]))
            self.assertTrue(torch.equal(restored(torch.ones(1,5,19))[0],model(torch.ones(1,5,19))[0]))
            wrong = dict(graph,ids=['a','b','c'])
            with self.assertRaises(ValueError):
                load_checkpoint(path,wrong)
