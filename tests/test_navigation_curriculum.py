import unittest
import tempfile
from pathlib import Path
import torch


class CurriculumTests(unittest.TestCase):
    def test_complexity_increases_and_families_cover_whole_assistance_cycles(self):
        from fly_brain_rc.imitation.curriculum import training_case
        expected = ((24,'straight'),(30,'cylinder'),(36,'turn'),(42,'straight'),
                    (48,'straight'),(54,'cylinder'),(60,'turn'),(66,'slalom'),
                    (72,'straight'),(78,'cylinder'),(84,'turn'),(90,'slalom'),
                    (96,'occluded'),(102,'recovery'))
        for age,family in expected:
            for offset in range(6):
                self.assertEqual(training_case(0,age+offset)[2],family)
        self.assertEqual(training_case(11,103),training_case(11,103))
        self.assertNotEqual(training_case(11,96)[0],training_case(12,96)[0])

    def test_teacher_can_solve_recovery_and_complex_training_starts(self):
        from fly_brain_rc.imitation.curriculum import training_case,place_start
        from fly_brain_rc.imitation.teacher import BicycleTeacher
        from fly_brain_rc.imitation.arrival import GoalStop,body_goal_distance
        from fly_brain_rc.sim.env import FlyCarEnv
        from fly_brain_rc.config import SIM,DRIVE
        for age in (84,90,96,102,103):
            case,start,family = training_case(0,age)
            env = FlyCarEnv(SIM,DRIVE,scenario=case,render=False)
            place_start(env,start)
            self.assertFalse(env._pose_collides(env.state),(family,start))
            teacher,stop = BicycleTeacher(env),GoalStop()
            success = False
            for _ in range(2400):
                env.step(*teacher.action(),observe=False)
                if stop.update(body_goal_distance(env),env.speed,.02):
                    success = True
                    break
            self.assertTrue(success,(family,start,env.car_pose()))
            self.assertEqual(env.collisions,0)


class SelectionTests(unittest.TestCase):
    def test_validation_runs_without_teacher_and_does_not_change_network(self):
        import numpy as np
        from unittest.mock import patch
        from tests.test_imitation import NetworkTests
        from fly_brain_rc.imitation.model import RecurrentPolicy
        from fly_brain_rc.imitation.validation import validate
        from fly_brain_rc.sim.env import FlyCarEnv
        from fly_brain_rc.contracts import SensorFrame
        model = RecurrentPolicy(NetworkTests().graph())
        before = {k:v.clone() for k,v in model.state_dict().items()}
        def env_factory(*args,**kwargs):
            kwargs['render'] = False
            env = FlyCarEnv(*args,**kwargs)
            env.sensor_frame = lambda: SensorFrame(np.zeros((4,4,3),np.uint8),np.zeros((4,4),np.uint8),0,0,1.2,False,0,speed=env.speed)
            return env
        with patch('fly_brain_rc.imitation.validation.FlyCarEnv',side_effect=env_factory),patch('fly_brain_rc.imitation.runner.BicycleTeacher',side_effect=AssertionError('teacher in validation')):
            result = validate(model,steps=1)
        self.assertEqual(result['episodes'],7)
        self.assertFalse(result['cancelled'])
        for k,v in model.state_dict().items():
            self.assertTrue(torch.equal(v,before[k]))

    def test_runner_checks_before_training_and_at_global_interval(self):
        import numpy as np
        from unittest.mock import patch
        from tests.test_imitation import NetworkTests
        from fly_brain_rc.imitation.model import RecurrentPolicy,save_checkpoint,load_checkpoint
        from fly_brain_rc.imitation.runner import run_experiment
        from fly_brain_rc.sim.env import FlyCarEnv
        from fly_brain_rc.contracts import SensorFrame
        graph = NetworkTests().graph()
        model = RecurrentPolicy(graph)
        def env_factory(*args,**kwargs):
            kwargs['render'] = False
            env = FlyCarEnv(*args,**kwargs)
            env.sensor_frame = lambda: SensorFrame(np.zeros((4,4,3),np.uint8),np.zeros((4,4),np.uint8),0,0,1.2,False,0,speed=env.speed)
            return env
        good = dict(successes=1,episodes=7,contacts=0,mean_final_distance=1.,cancelled=False,step_limit=1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.pt'
            save_checkpoint(path,model,torch.optim.Adam(model.parameters()),[],11)
            with patch('fly_brain_rc.imitation.runner.prepare_graph',return_value=graph),patch('fly_brain_rc.imitation.runner.FlyCarEnv',side_effect=env_factory),patch('fly_brain_rc.imitation.runner.validate',side_effect=[good,dict(good,successes=2)]):
                result = run_experiment(output=path,episodes=1,steps=1,updates=0)
            self.assertEqual([r['iteration'] for r in result['validations']],[11,12])
            self.assertEqual(load_checkpoint(path.with_name('model.best.pt'),graph)[3],12)
            self.assertEqual(load_checkpoint(path,graph)[3],12)

    def test_selection_does_not_reuse_another_suite_or_architecture(self):
        import json
        from tests.test_imitation import NetworkTests
        from fly_brain_rc.imitation.model import RecurrentPolicy
        from fly_brain_rc.imitation.validation import promote_best,previous_validation,best_paths
        model = RecurrentPolicy(NetworkTests().graph())
        good = dict(successes=4,episodes=7,contacts=0,mean_final_distance=.5,cancelled=False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.pt'
            promote_best(path,model,torch.optim.Adam(model.parameters()),[],3,good)
            _,metadata = best_paths(path)
            original = json.loads(metadata.read_text())
            for change in ({'suite':'different'},{'baseline':True},{'signature':'other'},{'step_limit':1}):
                metadata.write_text(json.dumps(dict(original,**change)))
                self.assertIsNone(previous_validation(path,model))

    def test_worse_validation_does_not_overwrite_best_or_latest_memory(self):
        from tests.test_imitation import NetworkTests
        from fly_brain_rc.imitation.model import RecurrentPolicy,save_checkpoint,load_checkpoint
        from fly_brain_rc.imitation.validation import promote_best
        from fly_brain_rc.imitation.connectome import checksum
        graph = NetworkTests().graph()
        model = RecurrentPolicy(graph)
        opt = torch.optim.Adam(model.parameters())
        good = dict(successes=4,episodes=7,contacts=0,mean_final_distance=.5,cancelled=False)
        bad = dict(good,successes=3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'model.pt'
            save_checkpoint(path,model,opt,[],107)
            latest_hash = checksum(path)
            self.assertTrue(promote_best(path,model,opt,[],107,good))
            best = path.with_name('model.best.pt')
            best_hash = checksum(best)
            with torch.no_grad(): model.readout.bias.add_(1)
            self.assertFalse(promote_best(path,model,opt,[],119,bad))
            self.assertEqual(checksum(best),best_hash)
            self.assertEqual(checksum(path),latest_hash)
            self.assertEqual(load_checkpoint(best,graph)[3],107)
            self.assertFalse(promote_best(path,model,opt,[],119,dict(good,cancelled=True)))

    def test_test_launcher_prefers_best_but_training_resumes_latest(self):
        from fly_brain_rc.dashboard.lab import build_job
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root/'artifacts/imitation'
            folder.mkdir(parents=True)
            (folder/'connectome.pt').touch()
            (folder/'connectome.best.pt').touch()
            self.assertIn(str(folder/'connectome.best.pt'),build_job(root,dict(mode='imitation_test')))
            self.assertIn(str(folder/'connectome.pt'),build_job(root,dict(mode='imitation_train')))
