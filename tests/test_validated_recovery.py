import unittest
import torch
from unittest.mock import patch
class ValidatedRecoveryTests(unittest.TestCase):
    def test_reference_route_retains_reverse_cusp_and_omits_stationary_duplicates(self):
        from scripts.train_validated_recovery import route_points
        poses=[(0.,0.,0.),(.01,0.,0.),(.03,0.,0.),(.02,0.,0.),(0.,0.,0.),(0.,0.,0.)]
        route=route_points(poses,torch.tensor([[.2,0.],[.2,0.],[-.2,0.],[-.2,0.],[0.,0.]]))
        self.assertEqual(route[0][3],1);self.assertTrue(any(p[3]==-1 for p in route))
        self.assertLess(len(route),len(poses));self.assertEqual(route[-1][:2],(0.,0.))

    def test_recovery_labels_only_follow_prefix_context(self):
        from scripts.train_validated_recovery import join_recovery
        prefix=dict(x=torch.ones(5,31),y=torch.ones(5,2))
        suffix=dict(x=torch.full((3,31),2.),y=torch.full((3,2),-.3))
        joined=join_recovery(prefix,suffix,2)
        self.assertEqual(joined['supervision_start'],2);self.assertEqual(joined['x'].shape,(5,31))
        torch.testing.assert_close(joined['x'][:2],prefix['x'][:2]);torch.testing.assert_close(joined['y'][2:],suffix['y'])
        self.assertEqual(prefix['x'].shape,(5,31))

    def test_recovery_accepts_only_executed_clean_suffix_and_replays_prefix(self):
        from scripts import train_validated_recovery as module
        prefix=dict(x=torch.ones(40,31),y=torch.full((40,2),.2),poses=[(0.,0.,0.)]*40)
        suffix=dict(x=torch.ones(3,31),y=torch.zeros(3,2))
        class Env:
            renderer=None;collisions=0
            def car_pose(self):return (0.,0.,0.)
        model=type('Model',(),{'observation_schema':'v2'})()
        for contacts,accepted in [(0,True),(1,False)]:
            with patch.object(module,'FlyCarEnv',return_value=Env()),patch.object(module,'place_start'),patch.object(module,'ReferenceRouteTeacher'),patch.object(module,'replay_prefix') as replay,patch.object(module,'collect_precontact',return_value=(prefix,dict(success=False,contacts=0))),patch.object(module,'teacher_suffix',return_value=(suffix,dict(success=True,contacts=contacts))) as teacher:
                data,report=module.collect_recovery(model,None,(0.,0.,0.),[])
            self.assertEqual(report['accepted'],accepted)
            self.assertEqual(replay.call_args.args[2],50)
            self.assertEqual(teacher.call_args.args[3],2350)
            if accepted:
                self.assertEqual(data['supervision_start'],10);self.assertEqual(len(data['x']),13)
            else:self.assertIsNone(data)

    def test_route_direction_uses_motion_while_reverse_command_brakes(self):
        from scripts.train_validated_recovery import route_points
        route=route_points([(0.,0.,0.),(.01,0.,0.)],torch.tensor([[-.4,0.]]))
        self.assertEqual(route[0][3],1)
