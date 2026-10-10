import unittest
import torch
from fly_brain_rc.sim.scenarios import Scenario,BLUE
class MirrorReplayTests(unittest.TestCase):
    def test_geometry_and_commands_are_reflections_and_involutions(self):
        from scripts.collect_mirror_replay import mirror_case,mirror_pose,mirror_actions,PlaybackTeacher
        c=Scenario('route',(((1.,.3),BLUE),),(2.5,-.1));m=mirror_case(c)
        self.assertEqual(m.obstacles,(((1.,-.3),BLUE),));self.assertEqual(m.goal,(2.5,.1))
        twice=mirror_case(m);self.assertEqual(twice.obstacles,c.obstacles);self.assertEqual(twice.goal,c.goal)
        p=(.1,.2,.3);self.assertEqual(mirror_pose(mirror_pose(p)),p)
        y=torch.tensor([[.4,.7],[0.,-.3]]);v=mirror_actions(y)
        torch.testing.assert_close(v,torch.tensor([[.4,-.7],[0.,.3]]));self.assertTrue(torch.equal(mirror_actions(v),y))
        teacher=PlaybackTeacher(v);self.assertEqual(teacher.action(),tuple(v[0].tolist()));self.assertEqual(teacher.action(),tuple(v[1].tolist()));self.assertEqual(teacher.action(),tuple(v[1].tolist()))
        self.assertTrue(torch.equal(y,torch.tensor([[.4,.7],[0.,-.3]])))
