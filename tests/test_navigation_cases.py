import math
import unittest

from fly_brain_rc.config import SIM, DRIVE
from fly_brain_rc.imitation.curriculum import place_start
from fly_brain_rc.sim.env import FlyCarEnv


class NavigationCasesTests(unittest.TestCase):
    def test_repeatable_geometry_and_disjoint_splits(self):
        from fly_brain_rc.imitation.navigation_cases import navigation_cases
        train = navigation_cases(42, 'train')
        assert train == navigation_cases(42, 'train')
        selection = navigation_cases(42, 'selection')
        final = navigation_cases(42, 'final')
        geometries = lambda cases: {(case.obstacles, case.goal, pose) for case, pose, _ in cases}
        assert not geometries(train) & geometries(selection)
        assert not geometries(train) & geometries(final)
        assert len(train) == 6


    def test_maps_have_safe_starts_and_accessible_goal_clearance(self):
        from fly_brain_rc.imitation.navigation_cases import navigation_cases
        for seed in (12,42,8001):
            for case, pose, family in navigation_cases(seed, 'final'):
                env = FlyCarEnv(SIM, DRIVE, scenario=case, render=False)
                place_start(env, pose)
                assert not env._pose_collides(env.state)
                assert all(math.dist(case.goal, point) > SIM.obstacle_radius + .5
                           for point, _ in case.obstacles)
                if family == 'search':
                    bearing = math.atan2(case.goal[1]-pose[1], case.goal[0]-pose[0])
                    assert abs((bearing-pose[2]+math.pi)%(2*math.pi)-math.pi) > math.pi/2
                else:
                    assert len(case.obstacles) >= 3

    def test_sequential_curriculum_changes_relative_geometry_and_keeps_starts_safe(self):
        from fly_brain_rc.imitation.navigation_cases import sequential_cases
        cases = sequential_cases(71, 'train', count=8, obstacles=2)
        self.assertEqual(cases, sequential_cases(71, 'train', count=8, obstacles=2))
        self.assertEqual(len(cases), 8)
        separations = []
        for case, pose, family in cases:
            self.assertEqual(family, 'sequential')
            self.assertEqual(len(case.obstacles), 2)
            separations.append(math.dist(case.obstacles[0][0], case.obstacles[1][0]))
            env = FlyCarEnv(SIM, DRIVE, scenario=case, render=False)
            place_start(env, pose)
            self.assertFalse(env._pose_collides(env.state))
            self.assertTrue(all(math.dist(case.goal, p) > .78 for p, _ in case.obstacles))
        self.assertGreater(max(separations)-min(separations), .15)
        other = sequential_cases(71, 'final', count=8, obstacles=2)
        self.assertFalse({c.obstacles for c,_,_ in cases} & {c.obstacles for c,_,_ in other})

    def test_mixed_curriculum_keeps_both_tiers_and_changes_proportion(self):
        from fly_brain_rc.imitation.navigation_cases import mixed_cases
        ratios = []
        for stage in range(3):
            cases = mixed_cases(1301, 'train', stage)
            self.assertEqual(len(cases),12)
            simple = sum(tier == 'simple' for _,_,tier in cases)
            complex_count = sum(tier == 'complex' for _,_,tier in cases)
            ratios.append((simple,complex_count))
            self.assertEqual(cases,mixed_cases(1301,'train',stage))
            for case,pose,_ in cases:
                env = FlyCarEnv(SIM,DRIVE,scenario=case,render=False)
                place_start(env,pose)
                self.assertFalse(env._pose_collides(env.state))
        self.assertEqual(ratios,[(8,4),(6,6),(4,8)])
        train=mixed_cases(1301,'train',2)
        final=mixed_cases(1301,'final',2)
        self.assertFalse({(c.obstacles,c.goal,p) for c,p,_ in train} &
                         {(c.obstacles,c.goal,p) for c,p,_ in final})
