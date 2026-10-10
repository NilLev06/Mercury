import unittest
class MasteryCasesTests(unittest.TestCase):
    def test_all_complex_mirrors_resolve_to_reflected_geometry_and_start(self):
        from scripts.train_mastery_curriculum import mastery_cases
        from scripts.collect_mirror_replay import mirror_case,mirror_pose
        cases=mastery_cases()
        original=[v for n,v in cases.items() if '-complex-' in n and not n.endswith('-physical-mirror')]
        self.assertEqual(len(original),24)
        for case,pose,tier in original:
            reflected,reflected_pose,reflected_tier=cases[case.name+'-physical-mirror']
            self.assertEqual(reflected,mirror_case(case));self.assertEqual(reflected_pose,mirror_pose(pose));self.assertEqual(reflected_tier,tier)
