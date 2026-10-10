import unittest


class PairedNavigationTests(unittest.TestCase):
    def test_pairing_distinguishes_clean_success_and_rejects_identity_mismatch(self):
        from scripts.evaluate_paired_navigation import paired_summary
        def row(name,success,contacts):return dict(scenario=name,family='complex',success=success,contacts=contacts,stalled_seconds=1.)
        a=[row('a',True,0),row('b',True,1),row('c',False,0)]
        b=[row('a',True,1),row('b',True,0),row('c',True,0)]
        r=paired_summary(a,b)['complex']
        self.assertEqual(r['gained'],2);self.assertEqual(r['lost'],1)
        self.assertEqual(r['source']['clean'],1);self.assertEqual(r['candidate']['clean'],2)
        with self.assertRaises(ValueError):paired_summary(a,list(reversed(b)))
        with self.assertRaises(ValueError):paired_summary(a,b[:2])

    def test_practical_gate_rejects_simple_regression_and_legacy_failure(self):
        from scripts.evaluate_paired_navigation import practical_gate
        old=dict(reports=[dict(success=True,contacts=0)]*7)
        previous=dict(reports=[dict(family='complex',success=True,contacts=0)]*6)
        summary=dict(simple=dict(source=dict(clean=3,contacts=2),candidate=dict(clean=3,contacts=1)),complex=dict(source=dict(clean=4,contacts=8),candidate=dict(clean=5,contacts=2)))
        self.assertTrue(practical_gate(old,previous,summary))
        summary['simple']['candidate']['clean']=2;self.assertFalse(practical_gate(old,previous,summary))
        summary['simple']['candidate']['clean']=3;old['reports'][0]=dict(success=False,contacts=0)
        self.assertFalse(practical_gate(old,previous,summary))
