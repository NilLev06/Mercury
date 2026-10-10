import unittest

class FrozenGateTests(unittest.TestCase):
    def test_clean_simple_protocol_does_not_count_contact_arrivals(self):
        from scripts.evaluate_frozen_navigation import frozen_gate
        old=dict(successes=7,contacts=0)
        c=dict(complex=dict(successes=29,clean_successes=29,contacts=0),simple=dict(successes=15,clean_successes=14,contacts=1))
        b=dict(complex=dict(successes=0,clean_successes=0,contacts=2000),simple=dict(successes=15,clean_successes=15,contacts=100))
        self.assertTrue(frozen_gate(old,c,b,dict(minimum_complex_successes=29)))
        self.assertFalse(frozen_gate(old,c,b,dict(minimum_complex_successes=29,simple_metric='clean_successes')))
        c['simple']['clean_successes']=15
        self.assertTrue(frozen_gate(old,c,b,dict(minimum_complex_successes=29,simple_metric='clean_successes')))
