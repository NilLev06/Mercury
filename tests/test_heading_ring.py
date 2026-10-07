import math
import unittest
import numpy as np


def error(a,b):
    return abs(math.atan2(math.sin(a-b),math.cos(a-b)))


class HeadingRingTests(unittest.TestCase):
    def test_long_turn_does_not_accumulate_large_lag_error(self):
        for velocity in (-2.,2.):
            ring=self.ring();initial=ring.heading
            for _ in range(6000):ring.step(velocity,.01)
            self.assertLess(error(ring.heading,initial+60*velocity),math.radians(5))

    def test_small_feedback_gain_errors_do_not_destroy_dark_memory(self):
        for gain in (.95,.99,1.01,1.05):
            ring=self.ring();initial=ring.heading
            ring._positive*=gain;ring._negative*=gain
            for _ in range(6000):ring.step(0.,.01)
            self.assertIsNotNone(ring.heading)
            self.assertLess(error(ring.heading,initial),math.radians(2))
            self.assertGreater(ring.contrast,.1)

    def test_radial_disturbance_recovers_without_changing_phase(self):
        ring=self.ring();initial=ring.heading
        ring._compass=1+.25*(ring.activity-1)
        ring._left=1+.25*(ring._left-1)
        ring._right=1+.25*(ring._right-1)
        for _ in range(1000):ring.step(0.,.01)
        self.assertGreater(ring.contrast,.35)
        self.assertLess(error(ring.heading,initial),.02)

    def ring(self,**kwargs):
        from fly_brain_rc.neural.heading_ring import HeadingRing
        ring=HeadingRing(**kwargs)
        for _ in range(100):
            ring.step(0.,.01,cue_phase=.37)
        return ring

    def test_dark_retention_requires_recurrent_feedback(self):
        healthy=self.ring();lesion=self.ring(recurrence=False)
        initial=healthy.heading
        for _ in range(1000):
            healthy.step(0.,.01);lesion.step(0.,.01)
        self.assertLess(error(healthy.heading,initial),.02)
        self.assertGreater(healthy.contrast,.1)
        self.assertLess(lesion.contrast,.001)
        self.assertIsNone(lesion.heading)

    def test_turns_shift_population_in_both_directions(self):
        for velocity in (-.5,.5):
            with self.subTest(velocity=velocity):
                ring=self.ring();initial=ring.heading
                for _ in range(200):ring.step(velocity,.01)
                self.assertLess(error(ring.heading,initial+2*velocity),.08)

    def test_turn_input_ablation_prevents_integration(self):
        ring=self.ring(turn_input=False);initial=ring.heading
        for _ in range(200):ring.step(.5,.01)
        self.assertLess(error(ring.heading,initial),.02)
        self.assertGreater(error(ring.heading,initial+1),.9)

    def test_visual_input_reanchors_displaced_population(self):
        ring=self.ring()
        for _ in range(200):ring.step(.5,.01)
        for _ in range(200):ring.step(0.,.01,cue_phase=-.8)
        self.assertLess(error(ring.heading,-.8),.03)

    def test_invalid_step_does_not_change_activity(self):
        ring=self.ring();before=ring.activity
        for velocity,dt,cue in ((float('nan'),.01,None),(0.,0.,None),
                                (0.,.01,float('inf')),(100.,.01,None)):
            with self.assertRaises(ValueError):ring.step(velocity,dt,cue_phase=cue)
            np.testing.assert_array_equal(ring.activity,before)

    def test_reversals_across_wraparound_are_timestep_consistent(self):
        from fly_brain_rc.neural.heading_ring import HeadingRing
        results=[]
        for dt in (.01,.02):
            ring=HeadingRing()
            for _ in range(round(1/dt)):ring.step(0.,dt,cue_phase=3.1)
            for velocity in (1.,-1.):
                for _ in range(round(2/dt)):ring.step(velocity,dt)
            for _ in range(round(.5/dt)):ring.step(0.,dt)
            self.assertLess(error(ring.heading,3.1),.08)
            self.assertTrue(np.isfinite(ring.activity).all())
            results.append(ring.heading)
        self.assertLess(error(*results),.01)
