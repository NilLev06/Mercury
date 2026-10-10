import unittest
from unittest.mock import patch
from types import SimpleNamespace

class BroadNeuralLabelsTests(unittest.TestCase):
    def test_old_uses_source_and_only_failed_neural_maps_use_planner(self):
        from scripts.train_broad_neural_dagger import label_teacher
        actors={t:SimpleNamespace(eval=lambda:None) for t in ('old','simple','complex')}
        specs={'old':dict(label_teacher='planner'),'easy':dict(label_teacher='neural'),'hard':dict(label_teacher='planner')}
        a=label_teacher(None,'old','simple',specs,actors,{'old'})
        self.assertIs(a.actor,actors['old'])
        b=label_teacher(None,'easy','simple',specs,actors,{'old'});self.assertIs(b.actor,actors['simple']);self.assertIsNone(b.state)
        self.assertIsNot(a,b)
        with patch('scripts.train_broad_neural_dagger.BicycleTeacher',return_value='expert') as expert:
            self.assertEqual(label_teacher(None,'hard','complex',specs,actors,{'old'}),'expert')
            expert.assert_called_once_with(None,cruise=.45,margin=.12)
