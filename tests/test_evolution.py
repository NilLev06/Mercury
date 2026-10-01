import json
from pathlib import Path
import tempfile
import unittest
from dataclasses import replace
from fly_brain_rc.evolution import EvolutionConfig, GenomeScore, EvolutionState, evolve
from fly_brain_rc.neural.genome import FlyGenome


def evaluator(genome, scenarios):
    return GenomeScore(successes=sum(int(genome.target_gain > 5) for _ in scenarios), collisions=0,
                       acquisition=genome.target_gain / 12, energy=genome.tau)


class EvolutionTests(unittest.TestCase):
    def test_same_seed_and_bounded_population(self):
        config = EvolutionConfig(seed=7, generations=2, population_size=4, mutation_scale=10)
        first = evolve(config, evaluator)
        self.assertEqual(first.to_dict(), evolve(config, evaluator).to_dict())
        self.assertEqual(first.generation, 2)
        self.assertEqual(len(first.population), 4)
        for genome in first.population:
            FlyGenome.from_json(genome.to_json())

    def test_collision_priority_and_tiebreakers(self):
        safe = GenomeScore(successes=0, collisions=0, acquisition=999, energy=999)
        crash = GenomeScore(successes=10, collisions=1, acquisition=1, energy=1)
        self.assertGreater(safe.rank_key(), crash.rank_key())
        self.assertGreater(GenomeScore(acquisition=.9).rank_key(), GenomeScore(acquisition=.1).rank_key())

    def test_checkpoint_continuation_matches_uninterrupted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'state.json')
            config = EvolutionConfig(seed=7, generations=1, population_size=4, checkpoint=path)
            evolve(config, evaluator)
            resumed = evolve(replace(config, generations=3, resume=path), evaluator)
            whole = evolve(replace(config, generations=3, checkpoint=None), evaluator)
            self.assertEqual(resumed.to_dict(), whole.to_dict())
            payload = json.loads(Path(path).read_text())
            payload['parameter_version'] = 999
            Path(path).write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                EvolutionState.load(path, config)

    def test_elite_survives_large_mutation(self):
        first = evolve(EvolutionConfig(seed=7, generations=1, mutation_scale=10), evaluator)
        second = evolve(EvolutionConfig(seed=7, generations=2, mutation_scale=10), evaluator)
        best = max(first.scores, key=lambda score: score.rank_key())
        self.assertGreaterEqual(max(s.rank_key() for s in second.scores), best.rank_key())

    def test_reject_invalid_config_and_checkpoint(self):
        for kwargs in ({'population_size': 0}, {'elites': 5}, {'mutation_scale': float('nan')},
                       {'scenario_ids': ('type-a-00',)}):
            with self.assertRaises(ValueError):
                EvolutionConfig(**kwargs)

    def test_resume_rejects_changed_episode_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'state.json')
            config = EvolutionConfig(checkpoint=path, steps=2)
            evolve(config, evaluator)
            with self.assertRaises(ValueError):
                evolve(replace(config, resume=path, steps=3), evaluator)
