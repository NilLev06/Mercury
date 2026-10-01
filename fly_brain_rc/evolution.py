"""Seeded bounded evolution of magnitudes in the fixed neural topology."""
from dataclasses import dataclass, asdict
import json
import math
import os
from pathlib import Path
import tempfile
import numpy as np
from .neural.genome import FlyGenome

TOPOLOGY_VERSION = 1


@dataclass(frozen=True)
class EvolutionConfig:
    seed: int = 0
    generations: int = 2
    population_size: int = 4
    elites: int = 1
    mutation_scale: float = .05
    steps: int = 1000
    scenario_ids: tuple = ('train-0-000', 'train-0-001')
    checkpoint: str | None = None
    resume: str | None = None

    def __post_init__(self):
        for name in ('seed', 'generations', 'population_size', 'elites', 'steps'):
            if type(getattr(self, name)) is not int or getattr(self, name) < (0 if name == 'seed' else 1):
                raise ValueError(f'invalid {name}')
        if self.elites > self.population_size:
            raise ValueError('elites exceed population')
        if not math.isfinite(self.mutation_scale) or self.mutation_scale < 0:
            raise ValueError('invalid mutation scale')
        if not self.scenario_ids or len(set(self.scenario_ids)) != len(self.scenario_ids):
            raise ValueError('scenarios must be unique and nonempty')
        for identifier in self.scenario_ids:
            parts = identifier.split('-') if isinstance(identifier, str) else []
            if len(parts) != 3 or parts[0] != 'train' or not all(p.isdigit() for p in parts[1:]):
                raise ValueError('evolution accepts training cases only')
        object.__setattr__(self, 'scenario_ids', tuple(self.scenario_ids))


@dataclass(frozen=True)
class GenomeScore:
    successes: int = 0
    collisions: int = 0
    acquisition: float = 0.0
    completion_time: float = 0.0
    energy: float = 0.0
    reverse_distance: float = 0.0
    steering_variation: float = 0.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f'invalid score {name}')
        if type(self.successes) is not int or type(self.collisions) is not int:
            raise ValueError('counts must be integers')

    def rank_key(self):
        return (self.collisions == 0, self.successes, self.acquisition,
                -self.completion_time, -self.energy, -self.reverse_distance, -self.steering_variation)


@dataclass
class EvolutionState:
    seed: int
    generation: int
    population: list
    scores: list
    scenario_ids: tuple
    rng_state: dict
    settings: dict

    def to_dict(self):
        return {'version': 1, 'topology_version': TOPOLOGY_VERSION,
                'parameter_version': FlyGenome.VERSION,
                'parameter_schema': {k: list(v) for k, v in FlyGenome.BOUNDS.items()},
                'seed': self.seed, 'generation': self.generation,
                'population': [g.vector().tolist() for g in self.population],
                'scores': [asdict(s) for s in self.scores],
                'scenario_ids': list(self.scenario_ids), 'rng_state': self.rng_state,
                'settings': self.settings}

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
                temporary = stream.name
                json.dump(self.to_dict(), stream, allow_nan=False, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def load(cls, path, config):
        try:
            data = json.loads(Path(path).read_text())
            if (data['version'] != 1 or data['topology_version'] != TOPOLOGY_VERSION
                    or data['parameter_version'] != FlyGenome.VERSION
                    or data['parameter_schema'] != {k: list(v) for k, v in FlyGenome.BOUNDS.items()}):
                raise ValueError('checkpoint topology or parameter schema mismatch')
            if (data['seed'] != config.seed or tuple(data['scenario_ids']) != config.scenario_ids
                    or data['settings'] != _settings(config)):
                raise ValueError('checkpoint configuration mismatch')
            if type(data['generation']) is not int or not 1 <= data['generation'] <= config.generations:
                raise ValueError('invalid generation')
            population = [FlyGenome.default().with_vector(v) for v in data['population']]
            scores = [GenomeScore(**s) for s in data['scores']]
            if len(population) != config.population_size or len(scores) != len(population):
                raise ValueError('invalid checkpoint population')
            rng = np.random.default_rng()
            rng.bit_generator.state = data['rng_state']
            return cls(data['seed'], data['generation'], population, scores,
                       tuple(data['scenario_ids']), data['rng_state'], data['settings'])
        except (KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
            raise ValueError('invalid checkpoint') from exc


def _settings(config):
    return {k: getattr(config, k) for k in ('population_size', 'elites', 'mutation_scale', 'steps')}


def evolve(config, evaluator):
    """Evaluator receives (genome, training identifiers) and returns GenomeScore.

    Checkpoints contain an evaluated generation. Continuation breeds from that
    generation with its saved RNG state, exactly as an uninterrupted run does.
    """
    rng = np.random.default_rng(config.seed)
    bounds = np.array(list(FlyGenome.BOUNDS.values()))
    def mutate(parent):
        values = parent.vector() + rng.normal(size=len(bounds)) * (bounds[:, 1] - bounds[:, 0]) * config.mutation_scale
        return parent.with_vector(np.clip(values, bounds[:, 0], bounds[:, 1]))
    if config.resume:
        state = EvolutionState.load(config.resume, config)
        rng.bit_generator.state = state.rng_state
    else:
        default = FlyGenome.default()
        state = EvolutionState(config.seed, 0, [default] + [mutate(default) for _ in range(config.population_size - 1)],
                               [], config.scenario_ids, rng.bit_generator.state, _settings(config))
    while state.generation < config.generations:
        if state.scores:
            ranked = sorted(zip(state.population, state.scores), key=lambda pair: pair[1].rank_key(), reverse=True)
            elite = [g for g, _ in ranked[:config.elites]]
            state.population = elite + [mutate(elite[int(rng.integers(len(elite)))])
                                         for _ in range(config.population_size - len(elite))]
        state.scores = [evaluator(g, config.scenario_ids) for g in state.population]
        if any(not isinstance(score, GenomeScore) for score in state.scores):
            raise ValueError('evaluator must return GenomeScore')
        state.generation += 1
        state.rng_state = rng.bit_generator.state
        if config.checkpoint:
            state.save(config.checkpoint)
    return state
