"""Structured neural evaluation."""
import argparse
from dataclasses import asdict
import json
from fly_brain_rc.config import DRIVE, SIM
from fly_brain_rc.sim.env import FlyCarEnv
from fly_brain_rc.sim.scenarios import get_scenario, scenario_names
from fly_brain_rc.sim.suites import training_cases, type_a_cases, type_b_cases
from fly_brain_rc.neural.brain import NeuralFlyBrain
from fly_brain_rc.runtime import FlyRuntime, load_genome, run_episode


def training_evaluator(steps):
    """Build an evaluator that can only resolve generated training cases."""
    from fly_brain_rc.evolution import GenomeScore
    def evaluate(genome, identifiers):
        successes = collisions = acquired = 0
        completion = energy = reverse = variation = 0.0
        for identifier in identifiers:
            _, seed, index = identifier.split('-')
            case = training_cases(int(seed), int(index) + 1)[int(index)]
            env = FlyCarEnv(SIM, DRIVE, seed=case.seed, scenario=case.scenario)
            try:
                def consume(record):
                    nonlocal energy
                    energy += (record.command.traction ** 2 + record.command.steering ** 2) * DRIVE.timestep
                result = run_episode(FlyRuntime(env, NeuralFlyBrain(genome)), steps, on_tick=consume)
            finally:
                env.renderer.close()
            successes += int(result.success)
            collisions += result.collisions + result.penetrations
            acquired += int(result.acquisition_time is not None)
            completion += result.stable_stop_time if result.success else steps * DRIVE.timestep
            reverse += result.reverse_distance
            variation += result.steering_variation
        return GenomeScore(successes, collisions, acquired / len(identifiers), completion,
                           energy, reverse, variation)
    return evaluate


def episode_seeds(count, start=0):
    return list(range(start, start + count))


def acceptance_passes(successes, episodes, penetrations, required_rate, collisions=0):
    return episodes > 0 and successes / episodes >= required_rate and penetrations == 0 and collisions == 0


def suite_required_rate(suite, override=None):
    return override if override is not None else {'type-a': .90, 'type-b': .70}.get(suite, .75)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--start-seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=SIM.episode_steps)
    parser.add_argument('--required-rate', type=float)
    parser.add_argument('--suite', choices=('train', 'type-a', 'type-b', 'random', 'scenarios'), default='type-a')
    parser.add_argument('--scenario', choices=scenario_names())
    parser.add_argument('--genome')
    parser.add_argument('--target', choices=('red_ball',), default='red_ball')
    args = parser.parse_args()
    if args.episodes <= 0 or args.steps <= 0:
        parser.error('episodes and steps must be positive')
    required_rate = suite_required_rate(args.suite, args.required_rate)
    if not 0 <= required_rate <= 1:
        parser.error('required-rate must be in [0, 1]')
    genome = load_genome(args.genome)
    if args.scenario:
        cases = [(args.scenario, 0, get_scenario(args.scenario))]
    elif args.suite == 'scenarios':
        cases = [(name, 0, get_scenario(name)) for name in scenario_names()]
    else:
        suite = (training_cases(args.start_seed, args.episodes) if args.suite in ('train', 'random')
                 else type_a_cases() if args.suite == 'type-a' else type_b_cases())
        if args.episodes > len(suite):
            parser.error('held-out suites contain 10 unique cases; request at most 10 episodes')
        cases = [(c.identifier, c.seed, c.scenario) for c in suite[:args.episodes]]
    successes = contacts = penetrations = 0
    for identifier, seed, scenario in cases:
        env = FlyCarEnv(SIM, DRIVE, seed=seed, scenario=scenario)
        try:
            result = run_episode(FlyRuntime(env, NeuralFlyBrain(genome), target=args.target), args.steps)
        finally:
            env.renderer.close()
        successes += int(result.success)
        contacts += result.collisions
        penetrations += result.penetrations
        print(json.dumps({'case': identifier, **asdict(result)}, allow_nan=False), flush=True)
    print(json.dumps({'suite': args.suite, 'successes': successes, 'episodes': len(cases),
                      'rate': successes / len(cases), 'required_rate': required_rate,
                      'collisions': contacts, 'penetrations': penetrations}), flush=True)
    return 0 if acceptance_passes(successes, len(cases), penetrations, required_rate, contacts) else 1


if __name__ == '__main__':
    raise SystemExit(main())
