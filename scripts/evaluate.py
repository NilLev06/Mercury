"""Repeatable headless evaluation for visual goal-seeking behavior."""

import argparse

from fly_brain_rc.config import DRIVE, SIM
from fly_brain_rc.sim.env import FlyCarEnv
from fly_brain_rc.sim.scenarios import get_scenario, scenario_names
from scripts.run_sim import build_brain, run_episode


def episode_seeds(count: int, start: int = 0) -> list[int]:
    return list(range(start, start + count))


def acceptance_passes(
    successes: int, episodes: int, penetrations: int, required_rate: float
) -> bool:
    return episodes > 0 and successes / episodes >= required_rate and penetrations == 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=SIM.episode_steps)
    parser.add_argument("--required-rate", type=float, default=None)
    parser.add_argument("--suite", choices=("random", "scenarios"), default="random")
    parser.add_argument("--scenario", choices=scenario_names())
    args = parser.parse_args()
    required_rate = args.required_rate
    if required_rate is None:
        required_rate = 0.75

    successes = 0
    contacts = 0
    penetrations = 0
    if args.scenario:
        cases = [(args.scenario, 0)]
    elif args.suite == "scenarios":
        cases = [(name, 0) for name in scenario_names()]
    else:
        cases = [(None, seed) for seed in episode_seeds(args.episodes, args.start_seed)]

    for scenario_name, seed in cases:
        scenario = get_scenario(scenario_name) if scenario_name else None
        env = FlyCarEnv(SIM, DRIVE, seed=seed, scenario=scenario)
        result = run_episode(build_brain(), env, args.steps)
        successes += int(result["success"])
        contacts += result["collisions"]
        penetrations += result["penetrations"]
        print(
            f"case={scenario_name or f'seed-{seed:02d}'} success={result['success']} steps={result['steps']} "
            f"contacts={result['collisions']} visible={result['visibility_ratio']:.2f} "
            f"min_goal={result['min_goal_distance']:.3f} penetrations={result['penetrations']} "
            f"reverse={result['reverse_steps']} near_goal_reverse={result['near_goal_reverse_steps']}"
        )

    total = len(cases)
    rate = successes / max(total, 1)
    print(
        f"summary: successes={successes}/{total} rate={rate:.1%} "
        f"contacts={contacts} penetrations={penetrations}"
    )
    return 0 if acceptance_passes(successes, total, penetrations, required_rate) else 1


if __name__ == "__main__":
    raise SystemExit(main())
