"""Repeatable headless evaluation for visual goal-seeking behavior."""

import argparse

from fly_brain_rc.config import DRIVE, SIM
from fly_brain_rc.sim.env import FlyCarEnv
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
    parser.add_argument("--required-rate", type=float, default=0.75)
    args = parser.parse_args()

    successes = 0
    contacts = 0
    penetrations = 0
    for seed in episode_seeds(args.episodes, args.start_seed):
        env = FlyCarEnv(SIM, DRIVE, seed=seed)
        result = run_episode(build_brain(), env, args.steps)
        successes += int(result["success"])
        contacts += result["collisions"]
        penetrations += result["penetrations"]
        print(
            f"seed={seed:02d} success={result['success']} steps={result['steps']} "
            f"contacts={result['collisions']} visible={result['visibility_ratio']:.2f} "
            f"min_goal={result['min_goal_distance']:.3f} penetrations={result['penetrations']}"
        )

    rate = successes / max(args.episodes, 1)
    print(
        f"summary: successes={successes}/{args.episodes} rate={rate:.1%} "
        f"contacts={contacts} penetrations={penetrations}"
    )
    return 0 if acceptance_passes(successes, args.episodes, penetrations, args.required_rate) else 1


if __name__ == "__main__":
    raise SystemExit(main())
