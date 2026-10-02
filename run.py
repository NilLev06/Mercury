"""Single entry point for the simulator, evaluations, and tests."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parent


def _mjpython_path() -> str:
    beside_python = Path(sys.executable).with_name("mjpython")
    if beside_python.exists():
        return str(beside_python)
    found = shutil.which("mjpython")
    if found:
        return found
    raise SystemExit("mjpython not found; activate the flybrain conda environment")


def _run_simulator(args: argparse.Namespace) -> int:
    if sys.platform == "darwin" and "MJPYTHON_BIN" not in os.environ:
        command = [_mjpython_path(), str(Path(__file__).resolve()), "sim"]
        if args.scenario:
            command += ["--scenario", args.scenario]
        command += ["--seed", str(args.seed), "--episodes", str(args.episodes)]
        command += ["--target", args.target]
        if args.genome:
            command += ["--genome", args.genome]
        if args.dashboard:
            command += ['--dashboard']
        if args.memory is not None:
            command += ['--memory', args.memory]
        if args.reset_memory:
            command += ['--reset-memory']
        os.execv(command[0], command)

    from scripts.fly_mode import main as fly_main

    sys.argv = ["scripts.fly_mode", "--seed", str(args.seed), "--max_episodes", str(args.episodes)]
    sys.argv += ["--target", args.target]
    if args.genome:
        sys.argv += ["--genome", args.genome]
    if args.scenario:
        sys.argv += ["--scenario", args.scenario]
    if args.dashboard:
        sys.argv += ['--dashboard']
    if args.memory is not None:
        sys.argv += ['--memory', args.memory]
    if args.reset_memory:
        sys.argv += ['--reset-memory']
    fly_main()
    return 0


def _run_rl_trainer(args: argparse.Namespace) -> int:
    """Use MuJoCo's macOS Python trampoline for offscreen camera training."""
    if sys.platform == "darwin" and "MJPYTHON_BIN" not in os.environ:
        command = [_mjpython_path(), str(Path(__file__).resolve()), "train-rl",
                   "--episodes", str(args.episodes), "--steps", str(args.steps),
                   "--seed", str(args.seed), "--epsilon", str(args.epsilon), "--out", args.out]
        os.execv(command[0], command)

    from dataclasses import asdict
    import json
    from fly_brain_rc.rl_training import train_q_learning
    result = train_q_learning(episodes=args.episodes, steps=args.steps, output=args.out,
                              seed=args.seed, epsilon=args.epsilon)
    print(json.dumps(asdict(result), allow_nan=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Digital Fly Car launcher")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sim = subparsers.add_parser("sim", help="open the interactive MuJoCo viewer")
    sim.add_argument("--scenario", choices=("random", "blocked_goal", "corridor", "slalom", "turn_corridor"), default="corridor")
    sim.add_argument("--seed", type=int, default=0)
    sim.add_argument("--episodes", type=int, default=1)
    sim.add_argument('--dashboard', action='store_true')
    sim.add_argument('--memory', metavar='PATH', help='opt in to persistent visual familiarity')
    sim.add_argument('--reset-memory', action='store_true', help='back up stored familiarity and start fresh')
    dashboard = subparsers.add_parser('dashboard', help='open a running simulation dashboard')
    dashboard.add_argument('--attach', default='127.0.0.1:8765', metavar='HOST:PORT')

    evaluate = subparsers.add_parser("evaluate", help="run repeatable navigation benchmarks")
    evaluate.add_argument("--suite", choices=("train", "type-a", "type-b", "random", "scenarios"), default="type-a")
    evaluate.add_argument("--scenario", choices=("blocked_goal", "corridor", "slalom", "turn_corridor"))
    evaluate.add_argument("--episodes", type=int, default=10,
                          help="number of generated maps for --suite random (fixed scenarios run once each)")

    from fly_brain_rc.config import SIM
    evaluate.add_argument('--steps', type=int, default=SIM.episode_steps)
    evaluate.add_argument('--start-seed', type=int, default=0)
    evaluate.add_argument('--required-rate', type=float)
    for command_parser in (sim, evaluate):
        command_parser.add_argument('--genome')
        command_parser.add_argument('--target', choices=('red_ball',), default='red_ball')
    subparsers.add_parser("test", help="run the automated test suite")
    evolution = subparsers.add_parser('evolve', help='evolve neural parameters on training maps')
    evolution.add_argument('--config', required=True)
    evolution.add_argument('--out', required=True)
    train_rl = subparsers.add_parser('train-rl', help='train the exploration Q-policy in MuJoCo')
    train_rl.add_argument('--episodes', type=int, default=120)
    train_rl.add_argument('--steps', type=int, default=SIM.episode_steps)
    train_rl.add_argument('--seed', type=int, default=0)
    train_rl.add_argument('--epsilon', type=float, default=.25)
    train_rl.add_argument('--out', default='artifacts/rl/q-policy.json')
    args = parser.parse_args()
    if args.command == 'sim' and args.reset_memory and args.memory is None:
        parser.error('--reset-memory requires --memory PATH')

    os.chdir(PROJECT_ROOT)
    os.environ.setdefault("PYTHONPATH", str(PROJECT_ROOT))
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mpl-flybrain")

    if args.command == "sim":
        if args.scenario == "random":
            args.scenario = None
        return _run_simulator(args)
    if args.command == 'evolve':
        import json
        from dataclasses import fields
        from fly_brain_rc.evolution import EvolutionConfig, evolve
        from scripts.evaluate import training_evaluator
        try:
            payload = json.loads(Path(args.config).read_text())
            if set(payload) - {f.name for f in fields(EvolutionConfig)}:
                raise ValueError('unknown evolution configuration fields')
            output = Path(args.out)
            output.mkdir(parents=True, exist_ok=True)
            payload.setdefault('checkpoint', str(output / 'checkpoint.json'))
            config = EvolutionConfig(**payload)
            state = evolve(config, training_evaluator(config.steps))
            best = max(zip(state.population, state.scores), key=lambda pair: pair[1].rank_key())
            (output / 'best-genome.json').write_text(best[0].to_json())
            print(json.dumps({'generation': state.generation, 'best': best[1].__dict__}, allow_nan=False))
            return 0
        except (ValueError, TypeError, OSError) as exc:
            parser.error(str(exc))
    if args.command == 'train-rl':
        if args.episodes <= 0 or args.steps <= 0 or not 0 <= args.epsilon <= 1:
            parser.error('--episodes and --steps must be positive; --epsilon must be in [0, 1]')
        return _run_rl_trainer(args)
    if args.command == 'dashboard':
        import webbrowser
        from urllib.parse import urlsplit
        address = urlsplit('http://' + args.attach)
        try:
            if not address.hostname or address.port is None or address.path or address.query or address.fragment or address.username:
                raise ValueError
        except ValueError:
            parser.error('--attach must be HOST:PORT')
        webbrowser.open('http://' + args.attach + '/')
        return 0
    if args.command == "evaluate":
        command = [
            sys.executable, "-m", "scripts.evaluate", "--suite", args.suite,
            "--episodes", str(args.episodes),
            '--steps', str(args.steps), '--start-seed', str(args.start_seed),
            '--target', args.target,
        ]
        if args.required_rate is not None:
            command += ['--required-rate', str(args.required_rate)]
        if args.genome:
            command += ['--genome', args.genome]
        if args.scenario:
            command += ["--scenario", args.scenario]
        return subprocess.call(command)
    return subprocess.call(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(0)
