"""Print sparse sensor/motor diagnostics for one training scene only."""
import argparse
import json
from dataclasses import asdict

from fly_brain_rc.config import DRIVE, SIM
from fly_brain_rc.neural.brain import NeuralFlyBrain
from fly_brain_rc.runtime import FlyRuntime, load_genome, run_episode
from fly_brain_rc.sim.env import FlyCarEnv
from fly_brain_rc.sim.suites import training_cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--steps', type=int, default=2400)
    parser.add_argument('--genome')
    parser.add_argument('--occluded', action='store_true')
    args = parser.parse_args()
    case = training_cases(args.seed, args.index + 1, occluded=args.occluded)[args.index]
    env = FlyCarEnv(SIM, DRIVE, seed=case.seed, scenario=case.scenario)
    def consume(record):
        if record.sequence % 100 == 0:
            print(json.dumps({'time': record.timestamp, 'target': None if record.target is None
                else [record.target.bearing, record.target.apparent_area], 'tof': record.sensor.tof,
                'traction': record.command.traction, 'steering': record.command.steering,
                'familiarity': record.brain.familiarity, 'exploration': record.brain.populations['exploration'],
                'threat': record.brain.threat}), flush=True)
    try:
        result = run_episode(FlyRuntime(env, NeuralFlyBrain(load_genome(args.genome))), args.steps, on_tick=consume)
        print(json.dumps(asdict(result)), flush=True)
    finally:
        env.renderer.close()


if __name__ == '__main__':
    main()
