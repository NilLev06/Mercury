"""Read-only interactive playback of the saved complex-map benchmark."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def benchmark_cases():
    from fly_brain_rc.imitation.navigation_cases import mixed_cases
    return [item for seed in range(950001, 950005)
            for item in mixed_cases(seed, 'final', 2) if item[2] == 'complex']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map', type=int, default=1, choices=range(1, 33))
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--checkpoint', type=Path,
                        default=ROOT/'artifacts/imitation/versions/main-23of32.pt')
    args = parser.parse_args()
    cases = benchmark_cases()
    if args.list:
        for index, (case, _, _) in enumerate(cases, 1):
            print(f'{index}: {case.name}')
        return
    import torch
    import mujoco.viewer
    from fly_brain_rc.config import SIM, DRIVE
    from fly_brain_rc.imitation.model import load_checkpoint
    from fly_brain_rc.imitation.curriculum import place_start
    from fly_brain_rc.imitation.runner import collect_episode
    from fly_brain_rc.sim.env import FlyCarEnv
    torch.set_num_threads(2)
    raw = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model, *_ = load_checkpoint(args.checkpoint, raw['graph'])
    number = args.map
    while True:
        case, pose, _ = cases[number-1]
        print(f'Карта {number}/32: {case.name}. Управляет только нейросеть.', flush=True)
        env = FlyCarEnv(SIM, DRIVE, scenario=case, render=True)
        viewer = None
        try:
            place_start(env, pose)
            viewer = mujoco.viewer.launch_passive(env.m, env.d)
            viewer.cam.distance = 5
            viewer.cam.azimuth = 90
            viewer.cam.elevation = -65
            viewer.cam.lookat[:] = [1.3, 0, 0]
            _, report = collect_episode(env, model, steps=2400, beta=None, viewer=viewer)
            print(json.dumps(report, ensure_ascii=False), flush=True)
            print('Закрой окно MuJoCo, чтобы выбрать другую карту.', flush=True)
            while viewer.is_running():
                viewer.sync()
                time.sleep(.02)
        finally:
            if viewer is not None:
                viewer.close()
            if env.renderer is not None:
                env.renderer.close()
        while True:
            try:
                choice = input('Номер карты 1–32, Enter — следующая, q — выход: ').strip()
            except EOFError:
                return
            if choice.lower() in ('q', 'й', 'exit'):
                return
            if not choice:
                number = number % 32 + 1
                break
            if choice.isdecimal() and 1 <= int(choice) <= 32:
                number = int(choice)
                break
            print('Введите номер от 1 до 32.')


if __name__ == '__main__':
    main()
