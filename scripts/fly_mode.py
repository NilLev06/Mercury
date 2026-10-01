"""Drive the neural runtime in the persistent passive MuJoCo viewer."""
import argparse
import sys
import time
from pathlib import Path
import mujoco.viewer
from fly_brain_rc.config import SIM, DRIVE
from fly_brain_rc.sim.env import FlyCarEnv
from fly_brain_rc.sim.scenarios import get_scenario, scenario_names
from fly_brain_rc.sim.kinematics import BicycleState
from fly_brain_rc.neural.brain import NeuralFlyBrain
from fly_brain_rc.runtime import FlyRuntime, load_genome, run_episode
from scripts.memory_store import open_memory, save_memory


def alive(viewer):
    return viewer.is_running() if callable(viewer.is_running) else bool(viewer.is_running)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--max_episodes', type=int, default=1)
    parser.add_argument('--scenario', choices=scenario_names())
    parser.add_argument('--close-on-finish', action='store_true')
    parser.add_argument('--follow', action='store_true', default=True)
    parser.add_argument('--genome')
    parser.add_argument('--target', choices=('red_ball',), default='red_ball')
    parser.add_argument('--dashboard', action='store_true')
    parser.add_argument('--memory', default=str(Path(__file__).resolve().parents[1] / '.local' / 'familiarity.json'))
    parser.add_argument('--reset-memory', action='store_true')
    args = parser.parse_args()
    try:
        memory = open_memory(args.memory, reset=args.reset_memory)
    except ValueError as exc:
        parser.error(str(exc))
    genome = load_genome(args.genome)
    env = FlyCarEnv(SIM, DRIVE, seed=args.seed,
                    scenario=get_scenario(args.scenario) if args.scenario else None)
    dashboard = None
    clean_shutdown = False
    try:
        if args.dashboard:
            import webbrowser
            from fly_brain_rc.telemetry import TelemetryBus
            from fly_brain_rc.dashboard.server import DashboardServer, MissionController
            bus, mission = TelemetryBus(), MissionController(running=True)
            dashboard = DashboardServer(bus, mission).start()
            webbrowser.open(f'http://{dashboard.address[0]}:{dashboard.address[1]}/')
        with mujoco.viewer.launch_passive(env.m, env.d) as viewer:
            viewer.cam.azimuth, viewer.cam.elevation, viewer.cam.distance = -30, -55, 5.5
            for episode in range(args.max_episodes):
                if not alive(viewer):
                    break
                if episode:
                    env.collisions = env.penetrations = env.sim_steps = 0
                    env.speed = env.heading = env._w_cmd = env._w_cmd_prev = 0.0
                    env._contact_active = False
                    env.state = BicycleState()
                    env.d.qpos[:] = 0
                    env._sync_kinematic()
                    env._last_encoder_position = env._encoder_position()
                runtime = FlyRuntime(env, NeuralFlyBrain(genome), target=args.target,
                                     memory=memory,
                                     emit=bus.publish if dashboard else None,
                                     mission=mission if dashboard else None)
                def display(record):
                    if not alive(viewer):
                        raise KeyboardInterrupt
                    if args.follow:
                        viewer.cam.lookat[:2] = env.car_pose()[:2]
                    viewer.sync()
                    time.sleep(DRIVE.timestep)
                if dashboard:
                    generation, active_steps = mission.generation, 0
                    while alive(viewer):
                        state = mission.state()
                        if state['generation'] != generation:
                            generation, active_steps = state['generation'], 0
                        record = runtime.tick()
                        if state['status'] == 'running':
                            active_steps += 1
                            if active_steps >= SIM.episode_steps and mission.state()['status'] == 'running':
                                mission.finish({'stable_arrival': False, 'reason': 'step_limit'}, generation=generation)
                        display(record)
                    break
                result = run_episode(runtime, SIM.episode_steps, on_tick=display)
                print(f'episode {episode}: {result}')
            if alive(viewer) and not args.close_on_finish:
                print('run complete; viewer stays open until Esc')
                while alive(viewer):
                    viewer.sync()
                    time.sleep(.05)
        clean_shutdown = True
    except KeyboardInterrupt:
        clean_shutdown = True
        raise
    finally:
        if dashboard:
            dashboard.close()
        env.renderer.close()
        if clean_shutdown:
            save_memory(memory, args.memory)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
