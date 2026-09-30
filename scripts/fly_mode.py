"""Fly brain drives the car autonomously, shown live in the MuJoCo viewer.

The brain (reflex + looming + goal/flow) navigates to the goal in
kinematic mode; the viewer only visualizes. Esc closes the window.

Usage:
    mjpython -m scripts.fly_mode --seed 0
    mjpython -m scripts.fly_mode --seed 5 --max_episodes 3
"""
import argparse
import sys
import time

import numpy as np
import mujoco
import mujoco.viewer  # noqa: F401

from fly_brain_rc.config import SIM, DRIVE
from fly_brain_rc.sim.env import FlyCarEnv
from fly_brain_rc.sim.kinematics import BicycleState
from scripts.run_sim import build_brain


def soft_reset(env, brain):
    brain = build_brain()
    d = env.d
    env.sim_steps = 0
    env.collisions = 0
    env.speed = 0.0
    env.heading = 0.0
    env._w_cmd = 0.0
    env._w_cmd_prev = 0.0
    env._contact_active = False
    for i in range(env.m.nq):
        d.qpos[i] = 0.0
    d.qpos[2] = 0.05
    d.qpos[3] = 1.0
    env.state = BicycleState()
    env._sync_kinematic()
    return brain


def alive(v):
    r = v.is_running
    return r() if callable(r) else bool(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_episodes", type=int, default=4)
    ap.add_argument("--follow", action="store_true", default=True,
                    help="camera follows the car (top-down view)")
    args = ap.parse_args()

    env = FlyCarEnv(SIM, DRIVE, seed=args.seed)
    brain = build_brain()
    dt = DRIVE.timestep

    print(f"fly mode: seed={args.seed}  max_episodes={args.max_episodes}")
    print("control: W/S throttle-A/D steer-NONE (autonomous brain). Esc quits.")

    with mujoco.viewer.launch_passive(env.m, env.d) as viewer:
        viewer.cam.azimuth = -30
        viewer.cam.elevation = -55
        viewer.cam.distance = 5.5
        for ep in range(args.max_episodes):
            n = 0
            if not alive(viewer):
                break
            brain = soft_reset(env, brain) if ep > 0 else brain
            succ = False
            for i in range(SIM.episode_steps):
                if not alive(viewer):
                    break
                step_start = time.time()
                obs = env.observe()
                o = obs["odom"]
                th, st = brain.step(
                    obs["frame"], obs["gray"], obs["ranges"], o["yaw_rate_deg"], dt
                )
                obs = env.step(th, st)
                if args.follow:
                    x, y, _ = env.car_pose()
                    viewer.cam.lookat[:2] = [x, y]
                viewer.sync()
                elapsed = time.time() - step_start
                if obs["odom"]["dist_to_goal"] < SIM.goal_radius:
                    succ, n = True, i + 1
                    break
                time.sleep(max(0.0, dt - elapsed))
            else:
                n = SIM.episode_steps
            print(f"episode {ep}: success={succ} steps={n} collisions={env.collisions} "
                  f"time={n * dt:.1f}s")
            if not alive(viewer):
                break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
