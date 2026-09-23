import argparse
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fly_brain_rc.config import DRIVE, RING, FLOW, LOOM, REFLEX, SIM
from fly_brain_rc.sim.env import FlyCarEnv
from fly_brain_rc.brain.ring_attractor import RingAttractor
from fly_brain_rc.brain.optic_flow import OpticFlowBrain, H1Steering
from fly_brain_rc.brain.looming import LoomingDetector
from fly_brain_rc.brain.reflex import BristleReflex
from fly_brain_rc.brain.controller import FlyBrainController


def build_brain():
    ring = RingAttractor(RING)
    flow_brain = OpticFlowBrain(FLOW)
    h1 = H1Steering(FLOW.steer_gain, FLOW.velocity_gain)
    loom = LoomingDetector(LOOM)
    reflex = BristleReflex(REFLEX)
    brain = FlyBrainController(DRIVE, ring, flow_brain, h1, loom, reflex)
    return brain


def run_episode(brain: FlyBrainController, env: FlyCarEnv, steps: int, seed, debug_dir=None):
    obs = env.reset()
    traj = []
    goals_hit = False
    throttle_log = []
    steer_log = []
    t = 0.0
    for i in range(steps):
        gb = obs["odom"]["goal_bearing_deg"]
        yaw_rate = obs["odom"]["yaw_rate_deg"]
        throttle, steer = brain.step(obs["gray"], obs["ranges"], gb, yaw_rate, DRIVE.timestep, obs["odom"])
        obs = env.step(throttle, steer)
        x, y, yaw = env.car_pose()
        traj.append((x, y))
        throttle_log.append(throttle)
        steer_log.append(steer)
        t += DRIVE.timestep
        if obs["odom"]["dist_to_goal"] < SIM.goal_radius:
            goals_hit = True
            break
    return dict(traj=traj, success=goals_hit, steps=i + 1, time=t,
                collisions=env.collisions, throttle=throttle_log, steer=steer_log)


def plot_episode(env, res, path):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    o = SIM.arena_half
    ax[0].set_xlim(-o - 0.4, o + 0.4)
    ax[0].set_ylim(-o - 0.4, o + 0.4)
    ax[0].set_aspect("equal")
    for (x, y), col in env.obstacles:
        circ = plt.Circle((x, y), SIM.obstacle_radius, color=col, alpha=0.9)
        ax[0].add_patch(circ)
    gx, gy = env.goal
    ax[0].scatter([gx], [gy], marker="*", s=260, c="red")
    sx, sy, _ = env.start_pose
    ax[0].scatter([sx], [sy], marker="o", s=60, c="k")
    tr = np.array(res["traj"])
    ax[0].plot(tr[:, 0], tr[:, 1], lw=1.8, c="b")
    ax[0].set_title(f"Trajectory  success={res['success']} collisions={res['collisions']} steps={res['steps']}")
    steps = np.arange(len(res["throttle"]))
    ax[1].plot(steps, res["throttle"], label="throttle")
    ax[1].plot(steps, res["steer"], label="steer")
    ax[1].set_xlabel("step"); ax[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--out", default="figures/run.png")
    args = ap.parse_args()
    steps = args.steps or SIM.episode_steps
    brain = build_brain()
    env = FlyCarEnv(SIM, DRIVE, seed=args.seed)
    stats = []
    for ep in range(args.episodes):
        res = run_episode(brain, env, steps, args.seed + ep)
        print(f"episode {ep}: success={res['success']} collisions={res['collisions']} "
              f"steps={res['steps']} time={res['time']:.1f}s")
        plot_episode(env, res, f"figures/ep{ep}_{args.out if False else args.out.split('/')[-1]}")
        stats.append(res["success"])
    print(f"success rate: {sum(stats)}/{len(stats)}")


if __name__ == "__main__":
    main()