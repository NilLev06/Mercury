"""Manual driving of the RC car scene in the MuJoCo viewer.

By default runs the same kinematics as the fly brain (v/omega ramps,
sharp turning, 2D obstacle stop) so you can steer around the arena.
Use --physical for the honest low-level contact physics (the floor
contact model pins yaw, so in that mode the car only goes straight).

Usage:
    python scripts/drive_manual.py scenes/scene_seed0.xml
    python scripts/drive_manual.py scenes/scene_seed5.xml --physical

Controls:
    W / Up        throttle forward
    S / Down      throttle reverse
    A / Left      steer left
    D / Right     steer right
    R             reset the simulation
    Esc / Ctrl+C  quit
"""
import argparse
import math
import sys
import time

import mujoco
import mujoco.viewer

from fly_brain_rc.config import DRIVE

GLFW_UP = 265
GLFW_DOWN = 264
GLFW_LEFT = 263
GLFW_RIGHT = 262
GLFW_W = 87
GLFW_A = 65
GLFW_S = 83
GLFW_D = 68
GLFW_R = 82

HOLD_S = 0.16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", nargs="?", default="scenes/scene_seed0.xml")
    ap.add_argument("--physical", action="store_true", help="use raw contact physics (no turning)")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)

    drive = DRIVE
    steering = not args.physical
    last_press = {}

    def held(key):
        return time.time() - last_press.get(key, -9.0) < HOLD_S

    def on_key(key):
        last_press[key] = time.time()
        if key == GLFW_R:
            mujoco.mj_resetData(model, data)
            for i in range(model.nu):
                data.ctrl[i] = 0.0
            state.update(speed=0.0, heading=0.0, w_cmd=0.0)

    state = {"speed": 0.0, "heading": float(0.0), "w_cmd": 0.0}

    obstacles = []
    if steering:
        for b in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, b)
            if name and name.startswith("obstacle"):
                obstacles.append(tuple(model.geom_pos[b][:2]))

    def kinematic_step(throttle, steer, dt):
        w_cmd = steer * drive.max_omega
        st = state
        st["w_cmd"] += dt * max(-dt * 16.0, min(16.0 * dt, w_cmd - st["w_cmd"]))
        v_cmd = throttle * drive.max_speed
        st["speed"] += dt * max(-8.0 * dt, min(8.0 * dt, v_cmd - st["speed"]))
        st["heading"] += st["w_cmd"] * dt
        nx = data.qpos[0] + st["speed"] * math.cos(st["heading"]) * dt
        ny = data.qpos[1] + st["speed"] * math.sin(st["heading"]) * dt
        if any(math.hypot(nx - ox, ny - oy) < 0.33 for ox, oy in obstacles):
            return  # blocked by obstacle, do not move
        data.qpos[0] = nx
        data.qpos[1] = ny
        data.qpos[2] = 0.05
        h = st["heading"]
        data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6] = (
            math.cos(h / 2), 0.0, math.sin(h / 2), math.cos(h / 2))
        jq_steer = model.jnt_qposadr[model.joint("steer_joint").id]
        data.qpos[jq_steer] = steer * drive.max_steer_rad
        spin = st["speed"] / drive.wheel_radius
        for name in ("wheel_fl", "wheel_fr"):
            j = model.jnt_qposadr[model.joint(name).id]
            data.qpos[j] += spin * dt
        mujoco.mj_forward(model, data)

    print(f"scene: {args.scene}  mode: {'kinematic (fun)' if steering else 'physical (honest)'}")
    print("controls: W/S throttle, A/D steer, R reset, Esc quit")

    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        dt = model.opt.timestep
        while viewer.is_alive():
            step_start = time.time()
            th = (1.0 if held(GLFW_W) or held(GLFW_UP) else 0.0) - (
                1.0 if held(GLFW_S) or held(GLFW_DOWN) else 0.0)
            st = (1.0 if held(GLFW_A) or held(GLFW_LEFT) else 0.0) - (
                1.0 if held(GLFW_D) or held(GLFW_RIGHT) else 0.0)
            if steering:
                kinematic_step(float(th), float(st), dt)
            else:
                data.ctrl[model.actuator("motor_fl").id] = 0.55 * th
                data.ctrl[model.actuator("motor_fr").id] = 0.55 * th
                data.ctrl[model.actuator("steer").id] = 0.25 * st
                mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(max(0.0, dt - (time.time() - step_start)))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)