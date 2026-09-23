from ..config import SimConfig, DriveConfig


def build_model_xml(cfg: SimConfig, drive: DriveConfig, obstacles, goal, steer_actuator: bool = False):
    o = cfg.arena_half
    ob = "\n".join(
        f'<geom name="obstacle_{i}" type="cylinder" size="{cfg.obstacle_radius:.3f} {cfg.obstacle_height / 2:.3f}" '
        f'pos="{x:.3f} {y:.3f} {cfg.obstacle_height / 2:.3f}" '
        f'rgba="{r} {g} {b} 1" friction="0.15 0.02 0.01" solref="0.05 1.2"/>'
        for i, ((x, y), (r, g, b)) in enumerate(obstacles)
    )
    steer_act = ""
    if steer_actuator:
        steer_act = (
            f'\n    <position name="steer" joint="steer_joint" kp="5" kv="1" '
            f'ctrlrange="-{drive.max_steer_rad:.3f} {drive.max_steer_rad:.3f}" ctrllimited="true"/>'
        )
    gx, gy = goal
    tr = drive.track / 2
    wr = drive.wheel_radius
    return f"""<mujoco model="flybraincare">
  <compiler angle="radian" coordinate="local" autolimits="true"/>
  <option timestep="{drive.timestep}" gravity="0 0 -9.81" integrator="Euler"/>
  <default>
    <geom type="box" density="1500"/>
    <joint damping="0.05"/>
    <motor ctrlrange="-0.6 0.6" ctrllimited="true"/>
  </default>
  <worldbody>
    <geom name="floor" type="plane" size="{o + 1} {o + 1} 0.01" material="floor"/>
    <geom name="wall_n" type="box" size="{o + 0.4} 0.15 0.3" pos="0 {o + 0.15} 0.3" rgba="0.5 0.5 0.55 1"/>
    <geom name="wall_s" type="box" size="{o + 0.4} 0.15 0.3" pos="0 {-o - 0.15} 0.3" rgba="0.5 0.5 0.55 1"/>
    <geom name="wall_e" type="box" size="0.15 {o + 0.4} 0.3" pos="{o + 0.15} 0 0.3" rgba="0.5 0.5 0.55 1"/>
    <geom name="wall_w" type="box" size="0.15 {o + 0.4} 0.3" pos="{-o - 0.15} 0 0.3" rgba="0.5 0.5 0.55 1"/>
    {ob}
    <geom type="sphere" size="0.22" pos="{gx:.3f} {gy:.3f} 0.22" rgba="1 0.25 0.2 1"/>
    <body name="chassis" pos="0 0 0.01">
      <freejoint name="chassis_free"/>
      <geom type="box" size="0.15 0.08 0.024" pos="0 0 0.024" rgba="0.25 0.5 0.8 1"/>
      <body name="front_axle" pos="0.09 0 -0.015">
        <joint name="steer_joint" type="hinge" axis="0 0 1" armature="0.002"/>
        <geom type="sphere" size="0.006" mass="0.01" contype="0" conaffinity="0"/>
        <body name="wheel_fl" pos="0 {tr} 0">
          <joint name="wheel_fl" type="hinge" axis="0 1 0"/>
          <geom type="cylinder" size="{wr} 0.022" zaxis="0 1 0" pos="0 0 0" rgba="0.15 0.15 0.15 1" friction="0.5 0.01 0.01"/>
        </body>
        <body name="wheel_fr" pos="0 -{tr} 0">
          <joint name="wheel_fr" type="hinge" axis="0 1 0"/>
          <geom type="cylinder" size="{wr} 0.022" zaxis="0 1 0" pos="0 0 0" rgba="0.15 0.15 0.15 1" friction="0.5 0.01 0.01"/>
        </body>
      </body>
      <body name="caster_l" pos="-0.08 {tr} -0.015">
        <joint name="caster_l" type="ball" armature="0.0005"/>
        <geom type="sphere" size="{wr}" pos="0 0 0" rgba="0.12 0.12 0.12 1" friction="0.05 0.01 0.001"/>
      </body>
      <body name="caster_r" pos="-0.08 -{tr} -0.015">
        <joint name="caster_r" type="ball" armature="0.0005"/>
        <geom type="sphere" size="{wr}" pos="0 0 0" rgba="0.12 0.12 0.12 1" friction="0.05 0.01 0.001"/>
      </body>
      <site name="sF_L" pos="0.20 0.11 0.12" size="0.006" xyaxes="-0.5 0.874 0 0 0 1"/>
      <site name="sF_R" pos="0.20 -0.11 0.12" size="0.006" xyaxes="0.5 0.874 0 0 0 1"/>
      <site name="sF_C" pos="0.22 0 0.12" size="0.006" xyaxes="0 1 0 0 0 1"/>
      <site name="sO" pos="0 0 0.05" size="0.01"/>
      <camera name="eye" pos="0.03 0 0.12" xyaxes="0 1 0 0 0 -1" mode="fixed"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="motor_fl" joint="wheel_fl"/>
    <motor name="motor_fr" joint="wheel_fr"/>{steer_act}
  </actuator>
  <sensor>
    <rangefinder name="b_fl" site="sF_L" cutoff="1.2"/>
    <rangefinder name="b_fr" site="sF_R" cutoff="1.2"/>
    <rangefinder name="b_fc" site="sF_C" cutoff="1.2"/>
    <gyro name="gyro" site="sO"/>
  </sensor>
  <asset>
    <material name="floor" rgba="0.85 0.85 0.8 1"/>
  </asset>
</mujoco>"""