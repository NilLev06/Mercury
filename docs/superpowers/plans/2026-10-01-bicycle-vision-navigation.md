# Bicycle Vision Navigation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a four-wheel bicycle-model RC car that searches for and approaches a red visual target without receiving its ground-truth bearing, while respecting solid obstacles.

**Architecture:** Keep deterministic vehicle geometry in a pure-Python kinematics module, with MuJoCo acting as renderer and sensor host. A focused OpenCV goal detector feeds a deterministic search/approach/avoid controller; the simulator alone retains target coordinates for termination and metrics.

**Tech Stack:** Python 3.11, NumPy, OpenCV, MuJoCo, pytest, matplotlib

## Global Constraints

- The controller must not consume ground-truth goal bearing or distance.
- MuJoCo free-joint quaternions use `(w, x, y, z)` and planar rotation around Z.
- Kinematic collisions must not permit obstacle or wall penetration.
- Search behavior and evaluation must be deterministic.
- Acceptance target: zero penetrations and at least 15 successes in 20 fixed reachable episodes.

---

### Task 1: Pure bicycle kinematics

**Files:**
- Create: `fly_brain_rc/sim/kinematics.py`
- Create: `tests/test_kinematics.py`
- Modify: `fly_brain_rc/config.py`

**Interfaces:**
- Produces: `Pose2D`, `BicycleState`, `step_bicycle(state, throttle, steer, drive, dt)`, `yaw_quaternion(yaw)`.

- [ ] Write failing tests for straight travel, signed turn radius, acceleration limits, and unit Z-axis quaternion.
- [ ] Run `PYTHONPATH=. /opt/miniconda3/envs/flybrain/bin/python -m pytest tests/test_kinematics.py -v` and verify failures identify missing interfaces.
- [ ] Implement the smallest pure functions and add `wheelbase`, `linear_accel`, `linear_decel`, and `collision_radius` configuration.
- [ ] Run the tests and refactor only after green.

### Task 2: Four-wheel MuJoCo chassis and solid contacts

**Files:**
- Modify: `fly_brain_rc/sim/car_model.py`
- Modify: `fly_brain_rc/sim/env.py`
- Create: `tests/test_environment.py`

**Interfaces:**
- Consumes: `BicycleState`, `step_bicycle`, `yaw_quaternion`.
- Produces: `FlyCarEnv.step()` with rollback collision constraint and contact-edge counting.

- [ ] Write failing environment tests for chassis layout, quaternion readback, obstacle rollback, wall rollback, reverse escape, and one count per continuous contact.
- [ ] Run the environment tests under the `flybrain` environment and observe the current failures.
- [ ] Replace rear casters with rear wheel hinges, place the body relative to the rear axle, and bind all wheel and steering indices.
- [ ] Route kinematic stepping through the pure bicycle model; reject colliding candidate poses before syncing MuJoCo.
- [ ] Run Task 1 and Task 2 tests until green.

### Task 3: Visual goal detector

**Files:**
- Create: `fly_brain_rc/brain/goal_vision.py`
- Create: `tests/test_goal_vision.py`

**Interfaces:**
- Produces: `GoalObservation(visible, bearing, area, confidence)` and `GoalDetector.step(frame_rgb)`.

- [ ] Write synthetic-frame tests for absent, centered, left, right, tiny-noise, and two-blob cases.
- [ ] Run the tests and verify they fail because the detector is missing.
- [ ] Implement dual red HSV bands, morphological cleanup, and largest-contour selection.
- [ ] Run detector tests until green.

### Task 4: Deterministic search, approach, and avoid controller

**Files:**
- Modify: `fly_brain_rc/brain/controller.py`
- Modify: `scripts/run_sim.py`
- Modify: `scripts/fly_mode.py`
- Create: `tests/test_controller.py`

**Interfaces:**
- Consumes: RGB frame, grayscale frame, range readings, yaw rate, and timestep.
- Produces: `(throttle, steer)` plus public `mode` in `search`, `approach`, or `avoid`.

- [ ] Write failing tests showing the controller turns toward a visible goal, searches deterministically when absent, and gives obstacle avoidance priority.
- [ ] Add a regression test whose control output is unchanged when diagnostic ground-truth goal fields change.
- [ ] Run tests and confirm the old ground-truth-dependent controller fails them.
- [ ] Integrate `GoalDetector`; implement deterministic timed search sweep, target-memory timeout, approach speed scaling, and existing reflex priority.
- [ ] Update both run scripts to pass RGB frames and remove goal-bearing arguments.
- [ ] Run all unit tests until green.

### Task 5: Metrics, repeatable evaluation, and launch verification

**Files:**
- Modify: `scripts/run_sim.py`
- Create: `scripts/evaluate.py`
- Create: `tests/test_run_sim.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces: per-episode success, distinct contacts, visibility ratio, mode durations, and minimum goal distance; `scripts/evaluate.py --episodes 20` exits nonzero below acceptance thresholds.

- [ ] Write failing tests for metric accumulation and deterministic seed scheduling.
- [ ] Implement the metrics and headless evaluator without exposing goal truth to the controller.
- [ ] Run the full test suite.
- [ ] Run 20 fixed episodes, inspect failures, and tune only controller/config parameters backed by the recorded metrics.
- [ ] Repeat until zero penetrations and at least 15/20 successes, or report measured architectural limits if the target cannot be reached without expanding the approved design.
- [ ] Launch `scripts/fly_mode.py --seed 0` in MuJoCo and verify the viewer runs the new controller without exceptions.
