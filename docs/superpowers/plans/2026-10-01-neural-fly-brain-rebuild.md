# Neural Fly-Brain Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the rule-based controller with an interpretable rate-based neural fly brain, persistent familiarity, live simulation dashboard, and reproducible offline parameter evolution.

**Architecture:** Keep the MuJoCo scene and bicycle backend, but place typed sensor, perception, brain, actuator, and telemetry contracts between components. Build and test a hand-configured neural genome before adding fixed-topology evolutionary optimization. Run the dashboard from the same telemetry stream used by record/replay and evaluation.

**Tech Stack:** Python 3.11, NumPy, OpenCV, MuJoCo, Python `unittest`, standard-library `ThreadingHTTPServer`, and Server-Sent Events.

## Global Constraints

- The brain may not consume target coordinates, target distance, absolute pose, map labels, or a planned route.
- The simulator uses a red-ball mock detector behind the same `ObjectObservation` interface intended for a future YOLO detector.
- The runtime contains no `search`, `approach`, or `avoid` behavior-mode enum.
- The vehicle remains bicycle/Ackermann: one normalized traction command and one normalized steering-servo command.
- Motor topology and connection signs remain fixed during evolution; only bounded numerical parameters evolve.
- Type-A held-out acceptance is at least 90%; type-B held-out acceptance is at least 70%; any collision fails an episode.
- A successful arrival requires three continuous seconds of centered target, safe proximity, and near-zero motion.

---

## File Structure

New runtime modules:

- `fly_brain_rc/contracts.py`: immutable cross-layer data contracts.
- `fly_brain_rc/perception/target.py`: detector protocol and red-ball mock.
- `fly_brain_rc/perception/motion.py`: optic-flow and looming signals.
- `fly_brain_rc/perception/familiarity.py`: visual descriptors and bounded persistent memory.
- `fly_brain_rc/neural/dynamics.py`: simultaneous rate-population update primitives.
- `fly_brain_rc/neural/compass.py`: compass ring and path integration.
- `fly_brain_rc/neural/brain.py`: fixed neural topology and motor readout.
- `fly_brain_rc/neural/genome.py`: versioned bounded parameter set.
- `fly_brain_rc/runtime.py`: sensor-to-actuator loop and stable-arrival tracking.
- `fly_brain_rc/telemetry.py`: in-process pub/sub, records, and deterministic replay.
- `fly_brain_rc/dashboard/server.py`: HTTP control endpoints and telemetry stream.
- `fly_brain_rc/dashboard/static/index.html`: dependency-free diagnostic UI.
- `fly_brain_rc/evolution.py`: seeded population evaluation and mutation.
- `fly_brain_rc/sim/suites.py`: separate train, type-A test, and type-B test definitions.

Existing modules retained and modified:

- `fly_brain_rc/sim/env.py`: emit `SensorFrame`, accept `ActuatorCommand`, and expose no ground truth to the brain.
- `fly_brain_rc/sim/scenarios.py`: replace ad hoc cases with train/type-A/type-B scenario factories.
- `scripts/fly_mode.py`, `scripts/evaluate.py`, `run.py`: use the new runtime and commands.
- `README.md`, `Digital Fly Car.command`: document and launch the rebuilt system.

Old controller modules removed after replacement tests pass:

- `fly_brain_rc/brain/controller.py`
- `fly_brain_rc/brain/reflex.py`
- `fly_brain_rc/brain/ring_attractor.py`
- `fly_brain_rc/brain/goal_vision.py`
- `fly_brain_rc/brain/looming.py`
- `fly_brain_rc/brain/optic_flow.py`

---

### Task 1: Typed runtime contracts and ground-truth isolation

**Files:**
- Create: `fly_brain_rc/contracts.py`
- Create: `tests/test_contracts.py`
- Modify: `fly_brain_rc/sim/env.py`
- Modify: `tests/test_environment.py`

**Interfaces:**
- Produces: `SensorFrame`, `ObjectObservation`, `MotionObservation`, `ActuatorCommand`, `BrainTelemetry`, and `RuntimeTelemetry` frozen dataclasses.
- Produces: `FlyCarEnv.sensor_frame() -> SensorFrame` and `FlyCarEnv.apply(command: ActuatorCommand) -> SensorFrame`.
- Constraint: no brain-facing dataclass has fields named `goal`, `goal_x`, `goal_y`, `goal_distance`, `pose`, `x`, `y`, or `absolute_yaw`.

- [ ] **Step 1: Write failing contract tests**

```python
class ContractTests(unittest.TestCase):
    def test_sensor_frame_exposes_only_embodied_sensors(self):
        names = {field.name for field in dataclasses.fields(SensorFrame)}
        self.assertEqual(names, {"rgb", "gray", "yaw_rate", "encoder_delta", "tof", "contact", "timestamp"})
        self.assertTrue(names.isdisjoint({"goal", "goal_distance", "x", "y", "pose", "absolute_yaw"}))

    def test_actuator_command_clamps_outputs(self):
        command = ActuatorCommand(traction=3.0, steering=-2.0, timestamp=1.0, sequence=4)
        self.assertEqual(command.traction, 1.0)
        self.assertEqual(command.steering, -1.0)
```

- [ ] **Step 2: Run the tests and confirm missing imports fail**

Run: `python -m unittest tests.test_contracts -v`
Expected: `ModuleNotFoundError: fly_brain_rc.contracts`.

- [ ] **Step 3: Implement frozen contracts and environment adapters**

Define exact fields from the test. Clamp `ActuatorCommand` values in `__post_init__` with `object.__setattr__`. `sensor_frame()` renders the camera and reads simulated yaw rate, encoder displacement, center rangefinder as ToF, and contact. `apply()` calls the existing kinematic step and returns the next sensor frame.

- [ ] **Step 4: Add an integration test proving ground truth stays outside the brain contract**

Instantiate a render-disabled environment for geometry tests and a rendered environment for `sensor_frame`. Assert that evaluation can still call `env.goal_distance()` while `SensorFrame` cannot expose it.

- [ ] **Step 5: Run contract and environment tests**

Run: `python -m unittest tests.test_contracts tests.test_environment -v`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add fly_brain_rc/contracts.py fly_brain_rc/sim/env.py tests/test_contracts.py tests/test_environment.py
git commit -m "refactor: isolate embodied runtime contracts"
```

### Task 2: Replace perception with backend-neutral observations

**Files:**
- Create: `fly_brain_rc/perception/__init__.py`
- Create: `fly_brain_rc/perception/target.py`
- Create: `fly_brain_rc/perception/motion.py`
- Create: `fly_brain_rc/perception/familiarity.py`
- Create: `tests/test_perception.py`

**Interfaces:**
- Consumes: `SensorFrame`.
- Produces: `TargetDetector.detect(frame: SensorFrame, target_class: str) -> ObjectObservation | None`.
- Produces: `MotionEstimator.update(frame: SensorFrame, dt: float) -> MotionObservation`.
- Produces: `VisualDescriptor.encode(gray: np.ndarray) -> np.ndarray` with fixed shape `(64,)` and unit norm.

- [ ] **Step 1: Write failing perception tests**

```python
def test_red_ball_detector_emits_backend_neutral_observation(self):
    rgb = np.zeros((96, 128, 3), dtype=np.uint8)
    cv2.circle(rgb, (96, 48), 14, (255, 0, 0), -1)
    obs = MockRedBallDetector().detect(make_frame(rgb), "red_ball")
    self.assertEqual(obs.class_name, "red_ball")
    self.assertGreater(obs.bearing, 0.0)
    self.assertGreater(obs.apparent_area, 0.0)

def test_detector_rejects_unknown_target_class(self):
    self.assertIsNone(MockRedBallDetector().detect(make_red_frame(), "bottle"))

def test_descriptor_is_fixed_size_and_normalized(self):
    descriptor = VisualDescriptor().encode(np.ones((96, 128), np.float32))
    self.assertEqual(descriptor.shape, (64,))
    self.assertAlmostEqual(float(np.linalg.norm(descriptor)), 1.0)
```

- [ ] **Step 2: Run and confirm imports fail**

Run: `python -m unittest tests.test_perception -v`
Expected: missing `fly_brain_rc.perception` modules.

- [ ] **Step 3: Implement mock detection, continuous flow, looming, and descriptors**

Move red segmentation behind `MockRedBallDetector`. Return bounding box, bearing, area, confidence, and timestamp. Convert the existing Farneback calculations into normalized left/center/right flow and continuous looming values without alarm booleans. Build descriptors from an `8x8` grayscale image, subtract the mean, and L2-normalize with a zero-image fallback.

- [ ] **Step 4: Test temporal motion behavior**

Use synthetic translated and expanding squares. Assert rightward translation changes lateral flow and an expanding square increases looming above a static pair.

- [ ] **Step 5: Run perception tests**

Run: `python -m unittest tests.test_perception -v`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add fly_brain_rc/perception tests/test_perception.py
git commit -m "feat: add replaceable visual perception pipeline"
```

### Task 3: Neural dynamics, compass, and path integration

**Files:**
- Create: `fly_brain_rc/neural/__init__.py`
- Create: `fly_brain_rc/neural/dynamics.py`
- Create: `fly_brain_rc/neural/compass.py`
- Create: `tests/test_neural_dynamics.py`
- Create: `tests/test_compass.py`

**Interfaces:**
- Produces: `RateLayer.step(previous: np.ndarray, recurrent: np.ndarray, external: np.ndarray, dt: float) -> np.ndarray`.
- Produces: `CompassState.update(yaw_rate: float, optic_rotation: float, encoder_delta: float, dt: float) -> None`.
- Produces: `CompassState.activity`, `heading`, and `home_vector`.

- [ ] **Step 1: Write failing simultaneous-update tests**

```python
def test_rate_layer_uses_previous_state_for_every_neuron(self):
    previous = np.array([1.0, 0.0])
    recurrent = np.array([[0.0, 1.0], [1.0, 0.0]])
    result = layer.step(previous, recurrent, np.zeros(2), dt=0.1)
    self.assertGreater(result[1], 0.0)
    self.assertLess(result[0], 1.0)

def test_compass_wraps_activity_across_ring_boundary(self):
    compass.reset(heading=math.pi - 0.01)
    compass.update(yaw_rate=1.0, optic_rotation=0.0, encoder_delta=0.0, dt=0.1)
    self.assertLess(compass.heading, -3.0)
```

- [ ] **Step 2: Run and confirm failures**

Run: `python -m unittest tests.test_neural_dynamics tests.test_compass -v`
Expected: missing neural modules.

- [ ] **Step 3: Implement leaky rate dynamics**

Use a numerically stable sigmoid and explicit Euler integration. Validate shapes, positive time constants, finite values, and `dt > 0`. Clip activities to `[0, 1]` after computing the complete next vector.

- [ ] **Step 4: Implement compass ring and home vector**

Use 72 preferred directions. Encode heading as a von-Mises-like bump, integrate yaw from IMU plus bounded optic-flow correction, and integrate encoder displacement along the decoded heading. Never accept absolute yaw or pose.

- [ ] **Step 5: Run neural and compass tests**

Run: `python -m unittest tests.test_neural_dynamics tests.test_compass -v`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add fly_brain_rc/neural tests/test_neural_dynamics.py tests/test_compass.py
git commit -m "feat: add neural dynamics and compass memory"
```

### Task 4: Persistent familiarity and habituation

**Files:**
- Create: `fly_brain_rc/memory.py`
- Create: `tests/test_memory.py`

**Interfaces:**
- Produces: `FamiliarityMemory.observe(descriptor: np.ndarray, now: float) -> float`.
- Produces: `FamiliarityMemory.save(path: Path) -> None` and `load(path: Path) -> FamiliarityMemory`.
- Produces: `ShortTermMemory.update(target, threat, dt)` and `reset()`.

- [ ] **Step 1: Write failing memory tests**

```python
def test_repeated_view_becomes_familiar(self):
    memory = FamiliarityMemory(capacity=8)
    first = memory.observe(unit_vector(0), now=0.0)
    second = memory.observe(unit_vector(0), now=1.0)
    self.assertGreater(second, first)

def test_persistence_is_atomic_and_bounded(self):
    memory = FamiliarityMemory(capacity=2)
    for i in range(4):
        memory.observe(unit_vector(i), now=float(i))
    memory.save(self.path)
    loaded = FamiliarityMemory.load(self.path)
    self.assertLessEqual(len(loaded.entries), 2)
```

- [ ] **Step 2: Run and confirm missing memory module**

Run: `python -m unittest tests.test_memory -v`
Expected: import failure.

- [ ] **Step 3: Implement bounded familiarity and decay**

Use cosine similarity, a configurable match threshold, visit-strength increments, exponential habituation decay, and least-useful-entry eviction. Save versioned JSON plus numeric arrays to a temporary sibling file and replace the destination atomically.

- [ ] **Step 4: Implement reset boundaries**

Verify short-term target/threat state resets on a new process or episode while persisted familiarity loads only when explicitly enabled.

- [ ] **Step 5: Run memory tests and commit**

Run: `python -m unittest tests.test_memory -v`

```bash
git add fly_brain_rc/memory.py tests/test_memory.py
git commit -m "feat: add persistent visual familiarity"
```

### Task 5: Fixed-topology neural fly brain and stable arrival

**Files:**
- Create: `fly_brain_rc/neural/genome.py`
- Create: `fly_brain_rc/neural/brain.py`
- Create: `tests/test_neural_brain.py`
- Create: `tests/test_arrival.py`

**Interfaces:**
- Consumes: `SensorFrame`, `ObjectObservation | None`, `MotionObservation`, descriptor familiarity, and `dt`.
- Produces: `NeuralFlyBrain.step(...) -> tuple[ActuatorCommand, BrainTelemetry]`.
- Produces: `FlyGenome.default()`, `to_json()`, `from_json()`, bounded `vector()`, and `with_vector()`.
- Produces: `ArrivalTracker.update(target, command, speed, tof, dt) -> bool`.

- [ ] **Step 1: Write failing topology and safety tests**

```python
def test_brain_has_no_behavior_mode(self):
    brain = NeuralFlyBrain(FlyGenome.default())
    self.assertFalse(hasattr(brain, "mode"))

def test_target_on_right_excites_right_motor_population(self):
    command, telemetry = brain.step(inputs(target_bearing=0.7), dt=0.02)
    self.assertGreater(telemetry.motor_right, telemetry.motor_left)

def test_threat_suppresses_forward_and_excites_reverse(self):
    command, telemetry = brain.step(inputs(tof=0.05, looming=1.0), dt=0.02)
    self.assertLessEqual(command.traction, 0.0)
    self.assertGreater(telemetry.motor_reverse, telemetry.motor_forward)
```

- [ ] **Step 2: Run and confirm imports fail**

Run: `python -m unittest tests.test_neural_brain tests.test_arrival -v`
Expected: missing brain and genome modules.

- [ ] **Step 3: Implement versioned fixed-topology genome**

Use named parameters with explicit lower and upper bounds. Reject unknown versions, missing fields, non-finite values, and out-of-range vectors. Connection signs live in code and cannot change through the genome.

- [ ] **Step 4: Implement sensory, memory, exploration, threat, and motor populations**

Compute every population from the previous state. Use opponent motor readout `traction = forward - reverse` and `steering = right - left`. Exploration receives unfamiliarity and unvisited-heading excitation. Threat receives looming, ToF, and contact excitation. Target activity influences steering and proximity braking.

- [ ] **Step 5: Implement stable-arrival tracking**

Accumulate time only while the target is visible, centered within the configured bearing, apparent size indicates the arrival band, absolute traction and measured speed are below thresholds, and ToF is safe. Reset accumulated time on any violation. Return success at 3.0 seconds.

- [ ] **Step 6: Run brain and arrival tests**

Run: `python -m unittest tests.test_neural_brain tests.test_arrival -v`
Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add fly_brain_rc/neural fly_brain_rc/memory.py tests/test_neural_brain.py tests/test_arrival.py
git commit -m "feat: add fixed-topology neural fly brain"
```

### Task 6: New runtime, scenario suites, and removal of the old controller

**Files:**
- Create: `fly_brain_rc/runtime.py`
- Create: `fly_brain_rc/sim/suites.py`
- Create: `tests/test_runtime.py`
- Create: `tests/test_suites.py`
- Modify: `scripts/fly_mode.py`
- Modify: `scripts/evaluate.py`
- Delete: `fly_brain_rc/brain/controller.py`
- Delete: `fly_brain_rc/brain/reflex.py`
- Delete: `fly_brain_rc/brain/ring_attractor.py`
- Delete: `fly_brain_rc/brain/goal_vision.py`
- Delete: `fly_brain_rc/brain/looming.py`
- Delete: `fly_brain_rc/brain/optic_flow.py`
- Delete: obsolete tests for those modules after equivalent new coverage exists.

**Interfaces:**
- Produces: `FlyRuntime.tick() -> RuntimeTelemetry` and `run_episode(...) -> EpisodeResult`.
- Produces: deterministic `training_cases(seed, count)`, `type_a_cases()`, and `type_b_cases()` with non-overlapping identifiers.

- [ ] **Step 1: Write failing end-to-end runtime tests**

Test that one tick reads exactly one frame, runs perception once, steps the brain once, applies one command, and emits one telemetry record. Use fakes implementing the contracts so the test needs no graphics context.

- [ ] **Step 2: Write suite-isolation tests**

Assert train, type-A, and type-B identifiers are disjoint. Assert type A starts with the target outside initial camera view and type B includes a simple turn or partial occlusion without dead ends.

- [ ] **Step 3: Implement runtime and episode results**

Keep ground-truth success measurements in the episode evaluator, outside `NeuralFlyBrain.step`. Record acquisition time, arrival time, stable-stop time, collision, penetration, reverse distance, steering variation, and familiarity novelty.

- [ ] **Step 4: Switch viewer and evaluator to the new runtime**

Preserve automatic `mjpython` re-execution and viewer persistence. Add `--genome` and `--target red_ball`. Remove imports of `build_brain()` and every old controller module.

- [ ] **Step 5: Run the full suite before deleting old files**

Run: `python run.py test`
Expected: new runtime tests and retained simulation tests pass.

- [ ] **Step 6: Delete old controller code and tests, then prove no imports remain**

Run: `rg "FlyBrainController|BristleReflex|RingAttractor|GoalDetector|LoomingDetector|OpticFlowBrain" . -g '*.py'`
Expected: no runtime references; historical documentation may still mention names.

- [ ] **Step 7: Run smoke evaluations**

Run: `python run.py evaluate --suite type-a --episodes 10`
Expected: structured episode output and no hidden-ground-truth contract violations. The hand genome may initially miss the final acceptance rate, which Task 8 addresses.

- [ ] **Step 8: Commit**

```bash
git add fly_brain_rc scripts tests
git commit -m "feat: replace rule controller with neural runtime"
```

### Task 7: Telemetry bus and live simulation dashboard

**Files:**
- Create: `fly_brain_rc/telemetry.py`
- Create: `fly_brain_rc/dashboard/__init__.py`
- Create: `fly_brain_rc/dashboard/server.py`
- Create: `fly_brain_rc/dashboard/static/index.html`
- Create: `tests/test_telemetry.py`
- Create: `tests/test_dashboard.py`
- Modify: `fly_brain_rc/runtime.py`
- Modify: `run.py`

**Interfaces:**
- Produces: `TelemetryBus.publish(record)`, `subscribe(maxsize)`, and `unsubscribe()`.
- Produces HTTP: `GET /`, `GET /api/state`, `POST /api/mission/start`, `POST /api/mission/stop`.
- Produces Server-Sent Events stream: `GET /api/telemetry` with the latest `RuntimeTelemetry` JSON records and encoded preview frames.

- [ ] **Step 1: Write failing bounded-pub/sub tests**

```python
def test_slow_subscriber_receives_latest_record_without_blocking_runtime(self):
    bus = TelemetryBus()
    queue = bus.subscribe(maxsize=1)
    bus.publish(record(1))
    bus.publish(record(2))
    self.assertEqual(queue.get_nowait().sequence, 2)
```

- [ ] **Step 2: Write API tests**

Start the server on an ephemeral port with a fake mission controller. Assert `/api/state` returns JSON, mission start validates `target_class`, mission stop changes state, and no endpoint accepts traction or steering fields.

- [ ] **Step 3: Implement telemetry serialization and bounded delivery**

Drop stale records for slow subscribers instead of blocking simulation. Encode NumPy arrays as lists except camera frames, which use capped-rate JPEG previews.

- [ ] **Step 4: Implement the dashboard server and static UI**

Use `ThreadingHTTPServer` and an `event-stream` response. Render the camera overlay, compass ring on canvas, population bars, sensor values, stable-arrival timer, and mission result. Serve all assets locally.

- [ ] **Step 5: Add launcher commands**

Support `python run.py sim --dashboard` and `python run.py dashboard --attach HOST:PORT`. Keep simulation functional without the dashboard.

- [ ] **Step 6: Run dashboard and runtime tests**

Run: `python -m unittest tests.test_telemetry tests.test_dashboard tests.test_runtime -v`
Expected: all tests pass.

- [ ] **Step 7: Manually verify live MuJoCo telemetry**

Run: `python run.py sim --scenario corridor --dashboard`
Expected: viewer and browser dashboard update concurrently; closing the browser does not stop simulation; Stop ends the mission without issuing direct actuator values.

- [ ] **Step 8: Commit**

```bash
git add fly_brain_rc/telemetry.py fly_brain_rc/dashboard fly_brain_rc/runtime.py run.py tests/test_telemetry.py tests/test_dashboard.py
git commit -m "feat: add live neural telemetry dashboard"
```

### Task 8: Reproducible fixed-topology evolution

**Files:**
- Create: `fly_brain_rc/evolution.py`
- Create: `tests/test_evolution.py`
- Modify: `run.py`
- Modify: `scripts/evaluate.py`

**Interfaces:**
- Produces: `EvolutionConfig`, `GenomeScore`, `EvolutionState`, `evolve(config, evaluator) -> EvolutionState`.
- Produces CLI: `python run.py evolve --config PATH --out PATH` and `python run.py evaluate --genome PATH --suite type-a|type-b`.

- [ ] **Step 1: Write failing reproducibility and constraint tests**

```python
def test_same_seed_reproduces_generation(self):
    first = evolve(tiny_config(seed=7), deterministic_evaluator)
    second = evolve(tiny_config(seed=7), deterministic_evaluator)
    self.assertEqual(first.to_dict(), second.to_dict())

def test_collision_rank_is_worse_than_any_collision_free_genome(self):
    safe = GenomeScore(successes=0, collisions=0, acquisition=999.0, energy=999.0)
    crashed = GenomeScore(successes=10, collisions=1, acquisition=1.0, energy=1.0)
    self.assertGreater(safe.rank_key(), crashed.rank_key())
```

- [ ] **Step 2: Run and confirm missing evolution module**

Run: `python -m unittest tests.test_evolution -v`
Expected: import failure.

- [ ] **Step 3: Implement seeded mutation and lexicographic selection**

Use fixed-size populations, elitism, Gaussian mutation scaled per parameter bounds, bound clipping, and deterministic seeded sampling. Rank first by zero collisions, then stable successes, acquisition rate, completion time, energy, reverse distance, and steering variation.

- [ ] **Step 4: Implement checkpoints and resume**

Persist version, seed, generation, population vectors, scores, scenario identifiers, and RNG state with atomic replacement. Reject resume when topology or parameter schema versions differ.

- [ ] **Step 5: Add CLI and short integration evolution**

Run a two-generation, four-genome configuration over two tiny fake episodes in unit tests. Then run a small real type-A smoke evolution to verify subprocess and MuJoCo integration.

- [ ] **Step 6: Run evolution tests and commit**

Run: `python -m unittest tests.test_evolution -v`

```bash
git add fly_brain_rc/evolution.py tests/test_evolution.py run.py scripts/evaluate.py
git commit -m "feat: add reproducible neural parameter evolution"
```

### Task 9: Acceptance tuning, documentation, and release verification

**Files:**
- Modify: `fly_brain_rc/neural/genome.py`
- Create: `config/genomes/hand-default.json`
- Create: `config/evolution/default.json`
- Modify: `README.md`
- Modify: `Digital Fly Car.command`
- Modify: tests only when a discovered defect needs a regression test.

**Interfaces:**
- Produces stable user commands and checked-in versioned baseline configurations.

- [ ] **Step 1: Tune the hand genome on training cases only**

Run type-A training cases while recording telemetry. Change named genome parameters, never topology or held-out cases. Save the first configuration that performs stable target acquisition and stopping without collision.

- [ ] **Step 2: Run offline evolution on training suites**

Use a committed config and explicit seed. Save the best genome and generation report under `artifacts/evolution/`, ignoring large intermediate populations in Git.

- [ ] **Step 3: Run held-out acceptance exactly once for the candidate**

Run:

```bash
python run.py evaluate --genome artifacts/evolution/best.json --suite type-a
python run.py evaluate --genome artifacts/evolution/best.json --suite type-b
```

Expected: type A at least 90%, type B at least 70%, zero accepted collisions, zero penetrations. If a threshold fails, return to training maps and evolution; do not tune against held-out episode details.

- [ ] **Step 4: Update README and double-click launcher**

Document simulator, dashboard, evolution, evaluation, memory location, reset procedure, and the fact that red-ball CV is a mock for future object recognition. Make the double-click launcher start simulation with the dashboard.

- [ ] **Step 5: Run final automated verification**

Run:

```bash
python run.py test
git diff --check
python -m compileall -q fly_brain_rc scripts run.py
```

Expected: all tests pass, no whitespace errors, and compilation succeeds.

- [ ] **Step 6: Run final GUI verification**

Run: `python run.py sim --scenario type-a-01 --dashboard`
Expected: MuJoCo remains open after the episode, dashboard streams live neural state, the target is acquired, and the vehicle holds a stable stop.

- [ ] **Step 7: Request code review and address findings**

Review contract isolation, neural simultaneous updates, persistence corruption handling, dashboard control boundaries, deterministic evolution, and held-out-suite separation. Add a regression test for each accepted defect.

- [ ] **Step 8: Commit release state**

```bash
git add README.md 'Digital Fly Car.command' config fly_brain_rc scripts tests run.py
git commit -m "feat: deliver neural fly brain simulator"
```
