# Neural Fly-Brain Rebuild

## Purpose

Rebuild Digital Fly Car around an interpretable, rate-based neural controller inspired by insect central-complex circuits. The controller must search for a selected visual target, avoid obstacles, approach the target, and stop in front of it. It must control a bicycle/Ackermann vehicle through motor and steering-servo commands without access to target coordinates or a planned route.

The first simulator version uses a red sphere and a color detector as a mock object-recognition backend. A physical version will replace that backend with a trained detector such as YOLO. Both backends must produce the same observation type, so the fly brain remains unchanged.

## Scope

### Retained components

- MuJoCo scene construction and rendering
- Bicycle/Ackermann kinematics
- One traction motor, passive mechanical differential, and front steering servo
- Collision geometry and swept-motion collision checks
- Fixed and generated test arenas
- Forward RGB camera

### Replaced components

- The `search`, `approach`, and `avoid` state machine
- Timer-driven escape behavior
- Controller-specific target observations
- Ad hoc evaluation and launch scripts where they conflict with the new runtime

The old controller will remain recoverable from Git history but will not ship in the rebuilt runtime.

### Deferred work

- Physical SBC and microcontroller integration
- Hardware watchdog and emergency-stop circuitry
- YOLO training and real-object datasets
- SLAM, metric maps, and complex maze planning
- Topological place memory
- Voice control and manual remote driving

## System Boundaries

The runtime separates hardware, perception, cognition, actuation, telemetry, and evaluation.

```text
SensorSource
  -> PerceptionPipeline
  -> NeuralFlyBrain
  -> ActuatorCommand
  -> VehicleBackend

Every component
  -> TelemetryBus
  -> Dashboard and episode recorder
```

The simulator and future physical vehicle implement the same `SensorSource` and `VehicleBackend` contracts. The dashboard consumes telemetry and does not depend on MuJoCo.

## Runtime Data Contracts

### SensorFrame

- RGB image and monotonic timestamp
- IMU yaw rate and acceleration when available
- wheel or motor encoder delta
- one forward ToF distance
- contact flag in simulation

The simulator initially derives the virtual IMU, encoder, and ToF signals from MuJoCo. Noise, delay, dropout, and bias can be enabled for domain randomization.

### ObjectObservation

- target class identifier
- confidence
- normalized bounding box
- horizontal bearing in image coordinates
- relative apparent area
- timestamp

The mock backend emits `red_ball`. A future detector can emit classes such as `bottle` without changing the brain interface.

### ActuatorCommand

- normalized traction command in `[-1, 1]`
- normalized steering-servo command in `[-1, 1]`
- timestamp and sequence number

The bicycle backend maps traction to rear-axle speed and steering to the front steering angle.

### BrainTelemetry

- activities of each neural population
- compass-ring activity
- target-memory activity
- path-integration estimate
- familiarity and habituation values
- threat activity
- motor-channel activities
- final traction and steering commands

## Perception

### Target detector

The simulator uses HSV segmentation only inside `MockRedBallDetector`. No color-specific logic may appear in the neural brain. The detector converts the largest valid red region into `ObjectObservation`.

### Optic flow and looming

The perception pipeline estimates left, center, and right optic flow and image expansion between frames. Looming activity represents time-to-contact evidence. The brain receives continuous normalized signals instead of alarm booleans.

### Visual descriptor

The perception pipeline converts a downsampled grayscale view into a compact descriptor. Familiarity memory compares the current descriptor with stored descriptors. The initial implementation can use a deterministic low-dimensional descriptor; the contract allows a learned embedding later.

## Neural Fly Brain

Every population follows leaky rate dynamics:

```text
tau * da/dt = -a + sigmoid(Wa + Ux + bias)
```

Activities remain in `[0, 1]`. The implementation updates all populations from the previous neural state before committing the next state, so source-code order cannot change the network dynamics.

### Sensory populations

- target-left, target-center, and target-right activity
- target apparent-size activity
- left, center, and right flow activity
- looming activity
- ToF proximity activity
- simulated contact activity

### Compass ring

A circular population maintains one activity bump representing estimated heading. IMU rotation and optic-flow rotation shift the bump. The controller must not read MuJoCo's absolute yaw.

### Path integration

Encoder displacement combines with compass activity to maintain an approximate home vector. This state supports exploration and detects approximate returns to previously visited regions. It does not create a metric obstacle map.

### Target memory

Recurrent populations bind a detected target bearing to compass activity. The activity decays when the target leaves the camera and updates when the detector sees it again.

### Visual familiarity and habituation

The brain keeps a bounded set of visual descriptors with visit strength and decay metadata. Familiar views suppress exploration interest. New views increase it. Habituation decays over time so old places can become interesting again.

Short-term target and threat activity resets between process runs. The runtime persists a bounded long-term familiarity store on disk using a versioned format and atomic replacement.

### Threat circuit

Looming, ToF proximity, and contact excite threat populations. Threat inhibits forward activity and excites reverse and steering-away activity. ToF remains an emergency proximity signal and does not supply a navigation map.

### Exploration circuit

Exploration activity receives excitation from unfamiliar views and unvisited compass directions. Familiarity, recent visitation, and threat inhibit it. Slow neural dynamics break symmetric choices without a scripted scan timer.

### Motor populations

Four opponent populations produce `forward`, `reverse`, `left`, and `right` activity.

```text
traction = clamp(forward - reverse, -1, 1)
steering = clamp(right - left, -1, 1)
```

Target bearing, exploration, threat, familiarity, and target proximity connect to these populations through the fixed neural topology. The runtime contains no behavior-mode enum or state-machine branch.

### Arrival behavior

Target-center activity excites steering alignment. Increasing target size suppresses forward drive. Excessive proximity can excite reverse drive. A successful arrival requires the target to stay centered while vehicle speed and motor activity remain below thresholds for three continuous seconds.

## Parameter Optimization

The neural topology, connection signs, input channels, and motor outputs remain fixed. Offline evolution tunes bounded numerical parameters:

- connection magnitudes
- neural time constants
- sensory gains
- memory decay rates
- familiarity and habituation rates
- motor scaling and smoothing

The first implementation must provide a hand-configured genome that completes basic type-A maps before optimization begins. Evolution evaluates genomes across batches of randomized training arenas. Test maps and seeds never participate in selection.

Fitness uses constrained, lexicographic objectives rather than one unconstrained weighted sum:

1. reject episodes with collision or penetration;
2. maximize successful stable arrivals;
3. maximize useful exploration and target acquisition;
4. reduce completion time, energy use, reverse motion, and steering oscillation.

The optimizer saves its configuration, random seed, genome, generation statistics, and training-map identifiers. Evaluation must reproduce any saved run.

## Domain Randomization

Training varies:

- target position and initial bearing
- obstacle positions within scenario constraints
- lighting and surface appearance
- camera exposure, noise, delay, and dropped frames
- IMU, encoder, and ToF noise and bias
- motor gain, steering gain, latency, and traction

Held-out evaluation uses unseen seeds and separate map definitions.

## Scenarios and Acceptance

### Type A

The target has an unobstructed route but starts outside the camera field of view. These cases test scanning, attention, approach, and stable stopping.

Acceptance: at least 90% success on held-out type-A episodes.

### Type B

The target lies beyond a simple turn or partial occlusion. A reachable route exists without maze-like dead ends. These cases test exploration, heading memory, path integration, and familiarity.

Acceptance: at least 70% success on held-out type-B episodes.

Every accepted episode must have zero collision and zero penetration. The brain cannot consume goal coordinates, goal distance, absolute pose, map labels, or a planned route.

## Dashboard

The dashboard runs during MuJoCo simulation and later against the physical backend. It provides:

- target selection
- mission start and software stop
- live camera image with target overlay
- compass-ring visualization
- target-memory, familiarity, habituation, and threat activity
- four motor-population activities
- traction, steering, speed, encoder, IMU, and ToF telemetry
- episode time, acquisition state, stable-arrival timer, and result

The dashboard communicates through the telemetry bus and mission-control API. It cannot issue direct traction or steering commands. The implementation uses the standard-library HTTP server and a Server-Sent Events telemetry stream, without a frontend build toolchain or an extra web framework.

## Commands

The rebuilt launcher exposes:

```bash
python run.py sim
python run.py dashboard
python run.py evolve
python run.py evaluate
python run.py test
```

`sim` can optionally start the dashboard in the same runtime. `dashboard` can attach to an already running simulation. Every command reads one versioned configuration format.

## Testing

Unit tests cover neural updates, simultaneous state commits, compass wrapping, path integration, memory decay, familiarity persistence, detector isolation, motor output bounds, and stable-arrival detection.

Integration tests cover the complete sensor-to-actuator loop, dashboard telemetry, deterministic replay, collision rejection, type-A behavior, and type-B behavior. Property tests verify that forbidden ground-truth fields cannot enter the brain input contract.

Evaluation reports success rate, collision count, acquisition time, arrival time, stable-stop duration, reverse distance, steering oscillation, explored-view novelty, and penetration count. A failed acceptance threshold returns a nonzero process status.

## Delivery Order

1. Define contracts, configuration, and deterministic telemetry records.
2. Refactor the retained simulator behind the new sensor and vehicle interfaces.
3. Implement mock perception, optic flow, looming, and visual descriptors.
4. Implement neural populations, compass, path integration, memory, familiarity, threat, and motor readout.
5. Replace the old runtime and delete the old controller modules.
6. Add type-A and type-B scenarios and evaluation gates.
7. Add dashboard and live simulation streaming.
8. Add genome serialization, evolutionary optimization, replay, and held-out evaluation.
9. Tune the hand-configured genome, run evolution, and publish measured results.
