# Connectome imitation implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this tightly coupled plan task-by-task.

**Goal:** Deliver a separate, resumable imitation/DAgger experiment with genuine anatomical recurrent wiring and learned half-metre stopping.

**Architecture:** A footprint-aware bicycle teacher supplies action labels; the student sees only SensorFrame and camera detections. Anatomical and ordinary RNN students share collection, arrival scoring, persistence and the existing lab.

**Tech Stack:** Python 3.11, NumPy, PyTorch CPU, existing MuJoCo renderer and standard-library web launcher.

## Global constraints

- Do not overwrite Q memories or remove existing web modes.
- Arrival: body centre within 0.50 m, absolute speed <= 0.03 m/s, hold 0.50 s; no evaluator-forced success.
- Anatomical data must have provenance; test fixtures are not biological data.
- Student has no ground-truth goal/pose/map input. Teacher is absent in evaluation.
- CPU-sized subgraph, bounded smoke training, report failures honestly.

### Task 1: Data feasibility and arrival

Files: `fly_brain_rc/imitation/arrival.py`, `tests/test_imitation.py`.

- [x] Test `GoalStop.update(distance, speed, dt)` rejects moving passes and resets after leaving radius: `.49,.1,.5 -> False`, then `.49,0,.25 -> False`, `.49,0,.25 -> True`.
- [x] Run `python -m unittest tests.test_imitation -v`; observe missing feature failure.
- [x] Implement a validated hold timer; use evaluator body-centre distance helper.
- [x] Download/checksum published FlyWire v783 archive, inspect CSV header and establish local budget.

### Task 2: Bicycle teacher and observable features

Files: `imitation/teacher.py`, `imitation/observations.py`, same tests.

- [x] Test teacher rollout using real render-free FlyCarEnv on straight and single-cylinder maps; require stopped arrival with zero contacts. Test features include apparent goal size and retain every range independent of detection.
- [x] Observe failures, implement collision-checked bicycle lattice planning, path-follow steering and braking near target; recover by replanning from actual pose.
- [x] Implement `encode_observation(sensor, target, obstacle) -> np.ndarray` with fixed versioned fields; no environment argument.
- [x] Run teacher/feature tests.

### Task 3: Anatomical RNN and reproducible persistence

Files: `imitation/connectome.py`, `imitation/model.py`, same tests.

- [x] Test CSV pre/post IDs stay exact, induced graph only contains actual supplied edges, training changes allowed internal weights but cannot create absent edges. Fixture manifest explicitly labelled synthetic.
- [x] Run failures, implement deterministic strongly connected size-limited subgraph extraction and checksum/provenance manifest.
- [x] Implement trainable masked leaky tanh recurrent network and matched-size dense RNN control. Shared observable encoder and continuous traction/steering head.
- [x] Test checkpoint round trip restores optimizer, RNG and accumulated trajectories; reject wrong graph/schema.

### Task 4: Demonstrations and DAgger

Files: `imitation/training.py`, `imitation/runner.py`, same tests.

- [x] Test initial teacher collection and subsequent student rollouts store teacher labels with per-episode recurrent reset. Evaluation does not query teacher or mutate checkpoint.
- [x] Implement initial demonstration batch then teacher-mixing DAgger, truncated recurrent sequence training, stop/cancel checkpointing and fixed held-out maps. Store dataset and optimizer in a distinct trusted local checkpoint.
- [x] Report imitation loss separately from navigation success, contacts and stalled time.
- [x] Run a bounded real-camera smoke test and ordinary RNN comparison; do not promote failed models.

### Task 5: Existing lab integration

Files: `run.py`, `dashboard/lab.py`, `dashboard/static/lab.html`, `tests/test_lab.py`.

- [x] Test safe job construction for train/evaluate/new MuJoCo modes with distinct paths; old jobs unchanged.
- [x] Add CLI commands using existing macOS trampoline, live TelemetryBus and cooperative cancellation. Add experimental mode choices to same web launcher.
- [x] Run integration tests and CLI smoke command.

### Task 6: Verification and handoff

Files: `docs/2026-10-06-connectome-imitation.md`, optional dependency file and ignore rules.

- [x] Run fresh relevant and full automated suite; inspect diff and old memory checksum.
- [x] Record data license, actual graph size, hardware timings and smoke results, distinguishing teacher success from student success.
- [x] Commit only relevant code/tests/docs, leaving local datasets and old artifacts untouched.
