# Surround ranging Implementation Plan

**Goal:** Expose wheel-edge, side and rear proximity to the bypass learner.
**Architecture:** MuJoCo sensors -> SensorFrame -> compact bypass-only state;
legacy Q fallback with independent new rows, telemetry visualization.
**Tech Stack:** Existing Python/MuJoCo and HTML/canvas.

- [x] Failing tests tests/test_surround_ranges.py: left/right cylinder gaps,
  side wall and rear wall detection, no self-hit, zero reading, bypass state
  changes with side/rear proximity, nine-field warm start independent updates.
- [x] car_model.py six named rangefinders (existing b_fl/fr/fc, new b_sl/sr/rc),
  env.py indexes and SensorFrame mapping, contracts.py three defaults.
- [x] rl.py bypass-only context + legacy fallback; sensor_schema metadata.
  bypass_learning.py log schema and retain backups; no reward changes.
- [x] dashboard/static/index.html six-ray canvas and distances, fly_mode.py
  viewer rangefinder flag; add telemetry test covering new fields.
- [x] Focused/full tests, short actual camera smoke, read-only code review,
  docs and scoped local commit. Preserve main and bypass artifact hashes.

Verification: 179 tests passed. Real-camera smoke on a temporary copy of the
existing bypass policy: train 1 x 40 ticks, evaluate 1 x 40 ticks. Run counter
continued from 1 to 2, schema became 2, three new ten-field contexts appeared;
the before-run backup matched the source policy hash. User artifact hashes
remained unchanged. Read-only review found no actionable defects. These checks
verify integration, not learned navigation performance. No full training run.
