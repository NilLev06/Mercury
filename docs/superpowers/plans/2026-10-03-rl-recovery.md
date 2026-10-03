# RL recovery implementation plan

Goal: stop repeated blocked travel and make unknown-state decisions deliberate.
Architecture: shared RLController plus directional simulated bumpers; versioned policy.
Stack: Python, unittest, MuJoCo, existing camera detectors.

- [x] Tests: reproduce repeated reverse failure, deterministic unknown choices,
  distinguish front/rear contacts, ensure schema backup and recovery curriculum.
  Run `python -m unittest tests.test_rl tests.test_environment` and observe red.
- [x] Contracts/env: append front/rear bumper fields; classify longitudinal
  contact from collision geometry without changing footprint or collision API.
- [x] RL: v5 state appends bumper direction, last action label, failure direction;
  shared controller accumulates blocked encoder time; selection masks failed
  travel direction and uses deterministic fallback for unknown/tied Q values.
  Load v4 for inference only, addressing Q values by its original eight fields.
- [x] Training: use shared controller for observation and action boundaries;
  prepend cylinder/wall recovery starts, preserve backups and v5 resume.
- [x] Verify tests, run fixed maps using actual camera and unchanged saved v4,
  write separate diagnostic artifacts, document how to start v5 training.
