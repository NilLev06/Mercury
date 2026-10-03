# Bypass lab Implementation Plan

> Execute inline in the user-requested Desktop checkout, using TDD and review.

**Goal:** Separate bypass experiment and persistent browser-controlled runs.
**Architecture:** task-tagged Q policy, sensor controller, dedicated trainer;
standard-library loopback process manager and web page, existing telemetry child.
**Tech Stack:** Python, MuJoCo, stdlib HTTP, HTML/CSS/JS.

## Constraints

Keep artifacts/rl/q-policy.json untouched; bypass-policy.json separate.
No scripted bypass in bypass-v1. Child jobs serial. No arbitrary subprocess args.

## Task 1: bypass experiment

- [x] Tests: task-tag roundtrip, no scripted override at a cylinder, seeded
  cases distinct/disjoint, progress reward/contact cost, cancellation checkpoint.
- [x] Add bypass_learning.py with bypass_cases(seed,count,held_out=False),
  train_bypass(episodes,steps,output,seed,epsilon,emit,progress,cancel,evaluate).
- [x] Integrate task tag in Q save/load, disable navigation commands only for
  bypass-v1; actual bumper change interrupts commands. CLI train-bypass/test-bypass.
- [x] Run unittest tests.test_bypass; retain old RL regression coverage.

## Task 2: persistent lab

- [x] Tests: reject unknown modes/fields, prevent concurrent start, return to
  idle/finished after child, stop and reuse, validate HTTP requests.
- [x] Add dashboard/lab.py with JobManager(root), start(request), stop(),
  state(); LabServer(manager,host,port). Ephemeral child port, lab port8765.
- [x] Add static/lab.html (mode cards, form, run state, logs, telemetry).
  Allow environment-selected dashboard port/no browser in child.
- [x] Change Finder launcher to run.py lab; reopen an existing matching lab.
- [x] Cooperative cancel via file flag, per-episode atomic checkpoints;
  preserve browser after job completion. Probe local UI, full tests, review.
- [x] Document launch and separate-model semantics, commit scoped files.

Verification: real-camera smoke 2x40 training ticks and 1x40 held-out ticks
using /private/tmp output. Temporary policy SHA unchanged during evaluation;
main q-policy SHA remains e885e5e8808a0c7099c20d2a958c63e73252dfd85dc57e2325bc014cec1ec1ee.
Browser launched reflex MuJoCo, embedded live camera, then stopped job while
lab persisted. Review issues fixed: pending-window cancel update, applied tick
counts, explicit bypass reach evaluator, whole-group stop completion before
restart. Full suite172 tests. No full training or navigation mastery claim.
