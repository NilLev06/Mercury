# Target docking Implementation Plan

**Goal:** Distinguish target approach from unsafe closing and learn stable parking.
**Architecture:** Shared sensor-only docking module consumed by bypass state,
reward and runtime. In-memory legacy Q warm start; files change only in training.
**Tech Stack:** Python, existing MuJoCo and unittest.

- [x] Tests: hand-calibrated sphere observations; nearer range occluder rejected;
  stop only in valid docking zone; stable arrival resets on motion/contact;
  old action vectors independently migrated; training/runtime shared predicate.
- [x] `fly_brain_rc/docking.py`: visual range/ray consistency, approach stage,
  one-second tracker. No absolute pose or goal input.
- [x] `rl.py`: explicit schema-3 preparation, old rows retained, slow/stop actions,
  extended state and safe stop eligibility; legacy bypass unchanged until upgrade.
- [x] `bypass_learning.py`, `runtime.py`: same arrival tracker; reward saturates
  progress at stand-off, penalizes overshoot and dangerous closing, parking hold.
- [x] Viewer upgrades bypass in memory, telemetry labels clear completion;
  docs explain migration, simulator-specific depth and retraining requirements.
- [x] Full tests, camera smoke on temporary memory, review and scoped commit.

Verification: 190 tests passed; dashboard JS syntax and diff checks passed.
Camera calibration at actual sphere distances .6/.7/.8 m yielded estimates
.606/.709/.817 m and valid center matches. On a temporary copy of user memory,
trainer and frozen evaluation parked for 1 s at .7 m in 50 ticks, and viewer
runtime reported stable_arrival under the same condition. Review sign and
moderate-occluder findings fixed with red-green regressions; re-review found no
remaining important issues. User policy hashes unchanged during this task.
No full navigation training or broad-map performance claim.
