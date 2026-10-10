# Task 9 checkpoint — work paused at user request

Interpreter: `/opt/miniconda3/envs/flybrain/bin/python`. Worktree:
`/Users/leonard/Desktop/DigitalFlyCar/.worktrees/neural-fly-brain`.
Delivery/persistence changes are committed separately at `287c86a`.
Tuning edits are saved as a runnable version-2 checkpoint. Task 9 acceptance
is **not complete**. The unverified threat-turn prototype is saved separately
in `.superpowers/drafts/threat-turn-v3.patch` and is not active.

## Completed measurements

All measurements below used 2400 maximum steps (48 simulated seconds), with
three continuous seconds of image-centered, safely spaced, near-zero motion.
Collision and penetration requirements were not relaxed.

- Hand candidate version 2: training seed 0, cases 000–007: **8/8 success**,
  zero contacts, zero penetrations. Training 000 acquired its initially unseen
  target at 6.98 s and held a stable stop at 16.96 s.
- Evolution: seed 42, two generations, population four, six training cases
  `train-0-000` through `train-0-005`. Best: **6/6 success**, acquisition 1.0,
  zero contacts/penetrations; summed completion 74.2 s, energy 44.7770792723,
  reverse distance 0.0179743334 m, steering variation 11.6351338167.
- Held-out candidate version 2, evaluated once per suite: **A 10/10**,
  zero contacts/penetrations, acquisition 7.88–9.16 s, stable stops
  19.26–22.30 s. **B 0/10**, 3860 collision/contact events, zero penetrations.
  Both results are retained; the B acceptance gate failed.
- Additional independent obstacle training batch `trainb-73-000` through
  `trainb-73-005`, before the new threat-turn prototype: **1/6 success**,
  2307 contacts, acquisition 1.0, summed completion 262.02 s.

## Commands executed

```sh
python run.py evaluate --suite train --episodes 8 --steps 2400
python run.py evolve --config config/evolution/default.json --out artifacts/evolution
python run.py evaluate --genome artifacts/evolution/best.json --suite type-a
python run.py evaluate --genome artifacts/evolution/best.json --suite type-b
python -m scripts.diagnose_training --index 0 --steps 1500
python -m scripts.diagnose_training --seed 73 --index 0 --occluded --steps 1000
python run.py test
git diff --check
python -m compileall -q fly_brain_rc scripts run.py
```

Commands used the absolute interpreter above. Rendering and socket tests ran
with required sandbox escalation. Last completed full test run: **111 passed**.
Whitespace and compilation checks passed at that earlier point. These checks
preceded the final unverified version-3 prototype and `trainb` additions.

## Implemented diagnosis and version-2 candidate

Familiar-view search produced traction 0.274 but steering only 0.0074 even at
maximum exploration gain. A bounded exploration-turn magnitude now drives
the existing steering edge. Training showed looming inhibition stalled
approach; pinhole IMU yaw compensation removes rotational flow before radial
expansion, and a bounded looming magnitude preserves full ToF/contact drive.
Target gain 12 centers the target before the bicycle stops. Neural populations,
simultaneous rate updates, connection signs, and sensor-only inputs remain.

The evaluator now measures conservative sphere-to-car-envelope clearance:
rear-axle distance minus collision radius 0.30 m minus sphere radius 0.22 m.
Clearance must exceed the existing safe-ToF threshold 0.20 m; the original
image-area band supplies proximity. This replaces the incorrect 0.50 m
rear-axle goal prerequisite, which rejected safe stand-off stops around 0.8 m.
Regressions cover unsafe clearance, familiar-view turning, and rotational flow.

The training generator originally contained no obstacles, so selection could
not learn obstacle avoidance. New explicit `trainb` IDs generate seeded,
partially occluded arenas with open routes and never resolve held-out IDs.

## Current unfinished state

- `artifacts/evolution/candidate-v2.json`, `best.json`, and `best-genome.json`
  retain the measured version-2 candidate. The ignored `checkpoint.json`
  retains the completed two-generation run and RNG state.
- The separately saved prototype adds bounded `threat_turn_gain=40` and
  increments parameter schema to **version 3**. It has no completed measurement
  or verification. The active loader was restored to version 2 so all measured
  candidate files remain usable. Do not present the checkpoint as a release.
- The final training experiment used `proximity_distance=.8` with the new
  threat-turn magnitude on six `trainb-73` scenes. It was interrupted cleanly
  via Ctrl-C at the user's stop request; tool session **50534**, exit **130**.
  No final score was produced and no new held-out run was started.
- Dirty tuning files: `.gitignore`, `fly_brain_rc/neural/brain.py`,
  `fly_brain_rc/neural/genome.py`, `fly_brain_rc/perception/motion.py`,
  `fly_brain_rc/runtime.py`, `fly_brain_rc/sim/env.py`,
  `fly_brain_rc/sim/suites.py`, `fly_brain_rc/evolution.py`,
  `scripts/evaluate.py`, `tests/test_neural_brain.py`,
  `tests/test_perception.py`, `tests/test_runtime.py`.
  New files: `config/`, `artifacts/evolution/` (except ignored checkpoint),
  `scripts/diagnose_training.py`, and this report.

## Next authorized work when resumed

1. Resolve the version-3 prototype versus saved version-2 schema explicitly;
   preserve the measured candidate and its report.
2. Verify the new threat magnitude with circuit regressions, and measure only
   the independent obstacle training suite. Inspect telemetry before extending
   evolution; do not tune held-out case identities.
3. Commit a mixed A/occluded training configuration; run bounded seeded
   evolution. Save per-generation statistics and the selected schema-compatible
   genome, then evaluate each held-out suite once for that new candidate.
4. Run full tests, diff checks, compilation, and actual GUI/dashboard verification;
   publish truthful acceptance metrics and commit the release only when justified.
