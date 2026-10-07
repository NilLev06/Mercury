# New heading stages: verification and stress tests

Run date: 2026-10-07. Tested code at `c781788`; no controller or checkpoint edits.
All **237 project tests pass** in the flybrain runtime with localhost access.

Reproducer: `artifacts/imitation/reports/heading-circuit-20261007/stress_check.py`.
Full report: `artifacts/imitation/reports/heading-circuit-20261007/stress-results.json`.
Run with `PYTHONPATH=.:.local/heading-deps` and the flybrain Python, passing a
new output JSON path. PyArrow 17.0.0 remains an optional offline dependency.

## Anatomical extraction and column hypotheses

- Re-extracted the original cell types and full connections CSV: exact report match.
- Re-loaded the checksum-verified NBLAST and metadata: exact mapping report match.
- Reversing candidate order changes no selected column or status.
- Sixteen score/margin combinations were checked: score 0.2/0.3/0.4/0.5,
  margin 0.025/0.05/0.10/0.15. Supported hypotheses range **95–124 of 149**;
  EPG coverage ranges **45–47 of 47**. Default remains 113 total, 46 EPG.

This proves reproducibility and limited threshold robustness, NOT anatomical
correctness. True column identities need independent terminal-morphology checks.
Most Delta7 assignments remain unresolved; no angular phase is assigned.

## Synthetic baseline failures

All cases first receive a one-second ideal visual cue at phase 0.37, followed
by 60 seconds in darkness with 10 ms outer steps and ≤2 ms inner steps.

| Stress | Outcome |
| --- | --- |
| No turn, no perturbation | Direction retained; contrast ≈0.4 |
| Constant ±2 rad/s turn | Final direction error ≈28.9° |
| Constant 0.5 rad/s turn | Error ≈0.60° |
| Feedback gain reduced 1% | Contrast falls to 0.0000743; heading becomes undecodable |
| Feedback gain reduced 5% | Contrast vanishes; heading becomes undecodable |
| Feedback gain increased 5% | 9.95° phase displacement at rest |
| Independent activity noise σ=0.01 each 10 ms, five seeds | Final errors 0.26–22.59° |
| Yaw-input offset 0.01 rad/s while actually stationary | 34.37° drift |

These are deliberately synthetic engineering perturbations, not calibrated
biological noise levels. Yaw-offset drift is an expected dead-reckoning
limitation, not a separate programming bug. Ideal symmetry without a cue
produces no spontaneous heading, which alone does not establish whether
symmetry-breaking disturbances could produce a bump.

The original short-turn bench still has maximum error 2.027°. Passing it does
not establish long-duration robustness. The failure under 1% lower feedback
matches the documented neutral first-harmonic gain: at exact unity the pattern
persists, below unity its amplitude decays. This synthetic motif is **not a
robust amplitude-restoring attractor**. No claim of a functional anatomical
compass or completed vehicle navigation stage is justified.

## Decision

Do not integrate this synthetic compass into the car. Next work should establish
anatomical PB↔EB phases and fit/test anatomically constrained dynamics with
amplitude stabilization. Repeat long dark holds, gain perturbations, noisy yaw,
neuronal disturbances, visual reanchoring and ablations before integration.
Do not hide drift by periodically injecting an exact simulator heading.

The best vehicle checkpoint SHA256 is identical before and after diagnostics.
No training was performed and the web interface remains unchanged.
