# Synthetic stabilization and anatomical phase progress

## Synthetic fix

The original paired-population loop had neutral amplitude gain. A 1% gain loss
erased its first harmonic. Added **synthetic radial feedback**, target amplitude
0.8 and gain 0.2, multiplying recurrent feedback. It cannot preserve a heading
when recurrent feedback is ablated. No scalar heading is stored or forcibly set.

Lag compensation follows the continuous loop characteristic equation:

`(1 + tauE*lambda)(1 + tauP*lambda) = gain*(1 + i*gate*tan(delta))`.

For `lambda=i*yaw`, real gain is `1−tauE*tauP*yaw²` and the complementary
turn gate is adjusted accordingly. Finite-step integration is not exact.

Regression tests failed against the old implementation for all three diagnosed
behaviors: long-turn drift, dark memory under gain loss, and radial recovery.
They pass after the fix; recurrence and turn-input lesion tests still pass.

Identical sixty-second stress cases, before → after:

| Condition | Before | After |
| --- | --- | --- |
| Feedback gain 0.99 at rest | Heading lost | Contrast 0.390; heading retained |
| Feedback gain 0.95 at rest | Heading lost | Contrast 0.343; heading retained |
| Feedback gain 1.05 at rest | 9.95° displacement | Negligible displacement; contrast 0.445 |
| Constant ±2 rad/s | 28.90° error | 2.74° error |
| Activity noise σ=0.01/10 ms, five seeds | 0.26–22.59° error | 0.68–23.27° error |
| Yaw offset 0.01 rad/s at rest | 34.37° drift | 34.37° drift |

The noise and bias failures are **not solved**. A radial controller stabilizes
amplitude, not absolute phase. A real visual reference and calibrated motion
input are necessary; no simulator heading is injected. This remains a synthetic
benchmark, not an anatomical fly compass and not deployed vehicle control.
Acceptance covers the stated 60-second conditions, not arbitrary long runs.

Fresh full report: `artifacts/imitation/reports/heading-circuit-20261007/stress-stabilized-final.json`.
The original stress report is retained for comparison.

## Anatomical step

New offline module: `fly_brain_rc/neural/heading_phases.py`.
Transcribed EPG PB↔EB angles from **Extended Data Figure 5c** (not Figure 6) in
[Mussells Pires et al., Nature 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC10881393/#Fig11).
The original figure was visually inspected; the source uses a distinct EPG map
from the Delta7 map. Only supported, single-column EPG morphology hypotheses
receive phases. No PEN, PEG or Delta7 phases are guessed. L9/R9 and the
unresolved EPG remain unassigned within this selected EPG scope.

Result: **46 EPG phase hypotheses**, **zero independently validated assignments**.
Output: `artifacts/imitation/reports/heading-circuit-20261007/phases.json`, retaining
the input mapping digest and explicit source conventions/uncertainty.

Connectivity sanity check excludes self-edges and uses only EB edges between
mapped EPG neurons: 237 rows, 3,762 synapses. Synapse-weighted mean cosine of
phase separation is **0.4515**, versus **−0.0194** mean and **0.1046** 95th
percentile under 1,000 phase permutations (seed 7). None of the permutations
reach the actual value; finite-sample tail fraction is 1/1001. This is an
exploratory structural consistency check, **not independent terminal-morphology
validation or a calibrated assignment confidence**.

## Next gate

Construct and test anatomically constrained dynamics while retaining unresolved
assignments, recording transmitter/sign and time-constant assumptions, and
checking missing input populations. Synthetic amplitude control must not be
relabeled as biological physiology. Genuine visual anchoring is required before
vehicle integration. The deployed car, web UI and best checkpoint are unchanged.
