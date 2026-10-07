# Anatomical heading dynamics implementation plan

**Goal:** Run rate dynamics constrained to real measured connections and measure
the stationary memory gate without importing the car.
**Architecture:** Fixed directed count matrix plus explicit type-pair gains;
ideal EPG cue and hypothesis-based population readout. Separate bounded grid bench.
**Tech stack:** Python 3.11, NumPy, unittest; no new application dependencies.

## Task 1: Network

Create `fly_brain_rc/neural/anatomical_heading.py` and
`tests/test_anatomical_heading.py`.

- [x] Write tests with hand-counted regional directed edges and four EPG phases.
- [x] Run `python -m unittest tests.test_anatomical_heading` and confirm missing API.
- [x] Implement `AnatomicalHeading(anatomy, phases, parameters, recurrence=True)`;
  `step(dt, cue_phase=None)`, `activity`, `heading`, `contrast`, `synapse_counts`.
- [x] Implement `pilot_parameters(excitation=4., delta7_gain=-4.)` with full
  type-pair gains, per-type bias -2, tau .05, scale1000, cue strength6.
- [x] Verify tests, preserve original graph and uncertainty without inferred phases.

## Task 2: Bounded diagnostic benchmark

Create `fly_brain_rc/neural/anatomical_heading_bench.py` and bench tests.

- [x] Test zero-gain memory failure and exact topology retention on a fixture.
- [x] Run failing tests, then implement `evaluate(anatomy, phases, parameters,
  cue_seconds=1., dark_seconds=3., directions=12)` and CLI grid over excitation
  2/4/8/12 and Delta7 gain -2/-4/-8/-12, plus positive Delta7 control.
- [x] Fresh model for each direction; no teacher, stored heading or sensory input
  during the dark interval. Save exclusive JSON output with source hashes,
  explicit parameters, all cases and no yaw-integration/physiology claim.
- [x] Run actual data, select diagnostic candidate by declared cue+maximum dark errors,
  not by an impression from one direction. Then test a fresh 60-second hold.
- [x] Run full tests and independent read-only review, document result even if
  no setting passes. Commit only task files; no vehicle integration.
