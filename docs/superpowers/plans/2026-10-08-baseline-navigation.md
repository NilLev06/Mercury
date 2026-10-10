# Baseline Navigation Implementation Plan

**Goal:** improve the selected residual32 model without compass or scripted inference control.

**Architecture:** isolated experiment files, deterministic maps, frozen core, balanced residual fitting with old-output protection.

**Tech Stack:** existing PyTorch, MuJoCo and pytest.

## Constraints
No hidden goal input, new memory or inference planner. Do not overwrite deployed weights without comparative acceptance.

- [x] Repeat seven-map baseline using validate(model).
- [x] Test deterministic split-disjoint navigation cases and actual footprint-safe starts before adding maps.
- [x] Implement navigation_cases(seed, split) returning scenario, pose, family triples.
- [x] Evaluate baseline on new maps and collect labelled student trajectories on separate training maps.
- [x] Fit only residual parameters; preserve complete recurrent context and old-output anchors.
- [x] Save candidates separately; select by seven-map gate then new-map success/contact ranking.
- [x] Evaluate selected candidate once on final maps; test and hash-check; write measured results.

Final series intentionally skipped: no candidate passed selection/regression gates. No promotion. Results: docs/2026-10-08-baseline-navigation-results.md.
