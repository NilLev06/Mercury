# Mixed curriculum plan

**Goal:** assess learning and transfer under a mixture of simple and complex scenes.

**Architecture:** frozen core, residual-only fitting, episode-balanced simple/complex teacher and student replay, low-weight old-output protection.

- [x] Check single hard-route learning without preservation constraints.
- [x] Add tested mixed_cases generator with stage ratios 8:4, 6:6, 4:8.
- [x] Collect full simple/complex demonstrations and student corrections at each stage.
- [x] Fit balanced replay, cache complete trajectories/states and evaluate train/selection skills.
- [x] Select one old-gate-compatible candidate before independent evaluation; keep baseline unchanged.
- [x] Verify tests, checkpoint lineage/core integrity and measured report.

Completed: selected parent stage 1, scale .25; independent total 11/24 vs8/24, complex3/16 vs0/16, simple8/8 both. Baseline preserved; no deployment.
