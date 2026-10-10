# Independent paired navigation Implementation Plan

> **For agentic workers:** Use executing-plans to execute inline. User authorised the independent comparison; no further training or promotion.

**Goal:** Compare the previously selected live800 against the unchanged favourite on fresh paired maps and diagnose regressions.

**Architecture:** Reuse standard autonomous episode evaluation; fixed checkpoint hashes and generated layouts saved before inference. Aggregate metrics and paired changes are separate from selected failure diagnostics.

**Tech Stack:** Python, PyTorch, MuJoCo, existing mixed_cases generator and episode evaluator.

## Global Constraints

- Favourite main-23of32 SHA e7155c9a2104a61f2936989a7d0b5b8bd2902e2e09ed2f1925c7c018958b5702.
- Candidate favourite-recovery-distillation-20261009/live-3.pt SHA b419701511c6a0296f4dc49bd7df223f552c817233596e75ab4feef7fe7f6356.
- Fresh seeds1050001..1050016, split final, stage2:64 simple128 complex. Earlier reserved1040001..8 remains sealed; no training data or development results from fresh seeds.
- 2400 physics steps; identical initial poses/layouts; no teacher, no weight updates, recurrent state reset per episode.
- Old7 and earlier complex6 evaluated for both policies. No default checkpoint writes or automatic promotion.
- Predeclared practical comparison passes only if candidate old7/priorcomplex6 are clean, complex clean count strictly exceeds favourite, simple clean count does not decrease, and contacts do not increase within either tier. This is a NEW comparison protocol authorised by the user, not a relaxation of the earlier strict58/64 zero-contact training gate. Report both intent and limitation.
- Report arrival and clean arrival separately; paired gains/losses, stall totals, total contact counts. Exact paired clean-outcome binomial test is descriptive, not a promotion gate. Shared generated map families do not imply generalisation to arbitrary mazes or hardware.

### Task1: Paired evaluation

Files: scripts/evaluate_paired_navigation.py, tests/test_paired_navigation.py.
- [x] Tests reject mismatched/reordered case identities, distinguish contacted arrivals from clean success, reject simple regression even when complex improves.
- [x] Implement paired_summary(source,candidate), practical_gate(old,previous,summary), run on fixed maps via existing episode(). Write protocol/layouts and source snapshot before evaluation; results incrementally.
- [x] Execute comparison once, preserve per-episode last20-decision target visibility, absolute throttle and steering, and the final six range readings. No teacher. Review clarified that the evaluator records final ranges rather than minima over the last20 decisions; minima are not claimed or used in selection.

### Task2: Evidence and review

- [x] Review analytical helpers with independent reviewer; verify both model and deployed checkpoint hashes before and after.
- [x] Report all metrics, paired case regressions and descriptive exact paired test. Categorise failures entering goal radius without stable arrival, final target absent, contacts and stalls; these are overlapping observations, not proven causes.
- [x] Save docs/2026-10-10-independent-navigation-results.md and link selected candidate. No commits or worktree mutation needed for read-only evaluation in the existing dirty workspace.
