# Recurring Teacher Demonstrations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Keep collecting complete teacher routes after the introductory stage without resetting memory.

**Architecture:** Extract the assistance schedule into a pure training function. The runner supplies the saved global iteration; evaluation bypasses this function and continues without a teacher.

**Tech Stack:** Python, unittest, PyTorch, MuJoCo.

## Global Constraints

- Preserve weights, optimizer, demonstrations, global iteration and the existing 128-route replay limit.
- Keep the first 24 iterations unchanged.
- Thereafter repeat teacher probabilities 1, 1, 0.5, 0.5, 0, 0.
- Do not change rewards, sensors, geometry, arrival or interface.

### Task 1: Persistent assistance cycle

**Files:** Modify `fly_brain_rc/imitation/training.py`, `fly_brain_rc/imitation/runner.py`, `tests/test_imitation.py`, `README.md`.

**Interfaces:** `teacher_probability(iteration: int) -> float`; runner evaluates it only for training.

- [ ] Add a table-driven test with introductory boundaries (0,5,6,11,12,23) and two mature cycles starting at 24 and 42. Expected values are literal, not derived with the schedule.
- [ ] Add a runner regression using a temporary checkpoint at iteration 43. Collect six one-step real bicycle episodes with a tiny synthetic network and stub only camera pixels; assert report probabilities `[1,0.5,0.5,0,0,1]`, iteration 49 and retention of existing replay and seed metadata. Do not alter user artifacts.
- [ ] Run `python -m unittest tests.test_imitation.TrainingTests` and observe failures against the old runner.
- [ ] Implement `teacher_probability`: introductory branches `<6:1`, `<12:.5`, `<24:.2`; otherwise index `(1,1,.5,.5,0,0)[(iteration-24)%6]`. Replace the runner's inline schedule with this call, leaving evaluation beta `None`.
- [ ] Re-run imitation tests, including the existing evaluation test that forbids teacher creation.
- [ ] Document recurring help and clarify that assisted success is not autonomous success.
- [ ] Run full unittest discovery with local server/renderer permissions.
- [ ] Collect one full teacher-controlled cylinder route on a disposable copy of the user's checkpoint. Compare user checkpoint checksum before and after.
- [ ] Commit only the scoped changes and report verification; no push or merge.

## Execution results

Implemented and verified on 2026-10-06. The old schedule failed the resumed
runner regression (six zero probabilities instead of the recurring cycle).
The new schedule passed all 19 imitation tests and all 201 project tests.
On a disposable copy of the 43-episode checkpoint, the next full teacher
route reached the goal in 590 ticks with zero contacts and a 0.5-second
stable stop. The user's checkpoint and old Q-policy checksums did not change.
The current project branch is retained; no push or merge is requested.
