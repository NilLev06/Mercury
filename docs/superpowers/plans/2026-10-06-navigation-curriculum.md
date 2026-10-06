# Navigation Curriculum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Teach broader navigation situations and select models by autonomous validation.

**Architecture:** New curriculum module owns training families and evaluation cases.
Validation module owns read-only scoring and best promotion. Runner orchestrates
collection and validation; lab selects best only for test jobs.

**Tech Stack:** Python, PyTorch, MuJoCo, unittest.

## Global Constraints

Preserve observation schema, graph, criterion, accumulated replay and optimizer.
Do not train during validation, promote cancelled suites or touch old Q memories.

### Task 1: Curriculum

- [ ] Add tests for families at ages 24,48,72+, stable resumed selection and legal recovery footprints.
- [ ] Run failures, implement `training_case(seed,iteration)` returning `(Scenario,start_pose,family)` and `validation_cases()`.
- [ ] Integrate episode starts through a simulator-only placement helper; forbid overlapping starts.
- [ ] Teacher-rollout all families with GoalStop and require success without collision.

### Task 2: Validation and selection

- [ ] Add regression with real temporary checkpoint: a good score promotes best; worse score cannot overwrite best; incompatible metadata is rejected.
- [ ] Implement `validate(model,steps,cancel)` using `collect_episode(beta=None)` and fixed cases, renderer cleanup and no labels.
- [ ] Implement `promote_best(path,model,optimizer,data,iteration,result)` writing score metadata and best checkpoint separately.
- [ ] Runner performs initial validation and every 12 iterations, emitting `phase=validation` with nested result.
- [ ] Lab test jobs use existing `.best.pt`; training keeps original path. Test argv behaviour.
- [ ] Explain latest versus best in README, run full suite and a resumed disposable smoke check, commit scoped files.

## Execution notes

Canonical teacher check: 5/5 successful with zero contacts. Recovery starts
at (.72,-.22) failed teacher tracking and were replaced with verified
(.65,-.30) starts, with optional 90-degree heading and mirrored layouts.
The anatomical model's original 107-episode checkpoint remains unchanged.
On its disposable copy, initial autonomous validation reproduced 4/7;
training resumed to 108 and repeated selection without resetting replay.
The post-update result was also 4/7, with different successful maps: no
robust improvement is claimed from this single-update smoke check.
Contact-route caching reduced the blocked rollout collection to 0.76s
while retaining the same 480 labels and observed student behaviour.
Read-only review found no remaining important issues; CLI explicit output
paths intentionally remain exact, while lab tests automatically use best.
