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
