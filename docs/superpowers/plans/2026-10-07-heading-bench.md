# Heading Circuit Bench Implementation Plan

> **For agentic workers:** Use test-driven implementation in this session. User approved the staged direction and autonomous work.

**Goal:** Test neural-population heading memory separately from the deployed car.

**Architecture:** Keep FlyWire anatomy separate from a synthetic population-motif baseline. A compass population and two shifted feedback populations evolve by rate equations. Decode phase only for measurement; never update a stored heading angle. Visual input is an idealised cue profile supplied by a test, not camera vision yet.

**Tech Stack:** Python, NumPy, unittest, existing CSV/JSON anatomy.

## Global Constraints

- No changes to the deployed controller, checkpoint, web launcher or bicycle.
- Synthetic matrices and calibrated rates must not be labelled anatomical reconstruction.
- No root-ID-derived sector assignments. Annotated soma/representative positions are not EB/PB terminal sectors.
- No navigation training, reward changes or hidden goal input.

### Task 1: Isolated rate-population bench

Create `fly_brain_rc/neural/heading_ring.py`; test `tests/test_heading_ring.py`.
Interface: `HeadingRing.step(yaw_rate, dt, cue_phase=None)`; `heading` returns phase or None for unlocalised activity; `contrast` quantifies representation. Constructor switches `recurrence` and `turn_input` support causal controls.

- [x] Write and run failing tests for 10-second dark retention, opposite turns, visual reanchoring, loss of contrast without feedback and loss of integration without turn input.
- [x] Implement independent E/P-left/P-right activity arrays, 16 computational sectors, rectified rate dynamics with 2ms maximum integration step. Effective shifted cosine feedback represents aggregate excitation/inhibition, not individual anatomical synapses. No phase integration in step.
- [x] Reject nonfinite/range-invalid inputs before mutation and test unchanged state.
- [x] Execute the functional tests and separate parameter sweep; report errors, contrast, ablations and limitations in JSON.

### Task 2: Anatomical audit and handoff

- [x] Join the 149 selected IDs to official pinned FlyWire v2.1.0 annotations, compare type coverage, side and available metadata; record SHA256 and source URL.
- [x] Document sector-data gap, effective/synthetic model boundaries and next integration gate.
- [x] Verify scoped/full tests (227 pass) and independently review. Scope commit excludes unrelated artifacts; deployed weights equal previous deployable checkpoint.
