# Navigation curriculum and autonomous validation

User approved the changes and requested no further confirmation gates.

The existing 255-neuron anatomical recurrent controller and observation schema
stay unchanged. It is a trainable controller with a real FlyWire subgraph, not
a complete biologically faithful fly brain. Camera CV and motor adapters are
learned/artificial interfaces; the teacher is an algorithm used only in training.

## Collection

Keep the first 24 introductory episodes. Thereafter dedicate each six-episode
assistance cycle to one family, ensuring full teacher, mixed and independent
rollouts of that family. Families rotate: straight, cylinder, turn, slalom,
occluded goal, recovery. Before age 48 only the first three are available;
before 72 the first four; thereafter all six. Mirror and small seeded
translations produce distinct training layouts. Recovery starts just outside
the cylinder footprint, sometimes with the goal outside the camera; never
start with geometric overlap. Hidden pose/map remain teacher/evaluator-only.

## Validation and persistence

Use the same seven autonomous cases as the diagnostic: five canonical named
maps plus two evaluation-namespace generated maps at seed 77. This is a fixed
validation suite for model selection, not an independent final test benchmark.
Run once before training if no compatible validation selection exists, and
every 12 saved episodes thereafter. Cap at the requested step count, 2400 max.
Never collect labels or optimize during validation. Honour cancellation without
promoting an incomplete suite. Preserve last weights, optimizer and 128 replay
routes. Store the selected checkpoint in a separate `.best.pt`, rank success
count first, fewer contacts second, lower mean final distance third. Save score,
suite version, graph signature and baseline architecture with the selection.
Do not reuse a score from an incompatible graph/model/suite. Test jobs use best
when present, otherwise last; training always resumes last. Old Q files untouched.

## Verification

Verify full teacher success on canonical maps (done: 5/5, zero contacts).
Regress staged selection, non-overlapping recovery, independent validation,
checkpoint preservation, best promotion and rejected worsening results.
Check a resumed short run on a disposable model copy and all project tests.
No claim of navigation improvement until autonomous metrics improve.
