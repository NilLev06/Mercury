# Sequential escape implementation plan

Goal: complete a meaningful clearance manoeuvre before learned pursuit resumes.
Architecture: compact v6 Q policy, sensor-only RecoveryManoeuvre shared by
training and inference. Work in requested Desktop checkout; preserve artifacts.

- [x] Add regression tests: encoder retreat must lead to forward arc, not
  immediate learned return; gyro arc completion; compact migration averages
  matching estimates and preserves training count. Observe failing tests.
- [x] Add recovery module (phase, distance, gyro, turn preference, safety).
  Update RLController to call it and expose override status for training.
- [x] Compact observation to nine fields; migrate v5/v4 on load; back up v5
  before training conversion. Run tests.test_rl and related integration tests.
- [x] Skip Q updates for recovery-started windows; retain update of initial
  learned action that caused collision. Bootstrap no value from active escape.
- [x] Record camera evaluation on simple wall/cylinder before complex maps;
  check source policy hash, full tests, code review, document limits and commit.
