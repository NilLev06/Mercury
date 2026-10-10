# Sequential transfer plan

**Goal:** test transfer of learned sequential avoidance while retaining old skills.

**Architecture:** protected residual-only fitting on complete state histories, actual successful baseline anchors and progressive diverse teacher/student replay.

- [x] Test and implement independently varied sequential_cases generator.
- [x] Record baseline anchors and baseline selection measurements.
- [x] Collect full demonstrations and student corrections on disjoint training cases; cache provenance and replay.
- [x] Fit and compare protected residual candidates; preserve core hashes and retain seven-map gate.
- [x] Select once; evaluate candidate/baseline on fresh final maps if eligible.
- [x] Verify tests/checkpoints/source hashes, review and report measured outcome.

Completed comparative experiment; selected candidate remains separate. Final result 2/24 versus 1/24; old7/7 zero contacts preserved.
