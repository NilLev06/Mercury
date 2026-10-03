# Target/bypass memory implementation plan

Goal: preserve pursuit and bypass intent across temporary camera occlusion.
Architecture: sensor-only NavigationMemory separate from Q and RecoveryManoeuvre.
Preserve rewards and actual detections; work in requested Desktop checkout.

- [x] Write failing tests for recalled direction gyro/encoder propagation,
  expiry, side persistence and version6 table retention.
- [x] Add navigation_memory.py; integrate effective target only into Q state,
  override priority recovery->bypass->learned, and telemetry memory confidence.
- [x] Use shared mode transitions for interruption/training; backup/migrate v6
  without reset. Test actual visual reward stays actual, reset loses memory.
- [x] Full tests, read-only review and paired true-camera cylinder evaluation;
  preserve main policy hash, save diagnostic report, document and commit.

Review found no critical/important bugs. Addressed both follow-ups: decaying
confidence telemetry and a trainer test for bypass update suppression / actual
visual reward. Paired evaluation did not solve any cylinder map; see
docs/target-memory-evaluation.md. No reward or main-policy changes.
