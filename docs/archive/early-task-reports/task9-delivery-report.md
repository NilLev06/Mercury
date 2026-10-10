# Task 9 delivery report

## Scope and design decisions

This delivery covers user commands, README, dashboard double-click launch, disk familiarity integration and detector configuration validation. Acceptance tuning and candidate metrics are owned by the tuning task; this report makes no claim about held-out navigation acceptance.

The launcher resolves its project directory from its own location and uses the existing flybrain environment at `/opt/miniconda3/envs/flybrain/bin/python`. Its command now includes `--dashboard`. `run.py` already relaunches through `mjpython` on macOS; both the macOS relaunch and direct viewer path now forward memory settings.

Plain CLI simulation defaults to ephemeral familiarity without filesystem reads or saves. The double-click launcher explicitly passes `--memory .local/familiarity.json` to preserve user-facing familiarity under the current project root. `--memory PATH` opts CLI runs into persistence, and `--reset-memory` requires an explicit memory path and moves the existing store to a unique timestamped adjacent backup. `.local/` is ignored. A corrupt or unsupported store fails before constructing the environment and names the reset recovery command, preserving the original file.

`scripts/memory_store.py` wraps the existing bounded, versioned `FamiliarityMemory` loader and atomic saver. It does not persist compass, target trace, threat state, neural populations, motor commands or weights. One familiarity object is shared across freshly constructed episode runtimes. Normal viewer shutdown and KeyboardInterrupt save the store; unexpected exceptions do not replace it. No runtime.py edits were required.

`MockRedBallDetector` now rejects zero, negative, nonfinite and boolean minimum area ratios before detection can divide by that ratio.

## Verification

- The initial new lifecycle test failed because its requested module did not exist. Lifecycle implementation subsequently passed its roundtrip, recoverable-reset and corrupt-file-preservation tests.
- Detector regression was separately verified red: temporarily removing the validation produced five `ValueError not raised` failures for zero, negative, NaN, infinity and bool. Restoring validation made those tests pass.
- Memory-focused suite: 16 tests passed. New CLI forwarding test also passed, bringing the focused delivery module to 4 tests.
- Full suite before the final forwarding test: 105 tests passed with local socket permissions enabled. Restricted execution had six dashboard socket-bind errors, all resolved by the permitted rerun.
- `git diff --check`, compileall for package/scripts/run.py, launcher shell syntax, and `run.py sim --help` passed.

## Remaining release verification

The parent release task should run final combined tests after tuning edits, measure its chosen candidate on held-out suites under the agreed rules, and perform GUI verification. This delivery does not claim that a GUI session was inspected. The README documents measured-rate thresholds as requirements rather than unverified performance achievements.

## Persistence review revision

Release review reconciled explicit opt-in CLI persistence with the user's persistent double-click experience. Added red-first regressions for disabled storage and reset without an explicit path, then implemented the `None` lifecycle without any Path, load, or save calls. Added a CLI default regression. All seven delivery tests pass; CLI forwarding only emits `--memory` when supplied. README and launcher now show the final contract.
