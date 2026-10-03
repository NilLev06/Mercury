# Sequential escape and compact Q state

Approved: replace retreat/re-impact with retreat, reorient, clear, resume.
Keep existing rewards, bicycle footprint and camera. No new distance sensors.
Controller uses encoder distance and integrated gyro, not world pose, for phase
completion. Front impacts retreat straight, then forward arc and clearance;
rear impacts use forward arc immediately. New impacts restart recovery, with
turn-side alternation only for blocked manoeuvres; repeated arcs keep their
turn side so they do not undo gained heading. Per-phase time limits prevent an
unending script if encoders do not move; instantaneous bumper safety remains.

Q state v6 is original eight sensory categories plus longitudinal contact.
Detailed previous action and failed-action memory are not Q-key dimensions.
Load v5 by averaging nonzero estimates for matching compact keys, preferring
nonzero ordinary-control estimates per action over forced-recovery rows, with
fallback to all rows if ordinary estimates are zero, preserving
run count; zero-only estimates stay zero. Backup exact input before training
migration. v4 inference also maps to compact keys. Do not rewrite main memory
during diagnostic evaluation. Recovery actions are safety overrides: training
does not assign their outcomes to normal learned commands.

Acceptance: tests for retreat->arc despite contact release, gyro completion,
rear escape, reset, Q migration and skipped recovery updates; camera checks of
front cylinder and wall plus straight and one-cylinder goal maps. Report
escape separately from successful navigation; do not promise all-arena success.
