# RL contact recovery

User approved implementation of failed-manoeuvre memory, directional contacts,
safe fallback and short recovery curriculum. Work in the actual Desktop checkout.
Avoid another approval round per user's explicit preference.

Keep bicycle geometry, camera detectors and reward unchanged. Add front/rear
contact flags supplied by simulated bumper geometry (not extra range sensors).
Policy v5 includes contact direction, previous command, failure direction and
the specific blocked action (a turn may sweep the rear into an obstacle).
Use one controller in training and frozen inference. A contact-blocked command
held for 0.4 s without encoder movement inhibits that travel direction until
actual progress. After impact keep the escape memory for at least 0.8 s so
brief contact release does not cause immediate re-impact. Never inhibit both
travel directions based on that one failure; if bumper and failure conflict,
try another steering action in the bumper-permitted direction.
Unknown states and equal-Q ties use a deterministic conservative fallback;
training epsilon still explores admissible commands. Frozen inference is not
entirely learned: it includes this explicit safety/recovery layer.

Preserve v4 bytes as a backup; v5 training starts a new table, subsequent v5
launches resume it. Legacy inference remains usable through an in-memory v4
adapter with the same safety layer; do not rewrite the trained policy for tests.
Add first-quarter recovery cases alternating cylinder and wall, followed by
open, single-occluder and clutter cases. Recovery cases start near contact and
reward escape using the existing exploration/idle/contact rewards, then pursuit.

Verify failing tests first, then unit/integration tests and fixed-map camera
evaluation of the existing v4 policy through the safety adapter. Report safety
recovery separately from new learning; no claim that a new v5 policy is trained.
