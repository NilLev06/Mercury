# Bypass learning and persistent web lab

User approved separate bypass learning plus a convenient web mode selector.
Proceed without extra approval rounds per their standing preference.

Bypass policy is separate artifacts/rl/bypass-policy.json, task bypass-v1.
One cylinder between start and ball, varied offsets, starts and heading.
Training seeds progress between resumed runs; frozen held-out cases use a
separate seed namespace. Train and evaluate the same sensor-only controller.
Keep short goal recall, disable scripted bypass/recovery for this task.
Directional bumpers restrict unsafe travel and interrupt held actions.
No world position in observations. Trainer alone uses goal-distance potential,
time/contact penalties and success reward. Report actual success/contacts;
no claim of mastery from code tests. Existing general policy/rewards untouched.
The bypass viewer uses the same reach-radius criterion and automatically ends
the simulator mission there; this is evaluator termination, not learned stopping.
Checkpoint after each episode and on cooperative cancellation; save atomically.

Persistent local web lab launches only one validated child job at a time.
Modes: bypass train, held-out bypass evaluation, general train, policy test,
reflex simulation. Fixed artifact paths; no arbitrary commands or paths.
Loopback only, reject cross-origin writes. Stop requests cooperative cancellation
with bounded process-group escalation. Lab persists after completion or error;
shows status/log tail and embeds existing camera telemetry, no new dependencies.
Finder command opens the lab or reuses its running instance. Viewer need not be
kept across jobs; the browser/launcher remains open. Never silently start120
episodes during implementation verification.
