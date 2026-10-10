# Rejected heading reward experiment

Only the progress potential changed: P = rear-axle goal distance + 2*absolute
heading error (metres/radian); reward = 8*(P_before-P_after), with original
time, contact, steering and terminal terms unchanged. No controller changes.
Actual MuJoCo camera, seed 42, frozen copy of current schema-2 memory.
Train 12 full-length episodes on four fixed maps repeated three times;
evaluate frozen before/after on those same maps, epsilon=0, 2400 ticks each.
This is an in-sample sanity check, not a held-out benchmark or A/B reward trial.

| Map | Before min body-centre distance | After | Contacts before/after |
| --- | ---: | ---: | ---: |
| Straight | 1.538 m | 0.189 m | 0 / 0 |
| Single cylinder | 1.295 m | 1.655 m | 0 / 26 |
| Offset cylinder, seed 1000042 | 1.511 m | 0.338 m | 0 / 0 |
| Occluded cylinder, seed 1000045 | 0.460 m | 0.061 m | 0 / 3 |

Old 0.5 m rear-axle zone: before 1/4, after 3/4. New 5 cm body-centre
success: before 0/4, after 0/4. One exploratory training episode succeeded
but with five contacts; this did not reproduce in frozen evaluation.
Direct steering probe: away reward -0.316, toward +0.518 over 0.2 s.

Decision: reject this as a default fix. Better goal orientation is not enough:
the single-cylinder regression is unacceptable. Production code restored to
69aba12, user memory untouched. Do not infer which reward term caused all
behaviour differences without a matched old-reward training control; this
experiment proves neither broad improvement nor robust navigation.

Temporary experimental source, policy snapshots, tests and probe script:
`/private/tmp/fly-heading.PsHdCJ`. This directory is diagnostic, not a new
user-facing training mode, and may be removed by system temporary-file cleanup.
Next hypothesis to test: coarse state categories may merge situations that
require opposite actions. This experiment does not establish that cause;
inspect observation/action traces before adding another reward term.
