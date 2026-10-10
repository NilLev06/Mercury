# Physical mirror augmentation diagnostic

Observation: candidate7 has much lower teacher-history error on four trained
complex routes than twenty other available training routes. Approximate feature
reflection also raises error, but v2 sector widths are unequal, so reflection
in feature space is not exact. Hypothesis: insufficient coverage of geometric
and left/right variation contributes to poor transfer. Memory absence is not
established.

Collect true reflected routes in MuJoCo, using reflected original teacher
commands for collection only. Reflect y, yaw, goal and obstacles; retain traction
and negate steering. Recompute camera/range inputs from the physical scene,
never reverse v2 sector arrays. Only successful zero-contact mirror realizations
are eligible. Keep every original and mark pairing/provenance. No new final
maps, no inference symmetry rule, no live promotion.

Controlled A/B: same candidate7 source, optimizer, seeds, fit scope, old teacher
anchors and update budget. Control uses all45 original gentle routes; treatment
adds physically collected mirrors. Both now contain all24 complex training
routes, so comparisons with earlier four-route mastery are not single-factor.
Judge old7 and repeated development12 autonomously; no teacher used at test.

## Broad student-state correction successor

A/B did not meet reliability gate. Train from control stage4 (old7/7 zero,
complex development2/6 with412 contacts), which is a candidate, not deployed.
Use the physically validated augmented replay (42 old including mirrors +48
complex). Add fresh teacher-assisted states on all48 complex maps and canonical
old7 each round; beta .5 is per-decision teacher probability. Keep last two
correction rounds plus fixed demonstrations. Evaluate all48 training maps and
old7/development12 autonomously, clearly separating training from transfer.
No final maps, compass, new model architecture or runtime teacher. Use six
500-update rounds, Adam5e-5, opening1, teacher anchor1. This experiment tests
whether corrections on broad geometry help where correction on four routes
and pure demonstration augmentation did not. It is not a single-factor A/B.
