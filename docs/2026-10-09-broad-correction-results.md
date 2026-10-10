# Broad correction experiment

Source: mirror-control stage4 (old7/7 zero contacts, complex development2/6,
412 complex contacts). Both runs use90 physically validated original/mirrored
teacher histories:42 old starts and48 complex routes. Six500-update full-core
rounds, Adam5e-5, opening1, teacher anchors1. Each round collects canonical
old7 plus all48 complex, keeping last two correction rounds. Difference is
per-decision teacher action probability .5 versus0. Teacher labels are recorded
in both, and autonomous evaluation always has no teacher.

The .5 branch completed with old6/7,11 contacts; known training9/48 with1280
contacts; development5/12 with65contacts. It failed. The zero branch is still
running at report creation; early rounds are also poor. Do not interpret fewer
contacts alone as better driving: one zero-contact round had zero known
training successes.

Initial pure-student replay has22113 frames,4229 stationary away from goal
and contact. Some recovery histories contain nine throttle-sign switches.
This supports examining collection dominated by long failures, but does not
prove observation aliasing or a causal explanation for all failures. The
original90 successful teacher histories contain450 stop frames; all have the
goal visible. Hidden-goal stopping is a student error, not a direct hidden-stop
label in those original demonstrations.

A separate offline capacity probe freezes the source brain and compares motor
heads32 and128 units on identical sampled episodes/frames. Paired geometries
stay together; complex seeds31002 and31003 are excluded from head fitting.
At3000 updates:32-unit train MSE .00260, held-out .01339;128-unit .00246 and
.01267. The difference is modest and does not establish an autonomous gain.
No architecture change follows from this probe. Source is the earlier
four-route candidate7, whose encoder was not trained on held-out seeds.

All candidates remain isolated; live baseline/backup unchanged. Standard
old7, known48 and repeated development12 are distinct evaluations. No new final
maps were used. Review verified physical poses and label-only teacher behavior.
Full project suite subsequently passed359 tests after adding pre-contact
collector tests.

The zero-assistance branch also completed: last old3/7 zero contacts, known
training1/48 with2612contacts, development4/12 with206contacts, mean replay
MSE .1117. It failed. The six-round learning budgets, source, original replay,
optimizer settings and stage seeds match; sampled trajectory distributions
necessarily differ because assistance is the intervention. Neither branch
produced a candidate for independent final evaluation or deployment.
