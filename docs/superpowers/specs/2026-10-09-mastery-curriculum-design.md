# Mastery-gated curriculum

Single-route controlled diagnosis showed autonomous success without contacts after500updates and on three later checks; teacher-path MSE .000158 down to .0000675. This establishes ability to learn one known route, not transfer or preserved old skills.

Start with seven exact canonical teacher routes and two complex teacher routes. Remove competing old-policy command targets: anchors use identical teacher targets on old routes. Full observable histories, current recurrent parameters and unchanged anatomical mask remain mandatory. Fit500updates per round, evaluate exact old seven and current complex training maps autonomously, plus repeated mixed development maps. Add two training maps only after all current routes are autonomously successful with zero contacts. Do not advance on low offline MSE alone.

If precision required by the teacher routes remains a problem, separately collect demonstrations at cruise .45 and lateral footprint margin .12 instead of defaults .7 and .045. Default teacher behavior stays unchanged. A custom teacher factory is used only when collection beta is notNone; autonomous inference must never instantiate any teacher, including a supplied factory. Larger margin is lateral sampling of the existing footprint, not a guaranteed all-direction obstacle clearance.

Teacher collection success is a prerequisite for including a demonstration and is never presented as autonomous model performance. Save collection failures. Training maps and repeatedly inspected development maps are known; reserve independent final maps until >=90% complex development success without contacts plus preserved old seven. Never replace baseline or backup before measured acceptance.
