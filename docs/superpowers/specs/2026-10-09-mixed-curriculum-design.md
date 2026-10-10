# Mixed simple/complex curriculum

User authorized mixed simple and complex training with progressive complexity. Source is immutable pre-compass residual32 checkpoint. Preliminary unprotected single-map probe tests whether current frozen-core architecture can learn one training route; it is not deployable or a transfer result.

Train on an explicit mixture every stage: simple teacher routes from existing straight/cylinder/turn/slalom families plus sequential two/three-cylinder routes. Increase complex-map fraction from 1/3 to 1/2 to 2/3. Collect full teacher and student-state labels, including teacher successes/failures, and retain complete recurrent histories. Include fresh simple student rollouts throughout rather than only static old-output penalties.

Only residual parameters learn; no compass, YOLO, hidden coordinates, inference planner or new architecture. Source successful-rollout distillation supplements direct simple-task teaching with low weight. Sample by difficulty tier and episode so long stalled scenes do not dominate.

Selection uses fresh seeds for simple and complex maps plus original seven-map gate. Choose one candidate and parameter scale before unseen final evaluation. Preserve deployed and backup hashes; save cache and explicit provenance. Report both trained-route success and independent results; do not promote low-reliability candidates.
