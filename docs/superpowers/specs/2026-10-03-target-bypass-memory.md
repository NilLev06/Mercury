# Target and bypass memory

Approved: add short-lived egocentric target memory and committed bypass side;
keep bicycle, sensors, RL and rewards. No new approval round per user request.
Use encoder/gyro only for memory propagation; target range from apparent area
is a coarse visual estimate, not ground-truth depth. Expire after 12 s without
reacquisition. Actual detections remain separate from recalled direction:
reward and dashboard target visibility must not treat recall as camera sight.

Version7 keeps nine policy fields; recalled target direction has a distinct
label, with ordinary-direction Q values as a warm start. v6 keys retain their
values and training count. Backup source before training migration; do not
rewrite main memory while evaluating.

Bypass holds a side through outward arc and along-obstacle travel, with gyro
and encoder completion and bounded phase duration. Safety recovery has higher
priority; its turn follows committed side unless stuck/bumpers require change.
After safety recovery, resume bypass at the along stage, not another outward
turn. Shared controller in training and frozen inference; interrupt held
commands on any mode transition. Scripted bypass/recovery windows do not
update Q; real detections alone provide visual reward.

Verify angle propagation, translation, expiry, side commitment, recovery
priority, version6 warm migration and mission reset. Run true-camera fixed
single-cylinder maps with and without new memory, same source policy/seed.
Do not claim navigation solved solely from unit tests.
