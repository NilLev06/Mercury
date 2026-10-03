# Target docking

Approved scope: user requested implementing the preceding target-aware approach
and stable-arrival design without more approval rounds. Keep the Desktop checkout.

Use a calibrated visual estimate for the simulator's known 0.22 m sphere and
75 degree vertical camera, not world coordinates in the controller. Project
the estimated sphere onto the three front rays; only consistent measured hits
receive a target label. A nearer occluder must remain an obstacle. This estimate
is simulator-specific, not a claim of general monocular depth or YOLO.

Bypass sensor schema 3 adds visual approach stage and target-ray labels, critical
range bins below 0.08 m, a slow forward action and stop. Preserve old Q rows and
append zero action estimates, warm-start new far states from old rows. Docking
states get independent initial values because old goal-reaching values did not
learn stopping. Equal-valued actions prefer stop in a valid parking zone, a
declared initialization heuristic rather than a claim of learned stopping.
Ordinary navigation remains unchanged. No scripted bypass.

Shared arrival predicate: confident visible target ahead, estimated rear-axle
distance 0.60–0.85 m, consistent center ray, all ray gaps at least 0.08 m,
no contact, near-zero speed and throttle for 1 s. Trainer and viewer use this
same sensor predicate. Ground-truth distance is used only in shaping reward;
reward progress saturates at 0.70 m, penalizes overshoot and dangerous closing,
and rewards valid parking hold and terminal success. Do not penalize mere
proximity or give positive reward for moving away from obstacles.

Retain before-run backup, run counters and frozen evaluation. No full training
on user's tables. Tests cover ray/occluder semantics, stop eligibility, migration,
shared arrival, reward directions. Short actual-camera smoke and full unit suite.
