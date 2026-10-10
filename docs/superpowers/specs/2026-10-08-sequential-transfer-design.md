# Sequential avoidance transfer

User authorized training one skill on diverse maps and testing on unseen same-task maps. Use the immutable pre-compass residual32 baseline. No compass, YOLO, planner at inference or new architecture.

Capture actual successful baseline rollouts on the original seven maps as output anchors; supplement stored old replay. Generate independently jittered obstacle positions, mirrored cases and small start perturbations. Teach progressively one, two and three obstacles with full-route demonstrations plus student-state corrections. Freeze neural core and base; train residual only. Cache all observations, states, labels and anchors.

Separate train, selection and final seeds. Selection includes original double layouts and new same-task layouts. Require original seven-map success/contact gate. Keep original checkpoints unchanged. A candidate must improve independent same-task successes without extra contacts before any promotion. Report failed teacher routes explicitly.
