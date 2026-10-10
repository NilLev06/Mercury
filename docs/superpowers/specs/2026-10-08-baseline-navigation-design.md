# Baseline navigation design

User-approved basis: pre-compass connectome.best.pt, residual32. Preserve the backup and deployed weights.

First stage: deterministic multi-obstacle and initially out-of-frame target maps, comparative diagnostics and bounded motor-adapter fitting. Freeze encoder, recurrence, leak, bias and base readout. Retain observation schema, continuous neural outputs, bicycle mechanics and arrival criterion. No YOLO, compass, new memory or inference planner.

Use disjoint training, selection and final seeds. The privileged teacher labels student observations only during collection. Cache complete recurrent histories; balance starts, turns, contacts and pre-contact frames; protect old safe outputs. Record teacher errors and cancelled episodes explicitly.

Acceptance: seven original maps must retain 7/7 success and zero contacts. Compare independent new maps for success, contact count and elapsed steps. Select one candidate before final evaluation. If no candidate improves, retain current model and report failures. Preserve all original checkpoint hashes. Complex rooms and dead ends are later stages.
