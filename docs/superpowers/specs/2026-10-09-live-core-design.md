# Live core learning

User requests sustained work toward reliable complex navigation. Existing residual-only attempts retain source but achieve low complex success. Diagnosis: selected policy has markedly higher teacher-command error on student observations than teacher routes; hidden-target observations are more frequent in complex student routes. These correlations do not establish a unique cause.

Experiment: enable learning of existing encoder and masked recurrent dynamics, plus existing motor readout, without changing graph connectivity or adding a compass/memory module. Preserve legacy residual-only default; persist explicit boolean trainable_core scope in checkpoints.

Train current recurrent dynamics on full observable sequences, not cached frozen states. Mix simple/complex teacher data and actual student corrections, with source old-policy output distillation on old successful trajectories. Cache labels, input sequences and provenance.

Select using fresh development maps and original seven-map gate. Independent final maps stay unopened until development complex success is >=90% with no contacts and old skills retained. Never overwrite source or backup. Record incomplete progress honestly; do not declare victory for low complex success.
