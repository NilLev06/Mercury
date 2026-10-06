# Connectome-grounded driving: first experiment

## Agreed direction

Replace tabular action learning with a recurrent, anatomically grounded fly network trained through demonstrations and DAgger. Keep the bicycle vehicle, camera, six range directions and existing web launcher. Preserve existing controllers and memories as a baseline; do not overwrite them or call their Q tables connectome training.

## Arrival contract

For the new experiment, arrival means body-centre distance to target centre at most 0.50 m and actual absolute speed at most 0.03 m/s for 0.50 seconds. Apply the same contract in demonstrations, training evaluation and live experiments. Merely crossing the radius while moving is not success. No sphere-centre entry is required. The teacher must brake before the boundary; the student has a continuous zero-speed action and must learn stopping. No evaluator-forced stop may count as learned success.

Exact simulator coordinates are allowed for teacher planning and scoring only. The student decides using camera-derived observations, range readings, ego motion and recurrent memory, not hidden target coordinates. A visible target does not disable obstacle readings. Goal geometry must remain distinguishable from other obstacles in teacher/scoring data; any student treatment of goal range returns must rely on observable evidence, never simulator object identity.

## Approaches and selection

1. Frozen anatomical network plus trained readout: cheaper, useful control, but limited adaptation.
2. Trainable anatomical recurrent network with demonstrations then DAgger: recommended first experiment, explicitly constrained to anatomical connectivity rather than invented wiring.
3. Local dopamine plasticity or RL from scratch: defer until imitation driving succeeds; neither fixes the current exploration and observation problems by itself.

## Experiment boundaries

Load a documented, versioned anatomical dataset with provenance and an explicit neuron/edge selection. No synthetic graph may be presented as a fly connectome. Before a large download or long training run, determine local memory/runtime feasibility. A smaller anatomically grounded subnetwork is acceptable but must be labelled as such, not a whole-brain simulation.

Build a bicycle-aware teacher respecting the swept wheel footprint and turning radius. Generate demonstrations for clear-target driving and stopping, then a single-cylinder detour. DAgger collects student trajectories and teacher corrections, including recovery situations. The teacher is absent during student evaluation. RL fine-tuning and long-horizon hidden-target search are outside this first experiment.

Use separate model artifacts with resumable optimizer state and data/version metadata. Old Q memories are not compatible with this network and are retained unchanged. Add one experimental mode to the current web interface rather than a new launcher or competing collection of trainers.

## Validation and reporting

Test arrival boundaries, speed/hold conditions, identical scoring across modes, observable-only student inputs, checkpoint resumption and footprint-aware teacher trajectories. Use fixed held-out maps/seeds and report success, contacts, stalled duration and goal-stop behavior. Compare with a small ordinary recurrent network trained on the same data to establish whether anatomical wiring adds value.

Begin with a bounded local smoke-training run before expensive training. A passing software test is not evidence of successful navigation. Report failed or infeasible training honestly; do not promote a checkpoint on reward alone.

## Risks

Sparse anatomical training may be too expensive locally; input/output mappings are modelling assumptions; camera target detection can fail near occlusion; imitation can fail away from teacher states. DAgger and held-out evaluation address distribution shift but do not guarantee autonomous search. This experiment is connectome-grounded control, not a biologically complete living fly.
