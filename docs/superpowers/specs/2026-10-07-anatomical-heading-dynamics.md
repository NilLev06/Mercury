# Anatomically constrained rate-dynamics pilot

User approved continuing without production visual landmarks. Build an isolated
research model, not a deployed navigator or claimed physiological reconstruction.

Use all 149 selected FlyWire v783 neurons and every positive internal regional
edge row. Aggregate regional counts only for numerical dynamics; original anatomy
retains NT/region provenance. No generated edges, cosine recurrent matrices,
symmetrization, amplitude-restoring synthetic controller, or per-neuron gains.

Model: `tau[type] * dr/dt = -r + sigmoid(W r + bias[type] + ideal_cue)`.
`W[post,pre] = synapse_count[post,pre] / 1000 * gain[post_type,pre_type]`.
The scale 1000 is an explicit numerical convention, not synaptic physiology.
Signed type-pair gains, biases and taus are explicit hypotheses. Initial pilot
assumes excitatory EPG/PEN/PEG and inhibitory Delta7, not an inference from NT.
Both positive and negative Delta7 configurations will be tested as sensitivity
controls. All 25 type-pair gains can be specified; the first bounded grid explores
shared excitatory/inhibitory gains only. No per-edge or topology learning occurs.

Only the 46 supported EPG phase hypotheses receive an ideal phase cue. Unknown
phases remain in recurrence but are excluded from cue injection and decoding.
Decode phase from column-averaged centered rates to avoid unequal neuron counts
or uniform activity generating a false heading. No scalar angle is stored.

Tests: exact directed counts, regional aggregation, no phantom edges, sign
assumptions independent of NT, uniform-activity decoder, cue response, recurrence
lesion, input/provenance validation, finite bounded dynamics and timestep check.
Bench: seed every case from zero; cue one second, dark three seconds, 12 evenly
spaced directions offset from anatomical phase centers. Report cue error,
dark error/contrast, lesion and nonuniform activity. Check maximum error and
minimum contrast at every 10 ms dark step, not just at the endpoint. Reject
promotion if any required direction fails (<30° cue/dark error, contrast >0.1).
No yaw integration claim: GLNO/ring inputs absent.

Alternative approaches rejected for this step: extending the synthetic cosine
ring (cannot test actual topology), or immediate full-circuit replication
(requires verified additional populations and input mapping first).

Work is scoped to new neural modules/tests/reports/docs. Keep vehicle controller,
web UI, teacher, best checkpoint and ordinary training configuration unchanged.
