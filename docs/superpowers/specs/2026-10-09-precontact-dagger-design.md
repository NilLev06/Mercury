# Pre-contact frozen-brain correction diagnostic

Broad assisted full-core correction failed; pure-student full-core correction
also has large replay error. First pure-student replay contains22113 frames,
4229 stationary away from contact/goal; failure trajectories add long recovery
labels, sometimes repeated forward/reverse switches. This does not establish
unavoidable aliasing, but motivates focusing on preventing first failure.

Freeze source encoder, recurrence and base readout. Preserve actual successful
old-seven source state/output anchors. Train the existing32-unit residual head
only, using the already-tested cached-state mixture fitter. Initial replay is
90 physically validated original/mirrored demonstrations. Each round collect
neural-only actions with teacher labels on canonical old7 and complex48. End
training collections at first contact,3 consecutive seconds stationary away
from goal, stable arrival or2400 steps. These cutoffs never control actions and
are absent from autonomous evaluation. Keep last three correction rounds plus
fixed demonstrations. Teacher is training-only.

Eight2000-update rounds, Adam3e-4, complex fraction .5, source anchor1; model
core/base tensors must remain exactly unchanged. Old7/development12 every round,
all48 known complex every second round. Old7 zero contacts and complex6/6 zero
contacts required before final independent evaluation. No live deployment.

Motor capacity probe on cached states:128 versus32 units gives only modest
fitting differences, not a demonstrated autonomous gain. Architecture stays
unchanged. This diagnostic changes scope and training-collection cutoffs; it is
not a single-factor comparison with prior broad correction experiments.

## Fixed-cache anchor A/B

First pre-contact candidate: old3/7 zero contacts, complex3/6 with99contacts.
Compare fixed replay0 and actual successful source anchors with weights1 and32,
same source, fresh Adam3e-4, seeds130000..130002 and three2000-update stages.
Only anchor weighting differs between these two fixed-cache arms. First
weight1 stage should reproduce the original pre-contact stage0 parameters;
later stages differ from active DAgger because replay is fixed. No new dataset
collection, architecture or runtime controller is introduced.
