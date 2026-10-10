# Broad neural transfer execution

1. Declare new train/dev/final ranges before any generation; preserve failed
   final2 and original baseline hashes. Never use final2 for training/selection.
2. Collect709trainingjobs:21old,320simple,48knowncomplex,320newcomplex.
   Neural-primary, expert fallback on failed/contact primary only; serialize
   both attempts and actual teacher provenance. Require completion and old7.
3. Verify tests for fallback, shared-frame neural labels, canonical old teacher,
   64complex gate, plus full regression suite. Read-only independent review.
4. Train8×200full-history masked-core updates atAdam1e-5, anchors4, rotation
   16simple16complex+old7, last3on-policyprecontactrounds. No stalecachedstates.
5. Assess old7/prior12 eachround, expandeddev96 everysecond. Select only if
   old7/priorcomplex6 clean, complex>=58/64zeroall, simple>=baselineclean and
   contacts<=baseline. If no pass, record failure and preserve milestones.
6. On first qualified candidate, freezehash/protocol then evaluate sealedfinal3
   1040001..1040008 against originalbaseline with clean-simple metric. No model
   replacement without verified final acceptance. Report limitations honestly.
