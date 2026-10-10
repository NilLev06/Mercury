# Validated recovery demonstrations

Historical draft below. Superseded after the user selected the favourite as
the starting point by [the approved experiment plan](../plans/2026-10-09-favourite-validated-recovery.md):
main-23of32 source, shared collection once (32 per tier), four matched stages,
frozen residual versus slow live weights. Clean self rollouts are audited but
not added again; existing 709 clean demonstrations already provide rehearsal.

Broad709/8rounds did not pass (best51/64complexclean, gate58; old7 retained).
Training-only intervention audit reproduced12 actual noncontact stationary
student states exactly. Neural label actors recovered3/12; gentle privileged
expert recovered12/12 clean. This supports unreliable off-policy neural labels
in this failure subset, not a universal explanation of all transfer failures.

Keep all709 clean original demonstrations. Start student from broadround1
(old7/priorcomplex6 clean; newdevsimple29/32,complex51/64,12complexcontacts).
For rotatingtrainingmaps, first collect pure-student precontact rollout and
actualstudent actions. If it cleanly succeeds, retain actualselflabels. If it
fails, start a separate simulated recovery at max(0,T-30)decision frames:
replay exact recorded student commands before that point, then let a
training-only route follower drive to the real goal. Its preferred route is
reconstructed from that map's successful original demonstration; it may use
footprint-safe A* when deviated/exhausted. This preserves initial route choice
when feasible and avoids forcing global-planner labels on the entire trip.

Accept recovery only after actual stable arrival withzero contacts. Retain the
student's observed prefix as recurrent context; mask its output loss. Supervise
only actualexpert recovery observations/actions. Hiddenworldcoordinates are
teacher/environment metadata only, never policy inputs. Student inference
continues to use the same31observable features and masked255-neuron graph.
Fullsequenceforward recomputes allstates; no stalecachedh is used.

8rounds×200 full-history Adam1e-5 updates, anchor4, oldsource21 anchors, rotating
16simple16complex+old7, last3recoveryrounds. Reset optimizer for changed objective.
Keep newdev1030001..8 gate58/64zeroall, simple>=baselineclean/contact<=baseline,
old7/priorcomplex6 clean. Final3 1040001..8 remains sealed. Earlierfinals950001..4
and710001..4 remain excluded. Expert controls training recoveries only; all
selection/final/runtime motor commands remain exclusively neural.
