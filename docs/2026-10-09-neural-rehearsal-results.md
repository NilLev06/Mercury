# Mixed neural rehearsal: completed planner-label fit and neural-label follow-up

The frozen milestone `frozen-navigation-final-20261009/selected.pt` remains
isolated and undeployed. Its prior independent audit had 27/32 clean complex
and 10/16 clean simple successes; the original baseline had 0/32 and 14/16.
This is progress, not a passed reliability gate.

New collection finished: 229 attempted, 212 accepted. Acceptance requires stable
arrival and zero contact beginnings. Failed rollouts remain in collection.json
but contribute no rehearsal labels.

| Teaching source | Attempts | Accepted clean |
| --- | ---: | ---: |
| Qualified model, old canonical/nearby starts | 21 | 20 |
| Original 19-input model, new simple maps | 80 | 76 |
| Qualified model, known complex maps | 48 | 36 |
| Gentle training-only planner, new complex maps | 80 | 80 |

All seven canonical old routes are clean. The original actor really drove the
simple rollouts. The student's 31 inputs came from the same actual rendered
sensor frame; hidden-state caches were recomputed through the qualified
student's frozen core. Scene coordinates are separate environment metadata,
never policy inputs. Tests and read-only review verified both actor outputs and
cached states against a saved real episode.

Training seeds: simple 810001..810010, complex 820001..820010. Historical final
710001..710004 is excluded. Expanded development is declared as selection
930001..930004, stage2: 16 simple + 32 complex. A fresh frozen audit is reserved
as final 950001..950004 and has not been inspected.

The current fit freezes encoder, recurrent core and base readout. Only the
residual32 motor network trains, using clean rehearsal, original canonical
anchors, and rotating pure-student precontact corrections. Teacher-generated
correction labels do not control motors. Eight rounds of 2000 updates are
planned. Runtime evaluation retains the standard 2400-step arrival/contact
rules, not the training cutoffs. Training probes sample 12 complex routes and
must not be described as full-pool mastery.

Gate: old7 clean, previous development complex6 clean, expanded complex >=29/32
clean with zero contacts on all 32, expanded simple clean successes at least the
original baseline and contact count no greater. No deployment is authorized by
intermediate training loss or rehearsal success.

Verification before training: 364 tests passed. Input hashes and completed
collection evidence are recorded in the fit directory's input-provenance.json;
run.py/fit.snapshot.py preserve the running implementation.

Functional role check of the qualified model: zeroing trained recurrent weights
in memory yields 2/7 old clean and 0/6 development complex clean (1008 total
contact beginnings; 3 arrivals, one with contacts). Intact weights pass all 13
clean. This establishes dependence on the trained recurrent weights, not
biological fidelity or temporal memory alone. The saved model was unchanged.


## Completed mixed rehearsal

All eight rounds completed without changing frozen core parameters. Round5 has
old7 clean, expanded complex30/32 clean and zero contacts on all32, but
simple12/16 clean with290 contacts and earliercomplex5/6. Round7 has simple15/16
clean, zero simple contacts, but complex5/32 clean and307 contacts. No round
passes the full gate. Training loss reduction did not reliably predict driving.

Throttle-row-only calibration (anchor1/4/16, scales.5/1) does not pass. The row
preservation invariant was independently checked against a saved checkpoint;
steering is identical at equal state, while throttle changes future states.

Three static blends between source and round5 also fail. Three static blends
between round5 and round7 produce a useful safe midpoint at scale.5:
old7 clean, earliercomplex6 clean, expanded simple16/16 and complex21/32 clean,
**zero contacts on all48**. This is a separate saved single neural checkpoint,
not a runtime model switch. It still falls short of29/32 complex and is not
sent to final2 or deployed.

## Next controlled hypothesis

Planner correction targets may conflict with successful neural route choices;
this is a hypothesis, not a proven cause. Neural-only rehearsal starts from the
safe midpoint, with original baseline as simple teacher and round5 as complex
teacher. Both control demonstration collection; only the student controls
correction collection. Each label actor receives the student's same sensor
frame and maintains recurrent state on that actual history. A unit test checks
single rendering per decision and distinct teacher labels/student commands.
No hidden pose or goal enters either network. Four rounds of1000 updates are
planned on the same declared training pool; final950001..4 remains sealed.

## Neural-only results and continuation

Collection completed: 206/229 accepted clean: old-source21/21,
baseline-simple76/80, known-source-complex42/48 and new-source-complex67/80.
Initial round3: old7/priorcomplex6 clean, expanded simple15/16 clean with3
contacts and complex26/32 clean with78 contacts. Gate failed.

Four more rounds restored Adam (four residual parameter states, 4000 prior
steps), used round offset4 for fresh rotation and RNG seeds, and restarted
correction history. Canonical anchors were recollected from candidate3, so this
is a continuation of weights/Adam with updated regularization, not an exact
continuation of the previous optimization objective. Portable replay preserves
all x/y/h bitwise and records original replay SHA. Its core fingerprint includes
schema, graph signature and every non-readout tensor; it exactly matches the
continued source. Tests reject sensory/core changes while allowing readout
changes. Independent review verified these conditions.

Continued round5: simple15/16 and complex24/32 clean, zero contacts on all48,
but priorcomplex5/6. Round7: simple16/16 clean, complex27/32 clean with10
contacts, old6/7 with125 contacts. No continuation round passes. The two saved
milestones remain read-only, undeployed: `complex-30.pt` and `safe-balanced.pt`
under `navigation-milestones-20261009`, with an explicit score/hash manifest.

367 tests passed after core compatibility/resume changes. Final2 has not been
used. A finite three-scale static blend between original mixed round5 and
neural continued round7 is the next development-only diagnostic.

## Matched slow-core A/B and first full development pass

Both arms used the same206 actual neural-teacher demonstrations, matched
full-history episode draws, Adam1e-5, old-source21 output anchors weighted4,
opening weight1, 6 stages200updates (1200total). Continuation restored Adam and
kept the original anchors/objective. Frozen arm changes only motor residual;
live arm learns existing encoder/recurrent/leak/bias/readout parameters. Both
retain the exact anatomical mask; cached h is never used during live fitting.
Independent review checked checkpoints, masks and matched provenance.

At1200updates: frozen old7/priorcomplex6 clean, simple16/16 clean, complex28/32
clean with219 contacts; live old7/priorcomplex5 clean, simple15/16 clean,
complex26/32 clean with7 contacts. Neither passes alone.

Static full-parameter blend .5 between these paired1200-update models passes
the entire declared development gate: old7 clean, priorcomplex6 clean,
expanded simple16/16 clean, complex29/32 clean, **zero contacts on all48**.
The architecture and mask are unchanged. This is one neural checkpoint, not
runtime action mixing. Scales.25/.5/.75 were declared before evaluation; .25
fails, .5 is the first pass, .75 is not evaluated.

The first full-gate checkpoint was copied read-only into
`frozen-navigation-final2-20261009/selected.pt`, SHA256
`e7155c9a2104a61f2936989a7d0b5b8bd2902e2e09ed2f1925c7c018958b5702`.
Protocol and hash were saved before generating reserved final950001..950004.
Final2 requires old7 clean, >=29/32 complex with zero contacts on all32, simple
clean successes>=original baseline and contacts<=baseline. Unlike legacy raw
success counting, its explicit `simple_metric=clean_successes` rejects contact
arrivals as safe successes. A test preserves legacy protocol behaviour while
verifying this stricter metric. All369 tests passed. Final audit is in progress;
development success is not claimed as independent reliability or deployment.

## Final2 did not confirm the development gate

Independent final2 completed: old7 clean; candidate simple13/16 clean with41
contact beginnings versus original14/16clean with43. Complex24arrivals but only
23/32clean,576contact beginnings across5episodes; original3arrivals but only
1/32clean,3232contact beginnings. `accepted=false`, `deployed=false`.
Read-only review verified hashes, exact .5 interpolation, unchanged masks and
protocol saved before final generation. Original best, training checkpoint and
backup remain unchanged. Development29/32zero was optimistic and is not proof
of independent reliability. Final950001..4 now joins earlierfinal710001..4 as
permanently excluded from future training/selection.

A new broader experiment is declared separately in the broad-neural-transfer
spec:320new simple+320newcomplex, mostly neural teachers with clean expert
fallback only on failed training maps. Newdev96 seeds1030001..8, reservedfinal3
1040001..8. Coverage/fallback/training scope are bundled, so its result cannot
isolate causal contributions of each change. 372 tests passed before training;
read-only review found no blocking defect.
