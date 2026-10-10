# Pre-contact frozen-brain correction (in progress)

Source control stage4 retains old7/7 zero contacts. Training uses actual old
successful state/output anchors,90 original/reflected demonstrations, and new
neural-only trajectories truncated at first contact or3 seconds stationary
away from goal. Teacher labels never control motors. Core, encoder and base
readout are fixed; only the existing residual32 learns. Cached states are valid
because the core is invariant, checked after every update.

Standard autonomous evaluation has full2400-step episodes and no training
cutoffs or teacher. Development never enters replay. Eight2000-update stages,
Adam3e-4, equal simple/complex mixture, source anchors1. Full known48 evaluation
every second stage, old7 and development12 every stage.

Independent code review found no blockers and verified state timing, gradient
scope and teacher separation. Two collector tests passed; full project359tests
passed. Teacher exceptions discard a whole collected prefix: error counts must
be inspected to identify excluded difficult trajectories.

This experiment changes training scope and collection boundaries together;
possible gains cannot be attributed to only one of those factors. No new final
maps or live promotion have occurred.

## Completed run and development qualification

Round3: old6/7 zero contacts, complex6/6 zero contacts, known training36/48.
Only old blocked_goal stopped early (minimum .585m). Round4 restoredold7/7
but lost complex safety. A static parameter blend of95% round3 +5% round4
passed old7/7 and complex6/6 without contacts; development overall11/12 zero
contacts. Blending changes one saved neural network, with no runtime switching.
Frozen selected checkpoint SHA256:
`1efdc4e9e47822dc1b1cde4b00dbf569e3d17d28f194dd8268d09f648430306f`.

The weight1 fixed-cache control exactly reproduced original precontact round0
model tensors, Adam state and RNG; independent review verified this. Weight32
preservedold7/7 but complex reachedonly2/6 at best. Full project360tests passed.

## Independent frozen audit

Protocol was saved before map generation/evaluation: final seeds710001..710004,
stage2,16simple+32complex, candidate frozen read-only. Candidate and original
were evaluated on identical maps. No training or selection used these maps.

| Model | Old clean/7 | New simple clean/16 | New complex clean/32 | Complex contact episodes |
|---|---:|---:|---:|---:|
| Original pre-compass | 7 (previously verified) | 14 | 0 | 24 |
| Frozen candidate | 7 | 10 | 27 | 2 |

Candidate simple arrivals11/16, one involving contact; complex arrivals27/32,
all27 clean. Original complex hadone arrival withcontact. Candidate contact
beginnings total177 versusoriginal2487; repeated contacts in a stuck episode
inflate this count, hence episode counts and clean successes are also reported.

This is a large transfer improvement, but FAILS predeclared reliability gate:
complex below29/32, contactepisodes remain, simple successes regress. It was
not deployed. Historical final maps710001..710004 must never enter training or
future development selection. Source and pre-compass backup remain unchanged.
