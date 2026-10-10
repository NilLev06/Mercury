# Broader neural teaching after failed final2

Frozen final2 e7155… passed development but independently scored simple13/16
clean vsbaseline14/16, complex23/32clean vsbaseline1/32clean. Old7 clean, gate
failed. Never train/select on final2 950001..950004 or earlierfinal710001..4.

Hypothesis: narrow training coverage and repeated small-pool selection inflated
transfer estimates. Broaden input maps, retain31 sensor features/255 neurons/
4088 directed anatomical edges, and keep runtime purely neural.

Declare before collection:
- student source: frozenfinal2 e7155… (old7 clean, architecture unchanged);
- simple320: seeds1020001..1020040, trainstage0 simple8 each, originalbaseline actor;
- complex320: seeds1010001..1010040, trainstage2 complex8 each, complex30 neural actor;
- knowncomplex48 replay actorcomplex30, plus actualstudent old21 starts;
- on a failed neural demonstration, collect a separate gentle expert rollout
  .45/.12; retain only stable arrivals withzero contacts. Record both attempts;
  expert labels are used for those training maps only. Other maps keep neural
  teacher labels. No expert controls the student during correction collection.
- keep every actual scene/pose and teacher provenance separate from policyx.

New development: seeds1030001..1030008, selectionstage2,32simple+64complex.
Require old7 and earliercomplex6 clean, >=58/64clean newcomplex andzero contacts
onall64, simpleclean>=originalbaseline/contact<=baseline. This new larger pool
is not final and may be used for selection. Reserved final3:1040001..1040008,
finalstage2, same96mapcount/criteria, inspect onlyafteronecandidatefreeze.

Training: full-history sequences, all existing masked neural parameters may
learn atAdam1e-5. Fixed successful demos plus last3on-policyprecontactcorrection
rounds, rotating16simple16complex pluscanonicalold7. Neural teachers receive
sameactualsensorframe and per-episode recurrent state. Preserve canonical old
source through its actualneuralanchors; recompute all hiddenstates fromfullx,
never use stalecachedh aftercoreupdates. Initial8rounds×200updates,anchor4,
opening1, fixed seeds980001+round. Development96 everysecondround, old7 and
previous12everyround. No deployment without the independentfinalgate.
