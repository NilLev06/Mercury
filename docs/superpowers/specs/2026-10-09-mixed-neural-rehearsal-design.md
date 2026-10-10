# Preserve complex transfer and restore simple completion

Frozen source1efdc4… has old7/7 clean, independent complex27/32 clean, simple
10/16 clean versus original14/16. It is a strong isolated milestone, not
accepted/deployed. Most simple failures stop at .61–.64m. Keep architecture and
31 observable inputs, with encoder, recurrent graph and base readout frozen.

New data, declared before collection:
- canonicalold7 and nearby starts: actual qualified-source neural commands;
- simple80: train seeds810001..810010, mixedstage0 simple8 perseed, actual
  original19-input baseline commands while recording actual student31 features;
- knowncomplex48: re-evaluate qualified source, keep actual clean successes only;
- newcomplex80: train seeds820001..820010, mixedstage2 complex8 perseed, gentle
  teacher cruise .45/margin .12, retain physically clean successful demos.

Cached h must be computed by the qualified student's frozen core, never the
baseline core. Retain full successful approaches/stops. Original and qualified
actors are teaching sources only; runtime remains one student network. Drop
unsuccessful rehearsal labels and record every dropped run. Serialize simulation
layouts separately from observations; hidden coordinates never enter inputs.

Train residual32 with the tested cached-state fitter, successful old-source
anchors and rotating pure-student pre-contact corrections on new train maps.
Preserve working complex behavior through clean self-labelled rehearsal.
Initial larger development pool: selection seeds930001..930004 stage2
(16simple+32complex), plus existingold7 and earlierdevelopment12. The original
baseline defines simple-success preservation on the new development pool.
Require old7 clean, earliercomplex6 clean, >=29/32 newcomplex with zero contact
and simple success>=baseline/contact<=baseline before a second frozen audit.
Historical final710001..710004 must never enter training/development. Fresh
final2 seeds950001..950004 are reserved, not inspected before a candidate freeze.

## Follow-up diagnostic, declared before running

The first three mixed-rehearsal rounds reduce offline baseline-label MSE but
lose old successes and drift steering on previously clean complex routes.
Hypothesis: joint motor fitting changes route choice while trying to repair
stopping. Test a limited neural calibration that trains only the 33 parameters
of the final throttle row; encoder, recurrence, hidden motor features and final
steering row remain bit-identical. This is still one network, with no runtime
switch, planner or conditional action rule.

Use original-baseline clean simple throttle targets and qualified-source clean
old/complex throttle targets. Do not use new A* routes in this diagnostic, since
they may represent a different route choice. Compare anchor weights 1, 4, 16,
3000 Adam updates each, fixed seed, then static parameter scales .5 and 1.
Screen old7 and previous complex6, evaluate larger development only after these
pass. The fresh final2 seeds remain sealed. This limited test can diagnose a
stopping fix; it cannot establish broad obstacle-avoidance improvement by loss.

## Neural-only teaching follow-up

Completed mixed rehearsal has a 30/32 clean-complex candidate with simple12/16;
its midpoint with round7 yields simple16/16 and complex21/32, zero contacts on
all48, old7 and priorcomplex6 clean. Throttle-only33-parameter calibration did
not pass. No candidate meets the declared full gate.

Hypothesis: on-policy neural labels avoid route conflicts introduced by the
training planner. Start from the safe midpoint. Use the original baseline as
simple-map neural teacher and round5 as complex-map neural teacher. Both are
training-only label sources; the student alone drives correction rollouts.
Cache successful real teacher-controlled simple and complex trajectories with
the midpoint's unchanged core. Reuse declared train810001..10/820001..10 and
knowncomplex48, never development/final maps. Keep canonical midpoint anchors.
Run four rounds of1000updates with rotating8simple8complex corrections. Use the
same gates and sealed final2. New source teacher controls collection demos only;
the deployed shape remains one residual32 connectome network.

## Matched slow core adaptation

Final three static blends still fail the previous complex6 gate. Frozen-core
neural-only rehearsal improves some skills but has not met the full gate.
Next hypothesis: the fixed representation limits simultaneous transfer of the
neural teachers. Compare frozen-core vs live-core on exactly the same206 neural
teacher demonstrations and matched episode draws, from safe midpoint.

Both arms use existing full-history sequence fitter, same Adam1e-5, anchor4,
opening1, seed970001+stage, three stages200updates (600total). Canonical and
nearby old-source21 trajectories are output anchors. No new maps or teacher
planner are introduced. Full-core arm may update existing anatomical weights,
encoder, leak, bias and motor readout; the graph mask remains exact. Frozen arm
updates only residual32 at the same learning rate and objective. Full histories
are recomputed rather than reusing cached recurrent states. Assess old7 and
previous12 each stage, expanded48 each stage for comparable diagnostics.
Gate and final2 remain unchanged. This A/B isolates trainable scope at a gentle
rate; failure would not establish a biological limit of flies.
