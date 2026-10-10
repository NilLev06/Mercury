# Neural PPO experiment (in progress)

Runtime commands remain outputs of the anatomical recurrent model and its
neural residual motor adapter. No teacher, critic, coordinates, planner or
hand-coded action sequence is used during autonomous evaluation. The critic
and true goal distance are used only for training. Encoder, recurrence and
base readout are frozen and checked byte-for-byte after every update.

The selected source is gentle-corrected mastery round7, which scored old7/7
zero contacts and known training4/4 zero contacts. Its complex development
score is1/6, with559 contacts across all six complex episodes. This is not a
full fly brain: the recurrent graph has255 selected neurons and4088 edges.

PPO uses fresh rollouts only, a squashed diagonal Gaussian, stored pre-tanh
samples, joint old log probabilities, per-episode GAE and equal episode
weighting. First contact and stable arrival terminate training episodes; time
limits bootstrap the critic. Evaluation keeps the established full episode
protocol. Potential shaping uses decision discount .995, consistent with GAE.
Reported raw reward is undiscounted and must not be interpreted as policy
success. Stable arrival and contacts determine success.

Six unit/integration tests cover the transformed density, both clipping signs,
terminal/truncation GAE, reward durations, frozen parameters and checkpoint
roundtrip, and actual collector cutoff/contact behavior using a controlled
environment. Full suite passed354 tests before adding the collector test;
the six PPO tests subsequently passed. Independent review found no blockers.

Smoke source: full-replay round3. One update yielded old4/7 with4contacts,
complex2/6; mixed6/12 with405contacts. This failed the gate.

Protected run: source gentle-corrected round7, actor Adam1e-5, old successful
output anchor weight16, four old + four simple + eight complex episodes per
round,40 rounds maximum. Initial std [.06,.3], steering std decreases to .15.
Every fifth round evaluates old7 and repeated development12. No automatic
promotion and no fresh final evaluation before the development gate.

Algorithm reference: https://arxiv.org/abs/1707.06347

## Completed protected run

| Round | Old/7 | Old contacts | Complex/6 | Mixed contacts |
|---:|---:|---:|---:|---:|
| 5 | 7 | 0 | 1 | 526 |
| 10 | 7 | 0 | 1 | 566 |
| 15 | 7 | 0 | 2 | 350 |
| 20 | 7 | 0 | 1 | 481 |
| 25 | 7 | 0 | 1 | 525 |
| 30 | 7 | 0 | 2 | 357 |
| 35 | 7 | 0 | 1 | 308 |
| 40 | 7 | 0 | 1 | 520 |

All40 rounds finished. Best complex development2/6; last1/6. Old7/7 zero contacts throughout the recorded deterministic checks. The run failed the reliability gate; no independent final evaluation or deployment. Full suite355 passed, plus subsequent mirror-geometry test separately passed.
