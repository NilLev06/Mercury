# Camera sectors and initial decision experiments

The new optional observation schema appends twelve left-to-right blue image pixel coverage channels. It retains the original nineteen observable inputs. It contains no map coordinates, hidden target, compass or motion decisions. The copied encoder starts with zero extra columns, preserving original outputs before learning. Neural recurrence topology is unchanged.

Legacy compatibility, real checkpoint upgrade equality and multi-component camera coverage were tested. Complete suite after schema changes: 344 passed. Source baseline and backup hashes still match `17c6eb06e6a02e0c2aa1e2e57cdab22ca9da751763535fd6de0a15c20274ee1b`; original latest hash remains `5ab54ad573e09c842e94e17dab9221e8d4d9a97b8bfaf4be8296f99c11d7e9ad`.

Training starts with 69 teacher episodes, all successful and with zero contacts: old seven with three nearby start poses and four mixed training seeds. Later rounds add mixed-map teacher and assisted student trajectories (assistance probability .8, .6, .4, .2). Those assisted successes are training data, never counted as autonomous success.

All following numbers are autonomous development results on repeatedly reused old seven and mixed 12 selection maps, not independent generalization evidence. Each mixed selection set contains six simple and six complex maps.

| Sector candidate, cumulative updates | Old success / 7 | Old contacts | Complex success / 6 | Mixed contacts |
|---|---:|---:|---:|---:|
| 0, 1000 | 6 | 2 | 0 | 471 |
| 1, 2000 | 6 | 0 | 0 | 568 |
| 2, 3000 | 6 | 4 | 1 | 208 |
| 3, 4000 | 6 | 3 | 1 | 15 |
| 4, 5000 | 6 | 3 | 2 | 221 |
| 5, 6000 | 4 | 65 | 2 | 27 |

These results do not establish an improvement from sectors. Several training and replay changes occurred simultaneously, and no matched v1/v2 ablation has been run.

## Controlled opening-loss A/B

Teacher trajectory replay revealed substantial initial steering errors, often predicted magnitude .2 versus teacher 1.0. Two continuations use the same source checkpoint (`sector/candidate-2`), replay (`replay-2`), two stage seeds [51000,51001], lr1e-4, anchor4 and 2000 updates. Only weighting of the first ten decisions differs: 20 versus 1. Both old source anchors and teacher sequences receive the temporal weighting, normalized within each episode. Padding is excluded and tested.

At 2000 updates:

| Variant | Old success / 7 | Old contacts | Mixed success / 12 | Mixed contacts | Complex success / 6 |
|---|---:|---:|---:|---:|---:|
| Opening20 | 6 | 0 | 4 | 28 | 0 |
| Opening1 control | 6 | 15 | 4 | 216 | 1 |

Fewer contacts do not constitute navigation success: a policy can stop or drive elsewhere without touching obstacles. Opening weighting alone did not solve the task.

Review found a concrete supervised objective conflict at exactly identical initial observations: turn_corridor source anchor steering .203 versus teacher .859; single_cylinder -.144 versus -.500. With zero initial state, one policy cannot match both exactly. This confirms conflicting targets, not that the conflict alone caused poor navigation. No exact opposite-sign duplicate was found among teacher-only initial observations; unobservable teacher ambiguity is not proven.

Next experiment substantially reduces source-command distillation weight while retaining successful teacher rehearsals on old maps. Skill preservation is judged by autonomous old-map performance, not exact output copying. All experiments remain separate. No deployment or new independent final tests yet.

## Further diagnostic runs

Fixed replay `replay-5`, anchor weight .01, opening10: old-map successes across four stages were 2,1,0,2; mixed successes 0,0,1,2. Reducing distillation alone failed. This was not a clean single-factor comparison against original sector training because source replay and temporal weighting also differ.

A new teacher-only run uses 69 successful examples, v2 initial source, anchor .01, opening10 and lr3e-4. A known-map trace after 2000 updates autonomously reached the goal in 1115 physics steps without contact, versus teacher625. Around3s student continued turning while teacher labels required returning toward the goal, leading to a long hidden-target detour. This single observed trajectory supports targeted correction collection; it does not establish transfer.

Current successor collects fresh teacher-assisted corrections with beta .9; beta is the per-decision probability of taking the teacher action, not autonomous performance. Source and backup remain unchanged.

## Restricted-scope and single-route results

Restricting learning to new sector encoder columns and residual motor head did not recover the gate. Last checkpoint: old3/7 with447 contacts, mixed3/12 with514 contacts. Tests verify bit-identical legacy parameters even with pre-existing Adam momentum; behavior can still change through new sensory columns and the residual head. Experimental continuation tools now inherit recorded fitting scope unless explicitly overridden. Generic trainers are not restricted by that experiment metadata; these candidates remain isolated and undeployed.

A single-route diagnostic uses exactly one teacher-successful complex training map, with no competing source-command targets. After500,1000,1500,2000 updates, all four autonomous checks succeeded with zero contacts. Teacher-path MSE ranged from0.000158 to0.0000675. This demonstrates ability to fit one route; it does not establish generalization or old-skill preservation.

Next curriculum begins with exact old-seven teacher routes plus two complex teacher routes and expands by two only after autonomous mastery of all current routes without contacts. Training route checks and repeated development checks remain distinct; no new independent final maps are used before the reliability gate.

Full tests after fitting-scope continuation fixes:347 passed.

## Gentle teacher, corrections and full-batch optimizer

The gentle teacher (cruise .45, lateral tracking margin .12) completed all45
collected demonstrations without contact. This is teacher performance only.
Teacher-only mastery still failed: final old5/7,70 contacts; known training2/2
zero contacts; development4/12,393 contacts.

Fresh teacher-assisted corrections (per-decision teacher probability .5) helped
master more known routes. At round3: old7/7 and known2/2, all zero contacts.
At round7: old7/7 and known4/4, all zero contacts. Development remained4/12
with566 contacts. Expanding to six known routes regressed; round9 old5/7,
6contacts, known5/6 with4contacts, development5/12 with337contacts. Four known
routes mastered is progress in fitting; it does not establish transfer.

An independent full-batch LBFGS diagnostic ended at old6/7 zero contacts,
known2/2 zero contacts, development6/12 with492contacts. Actual LBFGS optimizer
state is separate; the normal brain checkpoint explicitly contains fresh Adam
for loader compatibility. This diagnostic also failed the reliability gate.

All trials remain isolated. Independent final maps were not reopened. The live
pre-compass model was not replaced.
