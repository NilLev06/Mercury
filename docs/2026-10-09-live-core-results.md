# Live anatomical core experiments

No candidate was deployed; original live best and backup remain untouched.
Every developmental evaluation is autonomous. Teacher sees map information only during collection; inference uses camera summary, range sensors, ego motion and contacts. Anatomical connectivity remains masked and unchanged. Existing neural weights can learn; no new recurrent connections were added.

Development suite: seven old known rehearsal maps, plus six simple and six complex maps from `mixed_cases(20001, 'selection', 1)`. This repeatedly used set is for model selection, not independent evidence of generalization. No new independent final test was used because the development reliability gate was not met.

| Experiment, last checkpoint | Old success / 7 | Old contacts | Complex success / 6 | Complex contacts |
|---|---:|---:|---:|---:|
| Mixed residual selected source | 7 | 0 | 1 | 82 |
| Full-history core, 1200 updates | 5 | 0 | 3 | 28 |
| Additional old rehearsal, 1200 updates | 5 | 9 | 3 | 11 |
| Teacher-only warm, 4000 updates | 7 | 0 | 1 | 288 |

Simple development performance of the teacher-only warm checkpoint was 5/6, zero contacts. Its recovery of old 7/7 does not establish complex navigation.

Saved successful source observation histories yield small action MSE for some failed candidates while their autonomous rollouts fail. This is consistent with accumulating closed-loop errors and insufficient nearby-state coverage; it does not establish causality. In the latest complex student replay, approximately 52% of labels ask for reverse versus below 1% in teacher demonstrations. Approximately 68% versus 36% of frames hide the target. Those distributions suggest a training tradeoff, not a demonstrated sole cause.

A synthetic history probe found decaying state retention; it was not a physical navigation test and does not prove recurrence is the failure source.

Next isolated experiment adds twelve camera blue-coverage sectors rather than discarding all but the largest detected obstacle. Zero extra encoder columns preserve the copied model before learning. Training includes varied old starts and assisted corrections. New histories are collected with actual sector pixels; old 19-dimensional histories cannot reconstruct those pixels.

Artifacts:
- `artifacts/imitation/experiments/live-core-20261009/`
- `artifacts/imitation/experiments/live-core-rehearsal-20261009/`
- `artifacts/imitation/experiments/live-core-teacher-warm-20261009/` (includes exact run script)

Full suite before sector-input changes: 342 tests passed. Subsequent sector verification recorded separately.

## Causal recurrence intervention

Original baseline: old seven 7/7, zero contacts. Setting recurrent weights to zero only in a separate process, keeping encoder, leak and readout fixed: 0/7 and 1190 contact beginnings. No checkpoint was overwritten. This demonstrates dependence of this trained policy on its recurrent weights; it does not prove biological fidelity, superiority over other architectures, or that temporal memory alone caused the effect. See `live-core-20261009/baseline-recurrence-ablation.json` and saved script.

## Scope of the brain model

Checkpoint graph contains 255 selected neuron ids and 4088 directed edges. Its recorded selection is weighted-degree candidates, BFS and the largest induced strongly connected component. It is a selected anatomical subgraph, not the full fly connectome or a validated biological neuron simulation. Runtime units are learned leaky tanh rate units with learned sensory/motor adapters. Recurrence ablation establishes functional dependence in this artificial model, not biological fly cognition. Navigation failures here do not establish an animal capability limit.
