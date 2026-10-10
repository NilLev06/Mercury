# Navigation experiment, 2026-10-08

User-selected pre-compass residual32 model preserved. No experimental weights installed.

## Fresh measurements

Original seven-map suite: 7/7 successes, zero contact incidents. New selection suite: 0/12 successes and 1031 contact incidents. Contacts count repeated contact beginnings, not distinct obstacles.

| Trial | Candidate | Original successes | Original contacts | New successes | New contacts |
|---|---|---:|---:|---:|---:|
| trial-1 | candidate-0-selection | 5/7 | 241 | 2/12 | 228 |
| trial-1 | candidate-1-selection | 5/7 | 0 | 0/12 | 225 |
| trial-2 | candidate-0-selection | 6/7 | 2 | 0/4 | 58 |
| trial-2 | candidate-1-selection | 4/7 | 191 | 0/4 | 230 |

Trial 1 mixes sequential avoidance, staggered obstacles and initially invisible goals. Trial 2 trains sequential avoidance only, with full teacher routes, lower learning rate and stronger old-output protection. Its teacher completed 4/4 demonstrations without contacts. This establishes those four routes are traversable, not that every generated map is solved.

Blends of trial-1 candidate-0 at 0.25, 0.5 and 0.75 all achieved 6/7 original successes, with 3, 2 and 3 contacts respectively. None passed the regression gate.

No candidate qualified for independent final evaluation. Final seeds 6101/6102 remain unused. Selection scores are development measurements, not independent generalization results.

## Integrity and limits

Only residual head parameters trained; all encoder, recurrence, leak, bias and base weights verified unchanged. Camera schema, arrival criterion and vehicle mechanics unchanged. No inference planner, compass, YOLO or external memory added. Teacher map access is confined to training labels/demonstrations.

The four candidate checkpoints were repackaged with the standard Adam parameter-group layout after a compatibility review. Learned weights and populated optimizer states were preserved; standard loader roundtrips verified. Future runs use the compatible group layout directly.

This is a small failed improvement attempt, not evidence that the frozen brain cannot solve the task. Retraining reduced contacts in several cases but did not reliably combine new skills with old ones. Further blind retraining is not justified. Next investigation should capture observation/action sequences before a specific collision or premature stop, and determine whether the current 19-feature representation and frozen states distinguish the required actions.

## Files

`scripts/improve_navigation.py` reproduces isolated runs in a NEW output directory. `fly_brain_rc/imitation/navigation_cases.py` defines deterministic maps. New maps are available to the experiment API; dashboard menus were not changed. Existing deployed checkpoint and training checkpoint remain untouched.


- `artifacts/imitation/connectome.best.pt`: SHA256 `17c6eb06e6a02e0c2aa1e2e57cdab22ca9da751763535fd6de0a15c20274ee1b`
- `artifacts/imitation/connectome.pt`: SHA256 `5ab54ad573e09c842e94e17dab9221e8d4d9a97b8bfaf4be8296f99c11d7e9ad`
- `artifacts/imitation/backups/pre-compass-baseline-2026-10-08-17c6eb06e6a0/connectome.best.pt`: SHA256 `17c6eb06e6a02e0c2aa1e2e57cdab22ca9da751763535fd6de0a15c20274ee1b`

Review follow-up: checkpoint metadata records new training lineage separately from source iteration and anchor replay. The initial two trials did not retain cached training arrays; future runs retain cached neural states, labels, categories and anchors.

## Verification

Final full suite: 335 tests passed. One preceding run hit the already documented timing-dependent dashboard shutdown heartbeat test; no dashboard code was changed. Navigation tests, checkpoint roundtrip checks and syntax checks pass. Baseline SHA256 equals its saved backup.
