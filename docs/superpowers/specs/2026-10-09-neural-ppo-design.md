# Isolated neural reinforcement-learning diagnostic

Authorized aim: improve complex autonomous driving without a runtime planner,
compass or table of actions. Previous imitation runs did not meet the gate.

Freeze the selected connectome recurrence, encoder and base motor readout.
Train only the existing neural residual motor adapter using fresh on-policy PPO
rollouts; a separate training-only value network never enters inference.
Inputs remain the 31 camera/range/ego features. Coordinates enter reward only.
Each sampled squashed Gaussian action is held for five physics steps. Store
pre-tanh samples and joint old log probabilities. Never reuse old PPO rollouts.
Use per-episode GAE, distinguish terminal collision/arrival from time truncation,
and bootstrap truncations. Balance episodes, include old and complex maps.
Potential shaping uses the same decision discount as GAE. Actual arrivals and
contacts, rather than reward, determine acceptance. Keep successful old output
anchors as a soft regularizer. Save optimizer, critic, RNG and source hash.

Development gate: old seven successes with zero contacts, complex development
at least 90% successes with zero contacts (six of six for this set). Freeze a
candidate before independent final evaluation. No automatic live promotion.
Source and pre-compass backup must remain byte-identical.

Reference for clipped objective: https://arxiv.org/abs/1707.06347
