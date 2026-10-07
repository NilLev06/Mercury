# Anatomical rate-dynamics pilot: actual results

Implemented a separate research network and diagnostic bench. The vehicle,
web interface, teacher and best checkpoint are unchanged. The checkpoint SHA256
matches its value before the heading work.

## Structure versus assumptions

The model keeps **149 neurons, 5,303 regional connection rows, 4,287 directed
neuron-pair edges and 72,794 synapses** from the selected FlyWire v783 graph.
Regional counts are summed for numerical dynamics, without adding, deleting or
symmetrizing connections. Connectivity uses `[post, pre]` orientation.

All neurons remain in recurrence. The **46 EPG phase hypotheses** receive an
ideal test cue and participate in column-averaged population readout. Constant
activity is removed before decoding; unequal numbers of neurons per column do
not produce an artificial direction. Unknown phases stay unknown.

Sigmoid rate dynamics use explicit signed type-pair gains, type biases and time
constants. These are **engineering hypotheses**, not measured membrane dynamics
or receptor effects. No synthetic cosine recurrent matrix, radial controller,
learned per-neuron weights or scalar heading state is carried over.

This is inspired by the type-level parameterization in
[Duan et al., NeurIPS 2025](https://papers.nips.cc/paper_files/paper/2025/file/ba5c33f0d4fd94bb522d83d255377717-Paper-Conference.pdf),
not a reproduction. That work uses a larger hemibrain circuit including GLNO and
ring populations and fits type-level parameters. Our pilot uses an incomplete
FlyWire subgraph and only a coarse excitation/Delta7-gain grid.

## Protocol and outcome

20 settings: excitation 2/4/8/12, Delta7 gain −2/−4/−8/−12/+4; all type biases
−2, taus 0.05 s, global synapse scale 1000 and ideal cue strength 6.
Positive Delta7 is a sign-sensitivity control, not a biological assertion.

Each setting starts a fresh network at zero activity for each of 12 directions,
receives a one-second ideal cue, then evolves for three seconds without inputs.
Readout is checked at every 10 ms, not only at the last frame. Required for all
directions: cue error <30°, maximum dark error <30°, minimum dark contrast >0.1.

**0 of 20 settings pass.** The minimum-error diagnostic setting has excitation
8 and positive Delta7 gain 4. Its cue errors are 0.05–7.08°, so the ideal cue is
read successfully. In darkness, however, different initial directions converge
to approximately **−0.353 rad**. A sixty-second challenge on 12 interleaved,
previously unused directions shows the same convergence, not directional memory.
The recurrence lesion leaves no decodable heading after cue removal.

This rules out these particular shared-gain settings, not all possible type-pair
parameters or fly navigation. A stationary, decodable pattern is **not** evidence
of a compass if it forgets which direction was supplied. No candidate was
promoted or connected to the car. Yaw integration is deliberately unimplemented:
the anatomical GLNO input pathway is not included or verified yet.

## Verification and reproducibility

Final report:
`artifacts/imitation/reports/heading-circuit-20261007/anatomical-dynamics-verified.json`.
It includes every configuration/case, parameter assumptions, gate thresholds,
source hashes, long-hold and recurrence-lesion results. The earlier endpoint-only
pilot report is retained but superseded by this stricter report.

```sh
/opt/miniconda3/envs/flybrain/bin/python \
  -m fly_brain_rc.neural.anatomical_heading_bench \
  --anatomy artifacts/imitation/reports/heading-circuit-20261007/anatomy.json \
  --phases artifacts/imitation/reports/heading-circuit-20261007/phases.json \
  --out /tmp/anatomical-heading-new.json
```

The output must be new; earlier results are never overwritten. No camera,
simulator, vehicle controller or checkpoint is imported by this benchmark.

11 targeted tests and the fresh **253-test full suite pass**. Independent
read-only review found no critical/important issues. Regression cases were
observed failing before their corresponding implementation/hardening.

One earlier full run exposed an existing timing-dependent dashboard shutdown
test: the close event can precede the first SSE heartbeat, yielding an empty
stream instead of a heartbeat. The dashboard/test were not edited. The isolated
rerun and final full run pass; this unrelated intermittent test remains noted.

Identity/type joins and source hashes are checked/reported. A phase report's
full lineage back through morphological matching to the anatomy file is not
automatically cryptographically enforced by the network constructor; the
current reports were checked against their source files. This must be hardened
before accepting unrelated external dataset bundles.

## Next work

Audit and include the missing GLNO/ring input populations before attempting
velocity integration. The raw type table contains four GLNO and 304 ER/ExR
neurons outside this pilot; these are an inventory, not an approved complete
circuit selection or verified input mapping. Fit the full type-pair gains/biases/kinetics with stationary
memory and dynamical-consistency objectives rather than indefinitely extending
this coarse global-gain grid. Keep long holds, unseen cue phases, lesions and
false-compass rejection as gates. Real visual landmarks can wait; do not conceal
memory failure by continuously supplying an exact simulator heading.
