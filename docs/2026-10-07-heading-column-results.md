# Heading-circuit column hypotheses

Implemented offline mapping in `fly_brain_rc/neural/heading_columns.py`.
The vehicle, web interface and best checkpoint are unchanged.

## Evidence

Use the official [FlyWire–hemibrain unmirrored min-comp NBLAST release](https://zenodo.org/records/10877326)
and [pinned hemibrain metadata v2.1.0](https://raw.githubusercontent.com/flyconnectome/flywire_annotations/v2.1.0/supplemental_files/Supplemental_file5_hemibrain_meta.csv).
The latter labels PB columns in instance suffixes, e.g. `EPG(PB08)_L3`.
PB08 is a type code, not column 8. Delta7 suffixes can denote multiple columns.

Dataset MD5: `3ca6d79615e1cd30b37f32a8ae845f32`.
Metadata SHA256: `cf1c6d1821cf9179584e2ba89a614bd1b184234e547b660d378c964ed46fd9e1`.
Both are checked before loading. IDs remain strings, never floating-point values.
Only compatible types are matched; maxima are aggregated by column set before
comparing different-column alternatives. No one-to-one matching is imposed.

## Actual run

Report: `artifacts/imitation/reports/heading-circuit-20261007/columns-final.json`.
Engineering filters: similarity ≥0.30, different-column margin ≥0.05.
These thresholds are not biologically validated confidence levels.
If no different-column alternative exists, the match stays unresolved.

| Population | Supported hypotheses | Unresolved |
| --- | ---: | ---: |
| EPG | 46 | 1 |
| PEN_a/PEN1 | 20 | 0 |
| PEN_b/PEN2 | 22 | 0 |
| PEG | 20 | 0 |
| Delta7 | 5 | 35 |
| Total | 113 | 36 |

All 149 FlyWire roots had NBLAST columns. EPG hypotheses cover L1–L8 and
R1–R8. The unresolved EPG `720575940626315207` prefers R7, but its
similarity 0.274 is below the score threshold; it is not assigned.

## What this does not prove

Validated assignments: **zero**. Matches are cross-specimen morphological
hypotheses, not experimentally verified identities. Similarity is not a
probability. No angular phases have been assigned. PB columns must not be
treated as a naive 16-sector ordering; EPG, PEN and PEG projection relationships
require separate anatomical justification. No uncertainty is filled using root
order, soma position, graph spectral layout or vehicle performance.

Next gate: independently verify PB terminal morphology and published PB↔EB
phase relationships, including low-score EPG and ambiguous Delta7. Then test
constrained dynamics with type-level parameters, dark memory, yaw integration,
reanchoring and ablations before any vehicle integration. The 149-neuron induced
subgraph excludes additional inputs such as GLNO and ring neurons and is not
the complete biological compass.

## Reproduce

PyArrow is an optional offline dependency, not added to application startup.
This run used pyarrow 17.0.0 in `.local/heading-deps`.

```sh
PYTHONPATH=.local/heading-deps /opt/miniconda3/envs/flybrain/bin/python \
  -m fly_brain_rc.neural.heading_columns \
  --anatomy artifacts/imitation/reports/heading-circuit-20261007/anatomy.json \
  --metadata .local/connectome/annotations/hemibrain-v2.1.0-meta.csv \
  --nblast .local/connectome/annotations/nblast-flywire-hemibrain-v783.feather \
  --out /tmp/heading-columns-new.json
```

The output path must not already exist; prior results are not overwritten.
