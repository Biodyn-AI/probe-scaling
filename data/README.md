# Released data

235 result cards and 3 iso-performance sweeps. Together these are the complete
input to every number in the paper.

```
cards/seed0 … seed5/    one directory per repeat; seeds vary the train/test split
sweeps/                 iso_ladder_sweep.json (seed 0), iso_sweep_seed1, iso_sweep_seed2
```

Cards per task, across all seeds:

| task id | what it is | cards |
|---|---|---|
| `pr3-fold-recognition` | CATH fold recognition, protein | 72 |
| `pr1-tf-identity` | human transcription-factor identity, protein | 42 |
| `pr2-tf-family` | human transcription-factor family, protein | 42 |
| `pr4-ec-number` | Swiss-Prot enzyme commission number, protein | 31 |
| `dn1-enhancer-class` | enhancer classification, genome | 24 |
| `br1-viral-fitness` | deep mutational scanning, protein | 18 |
| `sc1`, `sc4` | single-cell | 6 |

The paper analyses the first five. `br1` carries an F surface but no clean scale
ladder; the single-cell tasks are out of scope for the structural reason given in
the Limitations, and are released for completeness rather than analysed.

## Card fields

A card is one `(task, dataset, model)` measurement.

| field | meaning |
|---|---|
| `model.id`, `model.params` | checkpoint and its parameter count — the x-axis of every fit |
| `model.control_arm` | true for randomly-initialised twins, which are excluded from ladders |
| `task.id`, `task.dataset_id`, `task.dataset_sha256` | what was measured, and on exactly which data |
| `scores.behavioural` | **B** — the sanctioned output surface |
| `scores.probe` | **P** — best internal layer, selected on validation |
| `scores.elicited` | **F** — after light fine-tuning, where an adapter supports it |
| `scores.probe_best_layer` | which layer won |
| `scores.layer_selection` | `validation`, or an explicit degraded fallback. Never `test` |
| `scores.readout_budget` | the readout configuration, identical across surfaces |
| `scores.effective_width` | width each surface was actually read at — makes the identical-capacity constraint checkable after the fact |
| `derived.elicitation_gap` | P − B |
| `derived.recoverability` | F − B |
| `derived.headroom` | distance left to the ceiling |
| `baselines.*` | at least one is required by the schema |
| `controls.null` | permutation null: method, p-value, null mean and spread |
| `controls.ceiling` | empirical ceiling, or an explicit statement that none applies |
| `controls.group_key`, `controls.extra.clustering` | how the split was made disjoint |
| `uncertainty.ci95` | bootstrap interval on the score |
| `provenance.seed`, `provenance.run_date`, `provenance.run_id` | which repeat this is |
| `caveats` | free text; read these |

Three of those are enforced by the schema rather than by convention: a card cannot
be constructed without at least one baseline, without a permutation null, or
without either a ceiling or a stated reason none applies.

## Two things to know before reusing these

**Not every card should enter a fit.** Rungs whose permutation null they do not
beat are excluded by the analysis code, as are randomly-initialised control arms
and any card whose `layer_selection` is a degraded fallback rather than
`validation`. `probe_scaling/analysis.py` applies these filters; if you read the
cards directly, apply them yourself. Fitting a scaling slope over rungs that
failed their own null is a defect we shipped and had to retract.

**`model.source` and some `caveats` strings name `probe.adapters...`.** These are
recorded provenance from runs made before the programme was renamed, and are left
exactly as written. Rewriting a provenance field to match a later name would make
the record say something that was not true when the run happened.
