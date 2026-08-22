# Same probe, different depths

Reproduction code and data for **"Same probe, different depths: how the gap between
output and internals scales in biological foundation models."**

Every number, table and figure in the paper is regenerated from the result cards in
this repository by the scripts below. Nothing is stored pre-computed: the test suite
recomputes each published value and fails if it moves.

```bash
pip install -r requirements.txt
make all          # analyses, figures, and the test that pins every published number
```

## What this repository is, and is not

**It is** the analysis surface of one paper: the released measurements, the code that
turns them into the reported statistics, and the manuscript source.

**It is not** the evaluation harness. The code that runs models — adapters, fine-tuning,
dataset construction, the job runner — belongs to the wider evaluation programme and is
not needed to check anything claimed in the paper. Reproducing the paper does not
require a GPU, model weights, or network access.

## The measurement in one paragraph

Each model is read at three surfaces. **B**, behavioural, is the model's sanctioned
output — its final-layer representation, used as intended. **P**, probe, is a linear
readout of its best *internal* layer, chosen on held-out validation data. **F**,
elicited, is the score after light fine-tuning under a pinned budget. B and P are read
through the same linear readout at the same width, so a difference between them is a
property of the model rather than of the probe. The **elicitation gap** is P − B and
**recoverability** is F − B. The paper asks how these scale with model size, and
separately measures the fine-tuning budget needed to reach a *fixed* score.

## Layout

```
data/
  cards/seed0 … seed5/   235 result cards: one (task, dataset, model) measurement each
  sweeps/                iso-performance budget sweeps, three seeds
probe_scaling/
  card.py                the result-card schema
  analysis.py            gap, recoverability and cost scaling, with their guards
  io.py                  card loading; refuses to silently double-count
scripts/
  analyse_gap.py             Table 4, Table 5, Figure 1
  analyse_recoverability.py  Table 6
  analyse_carp_geometry.py   the width/depth contrasts in Section 3.2
  make_figures.py            all three figures
  build_bib.py               regenerates paper/refs.bib from doi.org and arXiv
tests/
  test_paper_numbers.py  every published value, re-derived
paper/
  paper.tex, refs.bib, figures/
supplementary/
  failure-modes.md       the full taxonomy of measurement defects
```

## Reproducing each claim

| Paper | Command |
|---|---|
| Table 4, Table 5, Figure 1 — gap slopes | `python scripts/analyse_gap.py` |
| Table 6 — recoverability | `python scripts/analyse_recoverability.py` |
| Table 7, Figure 2 — cost scaling | `python scripts/analyse_cost.py` |
| Section 3.2 — CARP width vs depth | `python scripts/analyse_carp_geometry.py` |
| Figures 1–3 | `python scripts/make_figures.py` |
| Every published number | `pytest -q` |
| Bibliography | `python scripts/build_bib.py` |

## Result cards

A card records one measurement and the controls it passed. It cannot be constructed
without at least one baseline, without a permutation null, and without either a ceiling
or a stated reason none applies — those three rules are enforced by the schema in
`probe_scaling/card.py`, not by convention. Each card also records the readout width
each surface was read at, so the identical-capacity constraint is checkable after the
fact, and how the probe layer was selected (`validation`, or an explicit degraded
fallback — never `test`).

See `data/README.md` for the field-by-field description.

## Statistics

Analyses fit one ordinary-least-squares slope per seed over log₁₀(parameters), then test
the across-seed mean against zero with a Student-*t* interval. Seeds vary the train/test
split; the linear readout is a deterministic convex fit and the fine-tuning
initialisation is held fixed, so the spread across seeds is split variance, not
initialisation variance. Rungs that fail their own permutation null do not enter a fit,
and ladders exclude randomly-initialised control arms.

## A caution, which is also the paper's point

Several of the guards in `analysis.py` exist because their absence produced a number we
believed. The full account is in `supplementary/failure-modes.md`: twenty-nine defects,
of which the twenty-six that biased a number all biased it in our favour. If you reuse
this code, keep the guards.

## Citation

See `CITATION.cff`. The manuscript is in `paper/`.

## Licence

Code is MIT. Result cards and sweep files are CC BY 4.0. The underlying model
checkpoints and benchmark datasets are third-party and carry their own licences; they
are cited in the paper and are not redistributed here.
