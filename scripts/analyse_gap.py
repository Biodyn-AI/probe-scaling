"""Does the elicitation gap trend survive replication, per model family?

WHY. The site's headline is that the gap between what a model says and what it
internally knows shows no measurable change with scale. That was measured on
ESM2 alone: six rungs, 3.27 decades, slope +0.018 with an interval spanning
zero. A second family, RITA, then gave the opposite answer on the same task --
four rungs, 1.15 decades, slope -0.050 with an interval clear of zero, the gap
going negative at the largest model.

One of those is going on a public page and the other contradicts it, so the
question is which survives seeds. RITA's interval is tight because four points
happened to fall almost on a line, which is two degrees of freedom carrying a
conclusion; the same shape in the budget sweep was called fragile an hour
earlier and consistency demands the same treatment here.

WHAT IT DOES. Per seed, the guarded `gap_scaling` from the library -- not a
hand-rolled fit -- so the null-support gate and the headroom normalisation come
along. Then the across-seed mean against zero with a Student-t interval, the
same rule the recoverability pre-registration uses. Every seed is shown.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math
import os
import sys

from probe_scaling.analysis import _t_crit, gap_scaling
from probe_scaling.io import load_cards


# A slope is only comparable across seeds if the seeds climbed the same ladder.
# Seed 0 of the ESM2 protein set carries a 15B rung the others do not, because
# the replicates were produced for a different experiment. Mixing a six-rung
# ladder with five-rung ones compares slopes over different spans and different
# top rungs -- and the top rung is where a flattening curve gets caught, which
# is precisely how a missing 3B rung once turned a null into the strongest
# result in the programme. The intersection is used, and what was dropped is
# printed rather than absorbed.
def per_seed_gap(task: str, dataset: str, family: str, dirs: dict[int, str]):
    out = {}
    degenerate_seeds: list[int] = []
    ladders = {}
    for seed, d in sorted(dirs.items()):
        if not os.path.isdir(d):
            continue
        ladders[seed] = {c.model.params for c in load_cards(d)
                         if c.task.id == task and c.task.dataset_id == dataset
                         and c.model.family == family and c.model.params}
    ladders = {s: l for s, l in ladders.items() if len(l) >= 3}
    if not ladders:
        return out
    common = set.intersection(*ladders.values())
    dropped = sorted(set.union(*ladders.values()) - common)
    if dropped:
        print(f"  common ladder: {len(common)} rungs; dropped "
              f"{', '.join(f'{d:,}' for d in dropped)} (not present in every seed)")
    for seed, d in sorted(dirs.items()):
        if not os.path.isdir(d):
            continue
        cards = [c for c in load_cards(d)
                 if c.task.id == task and c.task.dataset_id == dataset
                 and c.model.family == family and c.model.params in common]
        if len(cards) < 3:
            if cards:
                print(f"  seed {seed}: only {len(cards)} rung(s) -- skipped")
            continue
        g = gap_scaling(cards)
        if g.n_degenerate:
            # Not a seed with a small effect -- a seed where the probe layer
            # could not be chosen at all, so every gap is zero by construction.
            degenerate_seeds.append(seed)
        if not g.raw:
            why = (f"all {g.n_degenerate} rung(s) degenerate"
                   if g.n_degenerate else "no fit")
            print(f"  seed {seed}: EXCLUDED -- {why}")
            continue
        pts = sorted(g.points, key=lambda r: r["params"])
        gaps = "  ".join(f"{r['gap']:+.4f}" for r in pts)
        extra = f", {g.n_degenerate} degenerate" if g.n_degenerate else ""
        print(f"  seed {seed}: {gaps}   raw {g.raw.slope:+.5f}  "
              f"norm {g.normalised.slope:+.5f}  ({g.n_supported}/{len(pts)} supported{extra})")
        out[seed] = (g.raw.slope, g.normalised.slope, len(pts))

    if degenerate_seeds:
        print(f"\n  !! {len(degenerate_seeds)} seed(s) dropped for a degenerate probe "
              f"layer: {degenerate_seeds}")
        print("     A seed survives only when its split left enough classes to carve a")
        print("     validation set from, so the surviving seeds are NOT a random subset --")
        print("     they are the ones whose split happened to be easier. Running more seeds")
        print("     does not fix this; it is a property of the task's class structure.")
    return out


def across(label: str, values: list[float]):
    n = len(values)
    if n < 2:
        print(f"  {label}: only {n} seed(s), no interval")
        return
    mean = sum(values) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    t = _t_crit(n - 1)
    half = t * sd / math.sqrt(n)
    lo, hi = mean - half, mean + half
    verdict = ("CLOSES with scale" if hi < 0 else
               "WIDENS with scale" if lo > 0 else
               "no trend detectable at this power")
    same = all(v < 0 for v in values) or all(v > 0 for v in values)
    print(f"  {label}: n={n} seeds  mean {mean:+.5f}  s.d. {sd:.5f}  "
          f"95% CI [{lo:+.5f}, {hi:+.5f}]  (t*={t})")
    print(f"     -> {verdict}" + ("   [all seeds agree in sign]" if same else
                                  "   [seeds DISAGREE in sign]"))


if __name__ == "__main__":
    # Every task that has more than one seed, so the same rule runs everywhere
    # rather than being retyped per task at analysis time. Usage:
    #   python analyse_family_gap.py                 -> all tasks, their own family
    #   python analyse_family_gap.py pr3 ESM2 RITA   -> one task, chosen families
    TASKS = {
        "pr1": ("pr1-tf-identity", "human-tf", ["ESM2"]),
        "pr2": ("pr2-tf-family", "human-tf-family", ["ESM2"]),
        "pr3": ("pr3-fold-recognition", "cath-fold", ["ESM2", "CARP", "RITA"]),
        "pr4": ("pr4-ec-number", "swissprot-ec", ["ESM2"]),
        "dn1": ("dn1-enhancer-class", "nt-enhancers", ["Nucleotide Transformer v2"]),
    }

    from probe_scaling.io import seed_dirs
    dirs = {s: str(p) for s, p in seed_dirs().items()}

    args = sys.argv[1:]
    which = [a for a in args if a in TASKS] or list(TASKS)
    fam_override = [a for a in args if a not in TASKS]

    for key in which:
        task, dataset, families = TASKS[key]
        for family in (fam_override or families):
            print("=" * 74)
            print(f"{key}  {task}  --  {family}  --  gap (probe minus behavioural) vs scale")
            print("=" * 74)
            res = per_seed_gap(task, dataset, family, dirs)
            if res:
                across("PRIMARY   raw gap slope/decade", [v[0] for v in res.values()])
                across("SECONDARY headroom-normalised", [v[1] for v in res.values()])
            else:
                print("  no usable seeds")
            print()
