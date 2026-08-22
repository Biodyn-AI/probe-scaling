"""Does the elicitation gap track parameters, depth, or width?

WHY ONLY CARP CAN ANSWER THIS. On ESM2 depth and width rise together at every
rung, so "the gap widens with scale" cannot be decomposed -- parameters, layers
and hidden size are one variable wearing three names. CARP's four rungs move them
separately:

    carp-600k    d_model  128, 16 layers,    607,630 params
    carp-38m     d_model 1024, 16 layers, 37,889,294 params   <- width x8, depth SAME
    carp-76m     d_model 1024, 32 layers, 75,736,334 params   <- depth x2, width SAME
    carp-640m    d_model 1280, 56 layers, 642,950,670 params  <- both, and slim off

So two adjacent contrasts isolate one variable each:

    600k -> 38m   width x8.0   depth x1.0   params x62.4
    38m  -> 76m   width x1.0   depth x2.0   params x2.0

If the gap moves on the first and not the second, it tracks width. If the reverse,
depth. If it moves on both roughly in proportion to parameters, then parameter
count is the right summary after all and ESM2's confounding never mattered.

Six seeds per rung, so each contrast is a paired difference with an interval
rather than a single subtraction. The pairing matters: seed noise is shared
within a seed (same split, same initialisation stream), so differencing within a
seed removes it, which a between-seed comparison would not.

This is written before the data arrives, deliberately. The last quantity this
programme cared about was computed in a throwaway script and inherited none of
the guards; see docs/failure-modes.md, entry 10.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math
import os

from probe_scaling.analysis import _t_crit
from probe_scaling.io import load_cards

# d_model and n_layers read from the official checkpoints, not from the paper.
GEOMETRY = {
    "carp-600k": {"params": 607_630, "width": 128, "depth": 16},
    "carp-38m": {"params": 37_889_294, "width": 1024, "depth": 16},
    "carp-76m": {"params": 75_736_334, "width": 1024, "depth": 32},
    "carp-640m": {"params": 642_950_670, "width": 1280, "depth": 56},
}

CONTRASTS = [
    ("width only", "carp-600k", "carp-38m"),
    ("depth only", "carp-38m", "carp-76m"),
    ("both", "carp-76m", "carp-640m"),
]


def gaps_by_seed(dirs):
    """{seed: {model_id: gap}} for CARP cards on fold recognition."""
    out = {}
    for seed, d in sorted(dirs.items()):
        if not os.path.isdir(d):
            continue
        for c in load_cards(d):
            if c.task.id != "pr3-fold-recognition" or c.model.family != "CARP":
                continue
            if c.scores.probe is None or c.scores.behavioural is None:
                continue
            out.setdefault(seed, {})[c.model.id] = c.scores.probe - c.scores.behavioural
    return out


def paired(values):
    n = len(values)
    if n < 2:
        return None
    m = sum(values) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1))
    t = _t_crit(n - 1)
    half = t * sd / math.sqrt(n)
    return m, sd, m - half, m + half, n


if __name__ == "__main__":
    from probe_scaling.io import seed_dirs
    dirs = {s: str(p) for s, p in seed_dirs().items()}

    by_seed = gaps_by_seed(dirs)
    if not by_seed:
        raise SystemExit("no CARP cards found yet")

    print("gap per rung, per seed")
    print(f"  {'seed':>4}  " + "  ".join(f"{m:>11}" for m in GEOMETRY))
    for seed in sorted(by_seed):
        row = by_seed[seed]
        print(f"  {seed:>4}  " + "  ".join(
            f"{row.get(m, float('nan')):>+11.4f}" for m in GEOMETRY))

    print("\npaired contrasts (within-seed differences, so seed noise cancels)")
    for label, a, b in CONTRASTS:
        diffs = [by_seed[s][b] - by_seed[s][a]
                 for s in sorted(by_seed) if a in by_seed[s] and b in by_seed[s]]
        r = paired(diffs)
        ga, gb = GEOMETRY[a], GEOMETRY[b]
        factors = (f"width x{gb['width'] / ga['width']:.1f}  "
                   f"depth x{gb['depth'] / ga['depth']:.1f}  "
                   f"params x{gb['params'] / ga['params']:.1f}")
        if not r:
            print(f"  {label:<11} {a} -> {b}: {len(diffs)} seed(s), no interval")
            continue
        m, sd, lo, hi, n = r
        verdict = "gap CHANGES" if (lo > 0 or hi < 0) else "no change detectable"
        print(f"  {label:<11} {factors}")
        print(f"              mean {m:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  n={n}  -> {verdict}")

    print("\nper-variable slopes across all four rungs, for comparison")
    for var in ("params", "width", "depth"):
        slopes = []
        for seed, row in by_seed.items():
            pts = [(math.log10(GEOMETRY[m][var]), g) for m, g in row.items() if m in GEOMETRY]
            if len(pts) < 3:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            n = len(xs)
            mx, my = sum(xs) / n, sum(ys) / n
            sxx = sum((x - mx) ** 2 for x in xs)
            if sxx <= 0:
                continue
            slopes.append(sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx)
        r = paired(slopes)
        if not r:
            print(f"  {var:<7}: too few seeds")
            continue
        m, sd, lo, hi, n = r
        note = "clear of zero" if (lo > 0 or hi < 0) else "spans zero"
        print(f"  {var:<7}: mean {m:+.5f} per decade  CI [{lo:+.5f}, {hi:+.5f}]  n={n}  {note}")
    print("\nNote: params, width and depth are correlated across these four rungs,")
    print("so the three slopes are not independent. The paired contrasts above are")
    print("the identifying evidence; these slopes are context.")
