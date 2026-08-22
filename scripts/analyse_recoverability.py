"""The six-seed recoverability test, exactly as pre-registered.

See docs/preregistration-recoverability-seeds.md. This file implements that rule
and nothing else: per-seed OLS across all five rungs, across-seed mean against
zero with a Student-t interval, headroom-normalised reported as secondary, every
seed shown. It exists as a script rather than as a thing typed at a prompt so the
analysis cannot drift between the writing of the rule and the arrival of the data.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import glob
import math
import os
import sys

from probe_scaling.analysis import _t_crit
from probe_scaling.io import load_cards


# The rung set, frozen by docs/preregistration-recoverability-seeds.md.
RUNGS = (8_000_000, 35_000_000, 150_000_000, 650_000_000, 3_000_000_000)


def per_seed(task: str, dataset: str, dirs: dict[int, str]):
    rows = {}
    for seed, d in sorted(dirs.items()):
        if not os.path.isdir(d):
            continue
        cards = [c for c in load_cards(d)
                 if c.task.id == task and c.task.dataset_id == dataset]
        pts = []
        for c in cards:
            b, f = c.scores.behavioural, c.scores.elicited
            if b is None or f is None or not c.model.params:
                continue
            null = getattr(c.controls, "null", None)
            p = getattr(null, "p_value", None) if null is not None else None
            pts.append({
                "params": c.model.params, "raw": f - b,
                "norm": (f - b) / (1 - b) if b < 1 else float("nan"),
                "supported": p is not None and p <= 0.05,
            })
        pts.sort(key=lambda r: r["params"])
        if pts:
            rows[seed] = pts
    return rows


def ols(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx


def report(task, dataset, dirs, label):
    rows = per_seed(task, dataset, dirs)
    print(f"\n{'=' * 74}\n{label}   ({len(rows)} seeds)\n{'=' * 74}")
    if not rows:
        print("  no cards found")
        return
    slopes_raw, slopes_norm = [], []
    for seed, pts in sorted(rows.items()):
        xs = [math.log10(p["params"]) for p in pts]
        sup = sum(1 for p in pts if p["supported"])
        vals = "  ".join(f"{p['raw']:+.4f}" for p in pts)
        # The pre-registration froze the rung set. A seed missing one is not a
        # smaller version of the same measurement -- on this ladder the top rung
        # is the one that comes back down, so a truncated seed reports a
        # steeper slope than it has. Seed 5 arrived with 4 of 5 rungs (its 3B
        # card was written as 0 bytes by a failing network volume) and scored
        # +0.045, the highest of six, which would have flipped the pooled
        # verdict to positive. Excluded and re-run, per the pre-registration.
        if len(pts) != len(RUNGS):
            missing = sorted(set(RUNGS) - {p["params"] for p in pts})
            print(f"  seed {seed}: {len(pts)}/{len(RUNGS)} rungs -- EXCLUDED, "
                  f"missing {', '.join(f'{m:,}' for m in missing)} (re-run, do not omit)")
            continue
        if sup < 3:
            print(f"  seed {seed}: {sup} supported -- REFUSED")
            continue
        sr, sn = ols(xs, [p["raw"] for p in pts]), ols(xs, [p["norm"] for p in pts])
        slopes_raw.append(sr)
        slopes_norm.append(sn)
        print(f"  seed {seed}: {vals}   raw slope {sr:+.5f}  norm {sn:+.5f}  ({sup}/{len(pts)} supported)")

    def summarise(sl, name):
        n = len(sl)
        if n < 2:
            print(f"  {name}: only {n} seed(s), no interval")
            return
        m = sum(sl) / n
        sd = math.sqrt(sum((x - m) ** 2 for x in sl) / (n - 1))
        se, t = sd / math.sqrt(n), _t_crit(n - 1)
        lo, hi = m - t * se, m + t * se
        verdict = ("RISES with scale" if lo > 0 else
                   "FALLS with scale" if hi < 0 else
                   "no trend detectable at this power")
        print(f"\n  {name}: n={n} seeds  mean {m:+.5f}  s.d. {sd:.5f}  "
              f"95% CI [{lo:+.5f}, {hi:+.5f}]  (t*={t})")
        print(f"     -> {verdict}"
              + ("" if lo > 0 or hi < 0 else
                 f"   [all {n} point estimates {'positive' if all(x > 0 for x in sl) else 'mixed in sign'}]"))

    summarise(slopes_raw, "PRIMARY   raw F-B per decade")
    summarise(slopes_norm, "SECONDARY headroom-normalised")


if __name__ == "__main__":
    from probe_scaling.io import seed_dirs
    dirs = {s: str(p) for s, p in seed_dirs().items()}
    print(f"seed directories found: {dirs}")
    report("pr3-fold-recognition", "cath-fold", dirs, "FOLD RECOGNITION")
    report("pr4-ec-number", "swissprot-ec", dirs, "ENZYME FUNCTION")
