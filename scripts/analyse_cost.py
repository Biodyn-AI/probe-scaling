"""Iso-performance cost scaling: Table 7 and Figure 2 of the paper.

Holds the score fixed and measures the fine-tuning budget required to reach it,
which removes the confound that a fixed budget is worth more to a larger model.
Each budget is a separate run annealed to its own endpoint; reading intermediate
checkpoints off one long run is biased, and not even with a constant sign.
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from probe_scaling.analysis import cost_scaling_from_sweep
from probe_scaling.io import SWEEPS

SWEEP_FILES = ["iso_ladder_sweep.json", "iso_sweep_seed1.json", "iso_sweep_seed2.json"]
TARGETS = (0.40, 0.45, 0.50)
T_CRIT_N3 = 4.303  # Student-t, 2 d.o.f., two-sided 95%


def main() -> None:
    print("=" * 74)
    print("iso-performance cost scaling: decades of budget per decade of parameters")
    print("negative means a larger model reaches the same score for less budget")
    print("=" * 74)

    all_negative = True
    for target in TARGETS:
        per_seed = []
        for f in SWEEP_FILES:
            r = cost_scaling_from_sweep(json.load(open(SWEEPS / f)), target=target)
            per_seed.append(r)
        slopes = [r.primary.slope for r in per_seed]
        all_negative &= all(s < 0 for s in slopes)

        m = statistics.mean(slopes)
        h = T_CRIT_N3 * statistics.stdev(slopes) / math.sqrt(len(slopes))
        clears = (m - h) * (m + h) > 0

        print(f"\ntarget macro-F1 = {target:.2f}")
        for i, r in enumerate(per_seed):
            print(f"   seed {i}: slope {r.primary.slope:+.4f}   rungs used {r.primary.n}"
                  f"   censored {r.n_censored}   non-monotone {r.n_nonmonotone}")
        print(f"   pooled : {m:+.3f}   95% CI [{m - h:+.3f}, {m + h:+.3f}]"
              f"   -> {'clears zero' if clears else 'spans zero'}")
        if clears:
            print(f"            a tenfold larger model reaches the same score on"
                  f" {10 ** m:.2f} of the budget")

    print(f"\nall nine seed-by-target slopes negative: {all_negative}")
    print("\nFragilities that belong with the headline number, and are not in the table:")
    print("  - only the hardest target clears zero;")
    print("  - the pooled interval spans zero if any single seed is dropped;")
    print("  - the smallest rung is censored in two of three seeds at target 0.50.")


if __name__ == "__main__":
    main()
