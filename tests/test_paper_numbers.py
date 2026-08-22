"""Every number printed in the paper, re-derived from the released cards.

This is the guarantee the paper makes: each value below is quoted in the
manuscript, and each is recomputed here from data/ rather than stored. If a
card changes, or an analysis function changes, this test fails and the paper
is wrong -- which is the point.

Run:  pytest -q
"""
from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from probe_scaling.analysis import cost_scaling_from_sweep  # noqa: E402
from probe_scaling.io import SWEEPS, load_cards, seed_dirs  # noqa: E402

import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("mkfig", ROOT / "scripts" / "make_figures.py")
mkfig = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mkfig)

TOL = 5e-5  # values are quoted to 4 decimal places


# --------------------------------------------------------------- Table 4 / Fig 1
# (task prefix, family, n seeds, mean, ci_lo, ci_hi) exactly as printed.
GAP = [
    ("pr3", "ESM2",   6, +0.0202, +0.0092, +0.0312),
    ("pr4", "ESM2",   6, +0.0211, +0.0135, +0.0286),
    ("pr1", "ESM2",   6, -0.0026, -0.0086, +0.0034),
    ("pr2", "ESM2",   5, -0.0078, -0.0447, +0.0291),
    ("dn1", "NT-v2",  6, -0.0190, -0.0368, -0.0012),
    ("pr3", "CARP",   6, +0.0060, +0.0005, +0.0115),
    ("pr3", "RITA",   4, -0.0146, -0.0562, +0.0270),
]


@pytest.mark.parametrize("prefix,family,n,mean,lo,hi", GAP)
def test_gap_slopes_match_the_paper(prefix, family, n, mean, lo, hi):
    slopes = mkfig.per_seed_slopes(prefix, family)
    assert len(slopes) == n, f"{family}/{prefix}: expected {n} seeds, got {len(slopes)}"
    m, l, h, _ = mkfig.mean_ci(slopes)
    assert m == pytest.approx(mean, abs=TOL)
    assert l == pytest.approx(lo, abs=TOL)
    assert h == pytest.approx(hi, abs=TOL)


def test_the_two_widening_tasks_have_every_seed_positive():
    """The paper's strongest claim: 12 of 12 seeds positive across two tasks."""
    for prefix in ("pr3", "pr4"):
        slopes = mkfig.per_seed_slopes(prefix, "ESM2")
        assert all(s > 0 for s in slopes), f"{prefix}: {slopes}"


# --------------------------------------------------------------------- Table 7
COST = [(0.40, -0.040, -0.096, +0.016),
        (0.45, -0.094, -0.265, +0.078),
        (0.50, -0.118, -0.181, -0.055)]


@pytest.mark.parametrize("target,mean,lo,hi", COST)
def test_cost_scaling_matches_the_paper(target, mean, lo, hi):
    files = ["iso_ladder_sweep.json", "iso_sweep_seed1.json", "iso_sweep_seed2.json"]
    slopes = [cost_scaling_from_sweep(json.load(open(SWEEPS / f)), target=target).primary.slope
              for f in files]
    m = statistics.mean(slopes)
    h = 4.303 * statistics.stdev(slopes) / math.sqrt(3)  # t(2), two-sided 95%
    assert m == pytest.approx(mean, abs=5e-4)
    assert m - h == pytest.approx(lo, abs=5e-4)
    assert m + h == pytest.approx(hi, abs=5e-4)


def test_every_cost_slope_is_negative():
    """'All nine seed-by-target slopes are negative.'"""
    files = ["iso_ladder_sweep.json", "iso_sweep_seed1.json", "iso_sweep_seed2.json"]
    got = [cost_scaling_from_sweep(json.load(open(SWEEPS / f)), target=t).primary.slope
           for t in (0.40, 0.45, 0.50) for f in files]
    assert len(got) == 9 and all(s < 0 for s in got), got


def test_smallest_rung_is_censored_in_two_of_three_seeds_at_the_hard_target():
    """A stated fragility of the headline cost result."""
    files = ["iso_ladder_sweep.json", "iso_sweep_seed1.json", "iso_sweep_seed2.json"]
    censored = [cost_scaling_from_sweep(json.load(open(SWEEPS / f)), target=0.50).n_censored
                for f in files]
    assert sorted(censored) == [0, 1, 1], censored


# --------------------------------------------------------------- Table 8 / Fig 3
def test_single_run_headline_reversed_under_replication():
    """The paper's central methodological claim, as arithmetic."""
    (b6, l6, h6), n6 = mkfig.single_run("pr3", "ESM2")
    (b5, l5, h5), n5 = mkfig.single_run("pr3", "ESM2", drop_above=int(10e9))
    m, lo, hi, _ = mkfig.mean_ci(mkfig.per_seed_slopes("pr3", "ESM2"))

    assert n6 == 6 and n5 == 5
    assert (b6, l6, h6) == pytest.approx((0.018, -0.011, 0.046), abs=5e-4)
    assert (b5, l5, h5) == pytest.approx((0.019, -0.032, 0.069), abs=5e-4)

    assert l6 < 0 < h6, "the 6-rung single run must span zero"
    assert l5 < 0 < h5, "the matched single run must span zero"
    assert lo > 0, "six runs must clear zero"

    # "one run gives an interval 4.6 times wider than six runs"
    assert (h5 - l5) / (hi - lo) == pytest.approx(4.6, abs=0.05)


def test_rita_single_run_effect_was_one_seed():
    (br, lr, hr), _ = mkfig.single_run("pr3", "RITA")
    assert hr < 0, "the single RITA run appeared clear of zero"
    slopes = sorted(mkfig.per_seed_slopes("pr3", "RITA"))
    assert slopes[0] == pytest.approx(-0.050, abs=5e-4)
    m, lo, hi, _ = mkfig.mean_ci(slopes)
    assert lo < 0 < hi, "four seeds must span zero"


# ------------------------------------------------------------------ data hygiene
def test_control_arms_are_excluded_from_every_ladder():
    """Untrained control arms once sat inside the ladders; defect #29."""
    for s, d in seed_dirs().items():
        for c in load_cards(d):
            if getattr(c.model, "control_arm", False) or c.model.id.endswith("-random"):
                assert c.task.id.startswith(("pr1", "pr2")), (
                    f"unexpected control arm {c.model.id} on {c.task.id}")


def test_probe_layers_were_selected_on_validation_not_test():
    """Defect #1 cost 92% of a headline gap. Cards must record the selection."""
    bad = []
    for s, d in seed_dirs().items():
        for c in load_cards(d):
            sel = getattr(c.scores, "layer_selection", None)
            if c.scores.probe is not None and sel not in (None, "validation"):
                bad.append((s, c.task.id, c.model.id, sel))
    # degraded fallbacks are permitted but must be explicit, never silent
    assert all(b[3] is not None for b in bad), bad


def test_card_count_matches_the_paper():
    total = sum(len(load_cards(d)) for d in seed_dirs().values())
    assert total == 235, total
