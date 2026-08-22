"""Reproduction code for "Same probe, different depths".

Everything needed to regenerate every number, table and figure in the paper
from the released result cards. The experiment-running harness (model
adapters, fine-tuning, dataset construction) is deliberately not included:
this repository is the analysis surface of one paper, not the programme.
"""
from .analysis import (  # noqa: F401
    cost_scaling_from_sweep,
    gap_scaling,
    recoverability_scaling,
)
from .io import CARDS, SWEEPS, load_all_seeds, load_cards, seed_dirs  # noqa: F401

__version__ = "1.0.0"
