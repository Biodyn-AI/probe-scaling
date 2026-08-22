"""Loading result cards.

A card is one (task, dataset, model) measurement. The loader refuses to
silently double-count: two files claiming the same key means a stale copy
survived a layout change, and averaging them would corrupt every aggregate
downstream.
"""
from __future__ import annotations

from pathlib import Path

from .card import ResultCard

DATA = Path(__file__).resolve().parents[1] / "data"
CARDS = DATA / "cards"
SWEEPS = DATA / "sweeps"


def seed_dirs() -> dict[int, Path]:
    """{seed index: directory}, for every seed present in data/cards."""
    out: dict[int, Path] = {}
    for p in sorted(CARDS.glob("seed*")):
        if p.is_dir():
            out[int(p.name.removeprefix("seed"))] = p
    return out


def load_cards(cards_dir: str | Path) -> list[ResultCard]:
    root = Path(cards_dir)
    if not root.exists():
        return []
    out = []
    for p in sorted(root.rglob("*.json")):
        # macOS writes AppleDouble sidecars (._name.json) on exFAT volumes.
        if p.name.startswith("._"):
            continue
        out.append(ResultCard.model_validate_json(p.read_text(encoding="utf-8")))

    keys: dict[tuple[str, str, str], int] = {}
    for c in out:
        k = (c.task.id, c.task.dataset_id, c.model.id)
        keys[k] = keys.get(k, 0) + 1
    dupes = {k: n for k, n in keys.items() if n > 1}
    if dupes:
        raise ValueError(f"duplicate cards under {root}: {dupes}")
    return out


def load_all_seeds() -> dict[int, list[ResultCard]]:
    return {s: load_cards(d) for s, d in seed_dirs().items()}
