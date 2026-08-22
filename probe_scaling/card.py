"""The ResultCard — SONDE's unit of output, and the enforcement point for the Standard.

A card cannot be constructed unless it carries its control triple: at least one
baseline, a null test, and either a ceiling or an explicit reason why a ceiling does
not apply. That is rule 1, 2 and 5 of the Standard enforced in code rather than in a
reviewer's memory.

Schema is versioned. Never mutate a released version in place; add a new one.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

SCHEMA_VERSION = "0.1"

Modality = Literal[
    "single-cell", "protein", "dna", "methylation", "spatial", "text"
]

# The ten capability axes from the programme plan (§5A).
Axis = Literal[
    "A1",  # representation quality
    "A2",  # native-task performance
    "A3",  # transfer / generalisation
    "A4",  # causal / interventional
    "A5",  # structural knowledge
    "A6",  # compositionality
    "A7",  # calibration
    "A8",  # robustness
    "A9",  # efficiency
    "A10",  # scaling
]


class ModelRef(BaseModel):
    """Identity and provenance of the evaluated model."""

    id: str
    modality: Modality
    params: int | None = None
    # True for an untrained architecture-matched control. These exist to ask
    # whether a scaling curve is about learned biology or about parameter count,
    # so they must never enter a scaling fit as if they were models. They were
    # doing exactly that on both transcription-factor tasks: four random-init
    # arms sat in the ESM2 ladder, at the same parameter counts as their trained
    # twins, silently doubling four of the six rungs.
    control_arm: bool = False
    checkpoint_sha256: str | None = None
    release_date: str | None = None
    # Model family, e.g. "ESM2". Points sharing a family and release date differ
    # only in scale, which is what makes a scaling trend readable.
    family: str | None = None
    source: str | None = None
    notes: str | None = None


class TaskRef(BaseModel):
    """Identity and provenance of the evaluation task."""

    id: str
    version: str
    axis: list[Axis]
    metric: str
    dataset_id: str
    dataset_sha256: str | None = None
    n_units: int | None = None
    higher_is_better: bool = True


class Scores(BaseModel):
    """The four measurement surfaces.

    zero_shot   -- the model used with no task training at all (what a practitioner
                   gets out of the box).
    behavioural -- B: fixed-capacity readout of the model's *sanctioned* output
                   representation.
    probe       -- P: the same fixed-capacity readout on the best internal layer.
    elicited    -- F: light fine-tuning under a fixed data/step budget.

    B and P must use an identical readout budget or the gap is meaningless. The
    budget is recorded in `readout_budget`.
    """

    zero_shot: float | None = None
    behavioural: float | None = None
    probe: float | None = None
    elicited: float | None = None
    readout_budget: dict[str, Any] = Field(default_factory=dict)
    probe_best_layer: str | None = None
    # How that layer was chosen. Anything other than "validation" means the
    # measurement degraded, and the gap is not the quantity the page describes.
    layer_selection: str | None = None
    # Readout width actually used for each surface. The identical-capacity
    # promise is what makes B and P comparable, and it is not automatic: a
    # representation narrower than the budget is read at its own width. When
    # these differ, part of any gap is capacity rather than representation.
    effective_width: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def at_least_one_score(self) -> "Scores":
        if all(
            getattr(self, k) is None
            for k in ("zero_shot", "behavioural", "probe", "elicited")
        ):
            raise ValueError("Scores must carry at least one measured value")
        return self


class Derived(BaseModel):
    elicitation_gap: float | None = None
    recoverability: float | None = None
    headroom: float | None = None  # (ceiling - best observed) / ceiling


class NullResult(BaseModel):
    """Rule 5: kill it with a null before making a positive claim."""

    method: str
    n_permutations: int
    matched_covariate: str | None = None
    p_value: float
    null_mean: float
    null_std: float
    observed: float

    @property
    def survives(self) -> bool:
        return self.p_value < 0.05


class CeilingResult(BaseModel):
    """Rule 2: report the score against what is actually achievable."""

    method: str
    value: float
    note: str | None = None


class Controls(BaseModel):
    """Rules 2, 4 and 5, made structural."""

    group_stratified: bool
    group_key: str | None = None
    n_groups_train: int | None = None
    n_groups_test: int | None = None
    null: NullResult
    ceiling: CeilingResult | None = None
    ceiling_not_applicable: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def ceiling_or_reason(self) -> "Controls":
        if self.ceiling is None and not self.ceiling_not_applicable:
            raise ValueError(
                "Standard rule 2: supply a ceiling, or state why one does not apply "
                "via ceiling_not_applicable"
            )
        if self.group_stratified and not self.group_key:
            raise ValueError("group_stratified=True requires group_key")
        return self


class Uncertainty(BaseModel):
    """Rule 3: publish reversal risk, not a bare rank.

    `gap_ci95` is computed *paired* — B and P are recomputed on the same resampled
    units and differenced within each replicate. B and P are strongly correlated,
    so the interval on their difference is far tighter than the interval on either
    one, and comparing a small gap against the marginal CI of the probe score would
    understate the evidence badly.
    """

    ci95: tuple[float, float] | None = None
    n_bootstrap: int | None = None
    std: float | None = None
    gap_ci95: tuple[float, float] | None = None
    gap_positive_fraction: float | None = None
    rank_reversal_risk: dict[str, float] = Field(default_factory=dict)


class Compute(BaseModel):
    device: str
    seconds: float | None = None
    gpu_hours: float | None = None
    peak_memory_gb: float | None = None


class Provenance(BaseModel):
    """Rule 6: pin everything."""

    code_commit: str | None = None
    run_id: str
    run_date: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    seed: int | None = None
    env_lock_sha256: str | None = None
    harness_version: str | None = None


class ResultCard(BaseModel):
    """One model, one task, one honest number — with the evidence to read it."""

    schema_version: str = SCHEMA_VERSION
    model: ModelRef
    task: TaskRef
    scores: Scores
    derived: Derived = Field(default_factory=Derived)
    baselines: dict[str, float]
    controls: Controls
    uncertainty: Uncertainty = Field(default_factory=Uncertainty)
    compute: Compute
    provenance: Provenance
    disclosure_tier: Literal[0, 1, 2] = 0
    caveats: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def enforce_standard(self) -> "ResultCard":
        # Rule 1: beat a dumb baseline, or say you didn't -- either way, carry one.
        if not self.baselines:
            raise ValueError(
                "Standard rule 1: every card must carry at least one baseline"
            )
        return self

    # -- derived quantities -------------------------------------------------

    def _surfaces(self) -> list[float]:
        s = self.scores
        return [
            v for v in (s.zero_shot, s.behavioural, s.probe, s.elicited) if v is not None
        ]

    @property
    def best_surface(self) -> float:
        """The most that could be extracted from this model, on any surface.

        Always the maximum, on both task directions, and the direction only
        changes how it is *interpreted* and sorted:

          capability task -- the most the model demonstrably knows. Good.
          nuisance task   -- the most an adversary or a confound can recover. Bad.

        Taking the minimum on nuisance tasks would let a model hide leakage behind
        one weak readout: Geneformer's zero-shot centroid recovers donor at 0.22
        while its layer-3 probe recovers it at 0.69, and reporting 0.22 would rank
        it the most donor-invariant model measured. It is the least.
        """
        return max(self._surfaces())

    def compute_derived(self) -> "ResultCard":
        """Fill `derived` from `scores` and `controls`. Idempotent."""
        s = self.scores
        if s.probe is not None and s.behavioural is not None:
            self.derived.elicitation_gap = round(s.probe - s.behavioural, 6)
        if s.elicited is not None and s.behavioural is not None:
            self.derived.recoverability = round(s.elicited - s.behavioural, 6)
        # Headroom is only meaningful when the ceiling is a target to approach.
        # On a lower-is-better task the "ceiling" is a nuisance denominator, and
        # (ceiling - best)/ceiling would read as virtue for a leaky representation.
        if self.controls.ceiling is not None and self.task.higher_is_better:
            ceil = self.controls.ceiling.value
            if ceil:
                self.derived.headroom = round((ceil - self.best_surface) / ceil, 6)
        return self

    @property
    def best_baseline(self) -> float:
        """The strongest cheap alternative — the most any baseline extracts.

        On a nuisance task this is the leakiest cheap approach, so clearing it
        means the representation is at least not the worst option available. It is
        a weak bar by design; the informative comparator on those tasks is the
        retention figure in `controls.extra`, measured under the same budget.
        """
        return max(self.baselines.values())

    @property
    def beats_baseline(self) -> bool:
        if self.task.higher_is_better:
            return self.best_surface > self.best_baseline
        return self.best_surface < self.best_baseline
