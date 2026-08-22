# Supplementary Note S2 — twenty-nine ways an elicitation measurement flatters you

Every defect below was found in this harness, by us, after it had produced a
number we believed. **Of the defects that biased a number at all, every single one biased it in our
favour.** The rest destroyed or corrupted work without pointing in a direction. Not most of them.

We report it as a classification rather than a tally, because a tally is exactly
the sort of number this document exists to distrust: an earlier version of this
header claimed "24 of 27" while naming only two exceptions, and a document about
miscounting that miscounts itself is worse than no count. #24, #26 and #27 biased
nothing and destroyed work instead. They are listed because they share the
mechanism — a check that looks present and is not.

That is the finding. A measurement pipeline is not a neutral instrument that
errs in random directions: it is built by people who want it to work, and the
errors that survive review are the ones whose output looks like success. A bug
that made the gap smaller would have been chased down in an afternoon, because
it would have looked like a bug. A bug that made the gap larger looked like a
result.

This document exists so the next group does not pay for these twice. Each entry
gives the mechanism, the measured magnitude where we have one, and what now
prevents recurrence. Where the fix is a test, the test name is given, because a
fix without a test is a fix that comes back.

---

## Class 1 — Selection on the test set

The single most productive defect family here. Six of them, the largest single class.

**1. The probe layer was chosen by argmax of test performance.**
`measure_bpf` scored every layer on the test split and reported the best. On the
one model where we isolated it, that inflated the headline elicitation gap by
about **92%**. Fix: layer chosen on a validation carve; the card records
`layer_selection` as `validation` or an explicit degraded fallback.
Test: `test_probe_layer_is_never_selected_on_the_test_set`.

**2. The fix for (1) left the same bug on its fallback branch.**
The comment said test selection "would silently reintroduce the bias". The code
beneath it did exactly that when no validation carve existed. A correct comment
is not a control.

**3. `empirical_ceiling` selected its configuration on test.**
The ceiling is the number every score is compared against, so a ceiling chosen
on test flatters every comparison at once. Fix: select on validation, refit on
train, score once on test.

**4, 5, 6. Three further selection-on-test paths** in task modules, found by
grepping for the pattern rather than by reading, after (1) and (2) showed it was a
habit rather than an incident. They are grouped here because the mechanism and the
fix were identical in each; they are three defects, not one, and the count of
twenty-nine treats them as three.

**The general rule this produced:** any argmax over a quantity you will later
report must consume held-out data that the report does not. Grep for `max(`
near a scoring call; that is where these live.

---

## Class 2 — Intervals that do not cover what they claim

**7. The gap-vs-scale interval covered 90% while claiming 95%.**
A percentile bootstrap over models, at n≈5 rungs. Measured against a true slope
of exactly zero it fired at roughly twice its nominal rate. This retracted the
only trend the programme had called significant at the time. Fix: Student-t
interval, which simulates at 95%.

**8. A "null" that was an underpowered interval.**
Three seeds of a recoverability ladder gave mean +0.021, CI [−0.0023, +0.0441],
and we wrote it up as *no effect*. Six seeds gave **+0.0203, CI [+0.0076,
+0.0330]** — the same mean, half the width, and the opposite verdict. Nothing
about the models changed; only the power did. An interval that spans zero is not
evidence of absence, and at n=3 the t-multiplier is 4.303, which will span zero
for almost anything.

---

## Class 3 — Controls computed, written down, and then ignored

**9. A scaling slope fitted over rungs that failed their own null.**
`gap_scaling` produced a tight 95% interval across five viral-fitness rungs.
Three of them had failed the position-matched null **printed on their own
cards** at p = 0.29, 0.14, 0.14. The pipeline computed those p-values, wrote
them to the card, and loaded them back into the function doing the fitting,
which never read them. Fix: fits count null-surviving rungs, refuse below three,
and print the count inside the verdict string.
Test: `test_scaling_refuses_a_fit_over_rungs_that_failed_their_null`.

**10. The quantity closest to publication had no guarded implementation at all.**
Recoverability lived in a throwaway script and so inherited none of the guards
the other scaling fits carried. The lesson is organisational: a result that
matters will be computed ad hoc under time pressure unless the library makes the
guarded path the easy one.

---

## Class 4 — Missing data is not missing at random

**11. An OOM removed the largest model, and the trend followed.**
ESM2-3B died of CUDA OOM on enzyme function. The four surviving rungs gave
+0.084 per decade, interval clear of zero — the strongest result we had.
Recovering the missing rung took it to **+0.040, spanning zero**. The recovered
point sat *below* the model a fifth its size.

This is systematic, not bad luck: **an out-of-memory error preferentially kills
the largest model, and the top of a ladder is exactly where a plateau
masquerading as a trend gets exposed.** Failed runs bias slopes upward. Fix:
gradient checkpointing so the rung fits; ladders with a crashed top rung are not
fitted.

**12. A truncated ladder scored highest.**
One seed arrived with 4 of 5 rungs. It scored +0.045, the highest of six, for
the same reason as (11). Fix: the analysis refuses any seed whose rung set is
not the pre-registered one, and names what is missing.

---

## Class 5 — Silent failure that looks like success

Every failure mode in this class was invisible in the output that a human
actually read.

**13. A 0-byte result card passed a completeness check.**
`find … | wc -l` counted the file. Fix: checks assert that a card parses and
carries a score, never that a file exists.

**14. Five runs exited in 2 seconds each and the log looked clean.**
A dataset path was set but the data never copied; `tail -8` showed only section
headers. **A loop whose every iteration finishes in two seconds is a signal.**

**15. Twelve of eighteen runs were silently overwritten.**
Card paths were keyed by (task, model) while a task ran on three datasets. Half
an hour of GPU time produced six cards that looked complete and were a mixture.
Fix: dataset in the key, plus a duplicate guard that raises.
Test: `test_two_datasets_on_one_task_do_not_overwrite`.

**16. A disk quota silently truncated every write.**
A 17 GB embedding cache filled the volume. Writes were then *accepted* and
truncated: a result card became 0 bytes, a shell script arrived as 2127 null
bytes, and ten runs died in 3 seconds each. The filesystem reported 106 TB free,
because that was the cluster, not the quota.

**17. Fine-tuning collapsed to a constant classifier.**
Four encoders returned exactly 0.3023255813953488 — the score of a head
answering one class on that label distribution, so a property of the data and
not of any model. The loss-descent check missed it because the loss does descend
to the class prior. Fix: flat-curve detector plus a top-class-share gate; F is
withheld rather than reported.
Test: `test_collapsed_finetuning_is_withheld_not_reported`.

**18. A withheld F was reported as a limitation of our tooling.**
ESM2-15B's card said "adapter does not support it". F had in fact run the full
budget and diverged (0.381 → 0.013). The card converted the most interesting
thing on it — the largest model in the ladder falling apart under a budget every
smaller model handled — into a note about our software that a reader skips.

---

## Class 6 — The instrument measured itself

**19. F measured the optimiser, not the model.**
A single shared learning rate left the classification head at its
initialisation, so F scored the optimiser's failure to converge. Fix: separate
head and adapter rates, chosen on a validation carve.

**20. The elicitation "gap" was partly a capacity artefact.**
B and P were read at different widths, so the difference included the probe's
extra capacity. Fix: identical readout budget, recorded per surface on the card.
Test: `test_identical_readout_capacity_is_recorded`.

**21. Homology "clustering" was a near-duplicate filter.**
What was described as a 30% identity control was doing something much weaker, so
the held-out set was not held out in the way the method section claimed.

**22. The measurement built to remove a confound contained one.**
Steps-to-target read off a single annealed trajectory is biased, because a
mid-run checkpoint sits at high learning rate while a dedicated run of that
length would have annealed. Measured on ESM2-8M: step 141 of a 400-step run
scores 0.3759; a run *budgeted* to 141 finishes at **0.4423**. That 0.066 is
larger than the entire effect under study — and the bias does not even have a
constant sign:

| steps | off trajectory | annealed | trajectory reads |
|---|---|---|---|
| 25 | 0.3162 | 0.2937 | high |
| 50 | 0.3963 | 0.3527 | high |
| 141 | 0.3759 | 0.4423 | low |

So no constant correction fixes it. The honest version needs one annealed run
per budget; bisection makes that affordable.

---

## Class 7 — Found after this document was first written

Three more inside a day of writing the first twenty-two, which is the honest rate.

**23. The embedding cache did not know which code produced an entry.**
Keyed on (model id, dataset hash) while its own docstring claimed it "cannot
serve activations computed from different preprocessing". After the RITA adapter
was fixed to read per-layer states and to trim an appended `<EOS>`, the cache
kept serving the pre-fix activations: a twelve-layer model reported sixteen
layers and a separability check that should have scored 1.0 scored chance. The
key now includes a hash of the source file behind the compute callable.
Tests: `tests/test_embcache_key.py`.

**24. A budget parameter that was accepted and never read.**
`finetune_score` took a `budget` argument. Both callers passed the *readout*
budget — a different object, governing B and P — and every implementation
discarded it and rebuilt from environment variables. Wrong type at the call
site, ignored at the implementation, silent. This one biased no published
number, because the value being dropped was never the right object; what it cost
was the interface, since the iso-cost harness had to drive an environment
variable instead of simply passing a budget.

**25. Quantisation reported as an effect.**
The first iso-cost probe reported a 2.8× budget saving for the larger model at
one target and *exactly* 1.00× at another — a slope of precisely zero. The
second was not a null: both models cross that target between the 50- and
100-step rungs, and a search that answers only in grid points rounds them to the
same one. Interpolating puts the true crossings near 78 and 66 steps, a real
gap of about 1.2×. **A suspiciously round number is a measurement artefact until
shown otherwise** — 1.00 and 0.000 should have been read as the grid speaking,
not the models. Crossings now carry an interpolated value, the analysis fits
both and refuses when they disagree, and the sweep evaluates every budget once
so several targets can be compared rather than one chosen.

**26. A deploy that reported success while destroying the queue.**
A code tarball was extracted over the live tree while jobs were running. The
transfer was interrupted and left one file — `sonde/report/card.py` — at zero
bytes. Every job afterwards died on `ImportError` in about a second, and because
each script checked only whether its command returned, **five enzyme runs, a
budget sweep and a four-model ladder all reported DONE in eight minutes having
computed nothing.**

It was caught by the shape of the log, not by any check: two "seed complete"
messages *two seconds apart* for work that takes a hundred minutes — which is
entry #14 on this list, being useful about eight hours after it was written
down. Nothing else would have noticed. The cards were simply absent, and absent
cards look identical to cards not yet written.

Now `scripts/deploy_to_pod.sh` refuses to deploy onto a pod with running jobs,
checksums the transfer, verifies a staged copy imports before touching the live
tree, and verifies the live tree again afterwards — the swap is a copy, and the
copy is the step that failed. Every queue script also opens with a preflight
that aborts loudly on a zero-byte source or a failed import.

**27. An unpinned dependency silently disabled every model.**
A fresh pod was provisioned with `pip install transformers`, unpinned. It
resolved to 5.15.1, which requires PyTorch ≥ 2.5 against an image shipping 2.4.1.
Transformers did not fail — it **disabled its torch backend** and replaced every
model class with a placeholder that raises only when a model is actually loaded.
Seventy queued runs each failed in about eight seconds, writing nothing.

Two things made it survivable and both were luck rather than design. It printed a
warning naming the exact cause — hidden by the run script's `| tail -2`, which is
entry #14 on this list, hit for the third time in one day. And the eight-second
iterations gave it away, which is #26's signature.

The fix is not "pin transformers". It is that **the base image's torch is the
fixed point** and everything resolves against it; `requirements-pod.txt` records
the set that actually works. Preflights now assert `is_torch_available()` and
that the model class is not a stub, because **"it imports" is not "it works"** —
that gap is what let a broken environment start a five-seed sweep.

**28. A gap of exactly zero because no probe layer could be chosen.**
Where a split left too little to carve a validation set from, the probe layer fell
back to the sanctioned surface, P equalled B identically, and the card reported a
gap of +0.0000 with no uncertainty. Averaging a seed of constructed zeros against a
seed of real gaps (up to +0.204 on the same task) describes the instrument failing
to choose a layer. Excluded and counted now.

**29. Untrained control arms were regressed as if they were models.**
Four randomly-initialised ESM2 checkpoints sat inside the ladder on both
transcription-factor tasks, at identical parameter counts to their trained twins,
silently doubling four of five rungs. They exist precisely to test whether a curve
is about learned biology or parameter count, so putting them *on* the curve inverts
their purpose. Found by an adversarial check on the paper draft, after the affected
results had been written up. Correcting it changed no verdict — which is luck, not
vindication, and the reason `ModelRef` now carries an explicit `control_arm` flag
rather than relying on a name suffix.

A pattern in all seven: each was found by *changing something else and noticing
the result did not move as it should have* — or, in #26 and #27, moved far too fast.
None would have been caught by re-reading the code that contained it.

---

## What we would tell someone starting

1. **Assume your errors are not random.** Ours were 22 for 22 in our favour.
   Budget review time by which direction a bug would push the answer, not by how
   likely it seems.
2. **Every control must be read by the code that consumes the result.** Writing
   a p-value onto a card and never checking it is worse than not computing it,
   because it looks like a control.
3. **Checks must be positive assertions.** Not "a file exists" but "it parses
   and carries a score". Every silent failure here passed a count.
4. **Re-run failures; never route around them.** What crashes is not a random
   sample of your ladder.
5. **Pre-register when a result is close.** Ours moved from positive to null to
   positive across 1, 3 and 6 seeds. The rule for what to do with each outcome
   has to be fixed before you know which one you have.
6. **A comment describing a control is not a control.** We shipped one that
   described the exact bug in the code beneath it, and a cache whose docstring
   claimed the exact property it lacked.
7. **Round numbers are suspects.** A ratio of exactly 1.00 or a slope of
   exactly 0.000 almost always means the instrument, not the world.
8. **Expect to keep finding them.** Three more surfaced within a day of writing
   this list, and all three were caught by changing something adjacent and
   noticing the answer failed to move — not by re-reading the code.
