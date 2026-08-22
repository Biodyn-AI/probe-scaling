# Supplementary Note S1 — detailed account of generative AI use

The ISCB acceptable-use policy for large language models, which *Briefings in
Bioinformatics* adopts, asks that any acceptable use of an LLM to produce or help
produce part of a paper be "explicitly declared and documented with sufficient
details in the supplementary materials." This note is that documentation.

**Tool.** Claude (Anthropic), used interactively through a coding assistant
interface, under the author's direction, over the course of the study.

## Where it was used, and what was checked

### 1. Code (policy: "assist in code writing"; "create documentation for code")

The assistant helped write and refactor the evaluation harness, the analysis
functions in `probe_scaling/analysis.py`, the card schema, the figure-generation
script and the test suite. All of this code is released in this repository.

*What makes this checkable:* `tests/test_paper_numbers.py` re-derives every
published value from the released result cards. If the analysis code were wrong,
the numbers in the paper and the numbers the code produces would disagree, and the
suite would fail. The author is responsible for the correctness of the code, as
the policy requires.

### 2. Finding defects (policy: "as an evaluation technique … to assist in finding inconsistencies or other anomalies")

The assistant was used to search the pipeline and the manuscript for
inconsistencies. This is how a number of the defects in
`supplementary/failure-modes.md` were surfaced — among them a scaling slope fitted
over rungs that had failed their own permutation null, untrained control arms
sitting inside two ladders, and a confidence interval that covered 90% while
claiming 95%.

Two points of honesty about this. First, the assistant also *introduced* errors
that were caught the same way: the count of measurement defects stated in an
early abstract was a figure the taxonomy had already retracted, and a claimed
card count matched nothing in the data. Both were found by re-deriving the
numbers from the released cards and are recorded in the git history. Second, the
checking process itself produced false alarms — for instance a verification
script that flagged a correctly rounded confidence bound because it applied
round-half-even to a float. Every flag was resolved against the data before any
change was made.

### 3. Bibliography

Every reference was resolved programmatically against doi.org or the arXiv API by
`scripts/build_bib.py`, which raises rather than emit an entry whose identifier
does not resolve. Each resolved record was then checked against its claimed
author, title, year and venue, because an identifier can resolve and still point
at the wrong paper. Six entries initially cited as preprints were replaced by
their peer-reviewed versions after confirming an exact title match.

The characteristic failure mode of LLM-assisted writing is the fabricated
citation. The generator is released so that this claim can be checked rather than
taken on trust: re-running it reproduces `paper/refs.bib` from the network.

### 4. Text (policy: "help to produce, part of the text")

The assistant helped produce parts of the manuscript text and the LaTeX sources
of the tables and figures. The scope, structure, claims, interpretations and
conclusions were set by the author, who reviewed and edited the output and takes
full responsibility for the accuracy of the text.

## What it was not used for

It was not used to generate the paper from a prompt describing it. It was not
used to produce, select, or alter any measurement: every number in the paper comes
from the released result cards, which were produced by running models on data, and
each is re-derived by the test suite. It is not an author.
