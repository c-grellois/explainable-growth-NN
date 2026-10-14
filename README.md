# Contracted growth

**Continual learning in which retention is a checked invariant rather than a
score.**

A classifier is grown as a cascade of **frozen modules** ("organs"), each
sitting behind a finite, machine-checkable **contract**: class anchors, a
margin, a radius, and an input-space territory saying which inputs it may
answer. New capacity is added only where the system declines to answer, and
nothing already in place is modified. Adding a module therefore cannot change
an answer already given. That is verified bit for bit rather than estimated,
and it has held in every run on record, including streams cut into a hundred
separate updates. Every answer that is given comes from inside a satisfied
contract, and the system may decline. Every structural reorganisation, such as
consolidating the cascade or learning a router, must pass an acceptance test on
held-out data or it is refused, with the reason recorded.

The project unifies two earlier projects: *ASDbrain* (sparse developmental
growth; training, not pruning, is what destroys retention) and *NNcontracts*
(finite contracts at network interfaces make modules replaceable).
[DESIGN.md](DESIGN.md) explains the combination.

## Results at a glance

The standard exemplar-free class-incremental protocol for pre-trained models:
a frozen ViT-B/16 (ImageNet-21k) encoder, classes revealed in equal steps,
nothing stored between steps. Last-step accuracy on the full test set (%),
mean of evaluation seeds 2–4, a declined input counted as an error.

| dataset | classes / steps | SimpleCIL | best APER | RanPAC | **accuracy config** | stable config |
|---|---|---|---|---|---|---|
| CIFAR-100 | 100 / 10 | 81.3 | 87.5 | 92.2 | **90.6** | 86.7 |
| ImageNet-R | 200 / 10 | 54.6 | 69.5 | 77.9 | **76.3** | 69.5 |
| ImageNet-A | 200 / 10 | 48.9 | 50.0 | 62.4 | **55.9** | 51.6 |
| CUB-200 | 200 / 10 | 86.7 | 87.1 | 90.3 | **89.1** | 85.4 |
| ObjectNet-200 | 200 / 10 | 53.6 | 58.2 | — | **63.9** | 53.6 |
| OmniBenchmark | 300 / 10 | 73.2 | 74.5 | 79.9 | **76.6** | 71.1 |
| VTAB | 50 / 5 | 84.4 | 85.4 | 92.2 | **93.9** | 88.2 |

Both configurations use a one-off adapter step on the first task, then freeze
the encoder. The **accuracy configuration** is a random-projection ridge
readout under the same contract, the lever RanPAC uses. It re-solves that
readout at every step, so over a run it changes between 7 and 105 answers it
had already given per 100 test points (an answer can change more than once).
The **stable configuration** is contracted prototypes, which change none. Accuracy and bitwise stability are two configurations, not one:
the obvious way to get both was tried and loses 20 to 54 points, for a reason
the record states (`B-head-blocks` in [REPORT.md](REPORT.md)).

How to read the comparison. Published numbers were read from the source PDFs,
with table, metric and protocol recorded per cell in
[ANCHORS_PTM.md](ANCHORS_PTM.md). SimpleCIL and APER (Zhou et al. 2023,
arXiv:2303.07338v1, supplementary tables) report last-step accuracy on the same
encoder; "best APER" is their best variant per dataset. RanPAC (McDonnell et al.
2023, Table 1) reports the mean of per-task accuracies after the last task,
does not say which ViT-B/16 checkpoint each cell used, and does not evaluate
ObjectNet. On the ImageNet-1K checkpoint its columns point to, the accuracy
configuration's forced accuracy (every input answered) lands within 0.2, 1.7
and 0.8 points of RanPAC on ImageNet-R, ImageNet-A and CUB
([BENCHMARKS.md](BENCHMARKS.md)).

Beyond that table:

- **Zero answers changed by growth, in every run** on every substrate tried:
  raw pixels, frozen vision encoders, a frozen sentence encoder, raw tabular
  rows, and a recurrent spatial sheet. The uncertified-answer fraction is
  0.0000 in all 45 evaluation runs of the certified configuration.
- **Nothing raw is stored.** On split-MNIST the certified system reaches 0.893
  storing nothing, against 0.871 for replay trained on the same data with 100
  raw images per class.
- **The gate is audited, misses included.** Over the 102 gated runs on record
  it refused all four known-harmful changes and no clearly helpful one. It let
  through two regressions smaller than its own stated resolution and one larger
  (2.3 points, under a permuted class order). REPORT.md has the full record.
- **What it does not do.** Nothing here beats a jointly trained model that sees
  all the data at once. The guarantees ride on the quality of a frozen encoder.
  More organs do not mean more accuracy.

Every number here traces to a per-seed record. Negative results and withdrawn
claims are kept in the record as results.

## Where to start

| file | what it is |
|---|---|
| [APPROACH.md](APPROACH.md) | the whole approach twice over: plain English, then the details |
| [overview/contracted_growth_overview.pdf](overview/contracted_growth_overview.pdf) | three pages: the guarantees and the scores, no mechanisms |
| [explainer/anatomy.html](explainer/anatomy.html) | an interactive walk through one organism's life (download and open locally) |
| [BENCHMARKS.md](BENCHMARKS.md), [LITERATURE_COMPARISON.md](LITERATURE_COMPARISON.md) | every comparison table against published methods and same-data baselines |
| [REPORT.md](REPORT.md) | the scientific record, one section per battery, negative results included |
| [ANCHORS_PTM.md](ANCHORS_PTM.md) | every published anchor with its source table, metric, protocol and verification date |
| [DESIGN.md](DESIGN.md) | the founding design: why the two parent projects combine |
| [TECHNICAL.md](TECHNICAL.md) | how the code works: every module, every experiment script, the invariants |
| [paper/](paper/README.md) | the draft paper, *Forgetting Is a Certificate* (unpublished, anonymised) |
| [docs/](docs/README.md) | the operational trail: battery protocols, the reports that came back, retractions |

## Repository layout

```
cgrow/          the library: organs, contracts, territories, the cascade, gates, consolidation
exp_*.py        the experiments, one script per family (listed in TECHNICAL.md §3)
claims.py       results/*.json -> results/claims.json, the single source of every reported number
manifest.py     content hashes of every result file -> results/MANIFEST.json
gen_overview.py claims.json -> overview/ (LaTeX, and the PDF if pdflatex is installed)
trace_anatomy.py, make_figures.py      the explainer and the paper figures
extract_cifar_*.py, build_fsdd.py      local feature and data caches
summarize_*.py  read-only tables for the market experiments
results/        per-seed JSON records (tracked); feature caches and organisms (not tracked)
results_dev/    development-seed records (seeds 0-1)
gpu/            feature extraction on a CUDA machine, and the launch scripts
paper/  overview/  explainer/  docs/
```

## Running

Python 3.11 or later, then:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt     # Windows: .venv\Scripts\pip
```

A CPU is enough for everything except feature extraction. `cgrow` refuses to
import under a conda *base* interpreter, because a silently wrong interpreter
once cost two batteries of results; any virtualenv or conda environment works,
and `CGROW_ALLOW_ANY_PYTHON=1` overrides the check.

The quickest end-to-end run is the split-digits class-incremental stream: the
grown organism against fine-tuning, EWC, replay and a joint bound, on the three
evaluation seeds. It needs no download and takes about a minute on one CPU.

```bash
python exp_continual.py --dataset digits --first-seed 2 --seeds 3
```

It rewrites `results/continual_digits.json`. On the analysis machine's package
versions the rewrite is byte-identical, so `git status` stays clean; elsewhere
`git diff` shows exactly how far your platform's numerics moved the record.
Split-MNIST and the other OpenML streams download once and are cached under
scikit-learn's data home (`cgrow_cache/`).

The pretrained-encoder experiments read frozen feature caches
(`results/*.npz`, about 8 GB, not tracked). [gpu/README.md](gpu/README.md)
rebuilds them; the organisms they grow are pickled next to them and rebuilt on
first use. The spatial-sheet benchmarks (`exp_sheet.py`, `build_fsdd.py`) also
need ASDbrain's stream generators (its `growth_by_need` package), which are not
part of this repository: check ASDbrain out next to this one or point
`ASDBRAIN_DIR` at it. Nothing else depends on it.

Every reported number comes from the tracked records:

```bash
python claims.py        # results/*.json  -> results/claims.json
python manifest.py      # content hashes  -> results/MANIFEST.json
python gen_overview.py  # claims.json     -> overview/
```

## Discipline

Development seeds 0–1, evaluation seeds 2–4, gate-confirmation seeds 5–7, kept
disjoint. Thresholds and configurations are frozen before an evaluation seed is
run, and preregistrations are recorded in REPORT.md before the runs they
govern. Published anchors are read from the source PDFs with the table number
and the metric recorded, never quoted from memory. Every number in the
documents is quoted from `results/claims.json`, which `claims.py` regenerates
from the per-seed records, and `manifest.py` hashes the files behind them.
Refusals, sub-resolution slips,
negative results and withdrawn claims are reported as results; see "What
failed, honestly" in [APPROACH.md](APPROACH.md).

## Status

A research prototype, and a record of the research behind it. The paper draft
in `paper/` is unpublished. Results were produced between August and October
2026 on a CPU analysis machine, with feature extraction on a second machine with
one RTX 3070.

The work was carried out with AI assistance (Anthropic's Claude, through Claude
Code), including the runs on the GPU machine, which a separate assistant session
executed from the instruction documents kept in [docs/](docs/README.md). Leo
Dobson contributed the performance work on growth and inference.

**License.** GNU GPL v3.0
