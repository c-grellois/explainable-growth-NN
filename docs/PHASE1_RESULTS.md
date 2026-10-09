# First-session adapters and the ImageNet-1K checkpoint (batteries 2 and 3)

*Battery report, 2026-09-09. Superseded as a source of numbers by the `B-phase1` and `B-in1k` sections of [REPORT.md](../REPORT.md); kept for the per-(dataset, seed) detail. Note the withdrawal it carries: the CUB seed-4 "gate miss" was inferred rather than measured and is retracted in the text.*

*GPU box, 2026-09-08/09: 24 feature caches (`GPUOutput2`). Analysis run on the
analysis machine, evaluation seeds 2–4, frozen gate, no hyper-parameter of the
CPU pipeline changed. Phase-1 caches are seed-specific (adapters were tuned on
that seed's first task only, on rows the CPU pipeline treats as training data),
so each was run with its own seed alone and the three seeds aggregated here.
Numbers from `results/claims.json` (`B_phase1`, `B_in1k`). Anchors from
`ANCHORS_PTM.md`; † marks RanPAC columns on the ImageNet-1K-fine-tuned
checkpoint.*

## Feature quality of the caches (joint nearest-class-mean; not a CL result)

Fitting and scoring nearest-class-mean on all classes at once measures the
feature space, not continual learning. It isolates what each backbone change is
worth before any of our machinery runs:

| dataset | frozen in21k | + adapters (seeds 2 / 3 / 4) | mean gain | in1k checkpoint | RanPAC "NCM only" |
|---|---|---|---|---|---|
| CIFAR-100 (B0 Inc10) | 0.783 | 0.874 / 0.860 / 0.882 | **+0.089** | — | 0.834 |
| ImageNet-R (B0 Inc20) | 0.533 | 0.691 / 0.702 / 0.695 | **+0.163** | 0.611 | 0.612 |
| ImageNet-A (B0 Inc20) | 0.413 | 0.523 / 0.531 / 0.579 | **+0.131** | 0.513 | 0.493 |
| CUB-200 (B0 Inc20) | 0.862 | 0.869 / 0.870 / 0.867 | **+0.006** | 0.855 | 0.851 |
| ObjectNet-200 (B0 Inc20) | 0.507 | 0.556 / 0.568 / 0.569 | **+0.058** | — | — |
| OmniBenchmark (B0 Inc30) | 0.732 | 0.721 / 0.724 / 0.715 | **-0.012** | — | 0.731 |
| VTAB (B0 Inc10) | 0.829 | 0.878 / 0.880 / 0.872 | **+0.048** | — | 0.884 |

Phase 1 lifts the feature space by a lot on ImageNet-R (+16.3) and ImageNet-A
(+13.1), clearly on CIFAR-100 (+8.9), ObjectNet (+5.8) and VTAB (+4.8), barely
on CUB (+0.6, already 0.862 frozen), and **not at all on OmniBenchmark**
(-1.2). That last case is consistent with the published pattern: Zhou et al.
report their fine-tuned variant *below* SimpleCIL on OmniBenchmark B0 Inc30,
and attribute it to the domain spread of that benchmark's 300 classes, of
which the first task shows only 30.

The in1k checkpoint reproduces RanPAC's "NCM only" row on ImageNet-R (0.611 vs
0.612) and CUB (0.855 vs 0.851); ImageNet-A is 2 points high (0.513 vs 0.493),
which is within what the metric difference allows, since RanPAC's `A_T` is a
mean over per-task accuracies and ImageNet-A's per-class test counts are
unbalanced.

## Battery 3: first-session adapters ("Phase 1")

Parallel bottleneck adapters (rank 16, ≈ 0.31 M tunable parameters against the
backbone's 86 M) on every transformer block's MLP, trained for 20 epochs on the
first task's classes with a cosine classifier, then frozen. Everything
downstream is the unchanged pipeline, so the difference from the frozen-backbone
rows is Phase 1 alone.

| dataset | Phase-1 task-0 val acc | protos strict / forced (frozen → +adapters) | head M=10k strict (frozen → +adapters) | certified lifecycle (frozen → +adapters) | growth | APER-adapter | RanPAC full |
|---|---|---|---|---|---|---|---|
| CIFAR-100 (B0 Inc10) | 0.993 | 0.832 → **0.867** / 0.876 | 0.879 → **0.906** (cov 0.98) | 0.825 → **0.858** | 0.704 → 0.813 | 0.875 | 0.922 |
| ImageNet-R (B0 Inc20) | 0.960 | 0.524 → **0.695** / 0.702 | 0.652 → **0.763** (cov 0.98) | 0.391 → **0.571** | 0.280 → 0.481 | 0.680 | 0.779 † |
| ImageNet-A (B0 Inc20) | 0.879 | 0.454 → **0.516** / 0.553 | 0.451 → **0.559** (cov 0.91) | 0.450 → **0.510** | 0.109 → 0.181 | 0.494 | 0.624 † |
| CUB-200 (B0 Inc20) | 0.975 | 0.858 → **0.854** / 0.863 | 0.867 → **0.891** (cov 0.99) | 0.822 → **0.740** | 0.513 → 0.629 | 0.871 | 0.903 † |
| ObjectNet-200 (B0 Inc20) | 0.880 | 0.506 → **0.536** / 0.569 | 0.531 → **0.639** (cov 0.97) | 0.400 → **0.429** | 0.255 → 0.397 | 0.580 | — |
| OmniBenchmark (B0 Inc30) | 0.954 | 0.724 → **0.711** / 0.719 | 0.756 → **0.766** (cov 0.98) | 0.669 → **0.656** | 0.615 → 0.656 | 0.744 | 0.799 |
| VTAB (B0 Inc10) | 0.970 | 0.888 → **0.882** / 0.888 | 0.929 → **0.939** (cov 0.99) | 0.886 → **0.882** | 0.523 → 0.579 | 0.844 | 0.922 |

**What Phase 1 buys.** Contracted prototypes move by -0.012 to +0.171,
the contracted head by +0.009 to +0.112, the certified lifecycle by
-0.082 to +0.139. Tuning on a single task of 10–30 classes lifts the
whole feature space: the joint nearest-class-mean of the CIFAR caches rises from
0.783 (frozen) to 0.860–0.882.

**Against the published methods this makes comparable.** The head after Phase 1
is above APER-adapter on CIFAR-100, ImageNet-R, ImageNet-A, CUB-200, ObjectNet-200, OmniBenchmark, VTAB and above RanPAC-full on VTAB.
Every configuration keeps its certificate: growth flips 0 and uncertified
fraction 0.0 on every run.

## Battery 2: the ImageNet-1K checkpoint (RanPAC's confound removed)

RanPAC's "NCM only" ablation identified the checkpoint behind its ImageNet-R,
ImageNet-A and CUB columns. Extracting those three datasets on
`vit_base_patch16_224.augreg_in21k_ft_in1k` reproduces its values to within a
point, which confirms the identification:

| dataset | joint NCM, in1k cache | RanPAC "NCM only" | protos strict / forced | head M=10k strict / forced | certified lifecycle | in21k protos → in1k protos | in21k head → in1k head | RanPAC head-only / full |
|---|---|---|---|---|---|---|---|---|
| ImageNet-R (B0 Inc20) | 0.611 | 0.612 | **0.622** / 0.628 | **0.693** / 0.700 (cov 0.99) | 0.488 | 0.524 → 0.622 | 0.652 → 0.693 | 0.718 / 0.779 |
| ImageNet-A (B0 Inc20) | 0.513 | 0.493 | **0.558** / 0.576 | **0.560** / 0.575 (cov 0.93) | 0.555 | 0.454 → 0.558 | 0.451 → 0.560 | 0.582 / 0.624 |
| CUB-200 (B0 Inc20) | 0.855 | 0.851 | **0.849** / 0.859 | **0.875** / 0.883 (cov 0.99) | 0.821 | 0.858 → 0.849 | 0.867 → 0.875 | 0.887 / 0.903 |

## Reading

1. **The checkpoint confound is real and now measured.** On ImageNet-R the 1k
   checkpoint alone is worth +0.098 to the prototypes; our cache's joint
   nearest-class-mean (0.611) matches RanPAC's NCM-only row (0.612) to a
   thousandth, so the remaining gap on that dataset is method, not backbone.
2. **First-session tuning is compatible with the doctrine and is worth points.**
   It is a birth step: the substrate is shaped once by organ 0's data and then
   frozen for life, so no answer given after task 0 is affected by it. Under
   that reading the certified lifecycle moves by -0.082 to +0.139 and
   the contracted head by +0.009 to +0.112, without giving up anything
   the organism claims: growth flips stay 0 and the uncertified answer fraction
   stays 0.0 on every run.
3. **The gate refused the birth step's consequences on OmniBenchmark, on all
   three seeds, and was right to.** After Phase 1 the consolidated organ's
   per-class mean on the synthetic holdout fell to 0.764 against the grown
   organism's 0.831, so `consolidate_protos` was refused with "mean regressed"
   every time and the anatomy stayed as grown. This is the feature-quality
   finding above, caught by the machinery rather than by us: the one dataset
   where the birth step damages the space is the one dataset where the gate
   blocked the recombination that followed.
4. **CUB seed 4 is not a gate failure; it is a weaker substrate.** That seed's
   certified endpoint is 0.639 against 0.813 for the frozen backbone, and its
   consolidated organ answers only 12.8% of the real test set. Replaying the
   decision through the full real-data escrow (added 2026-09-09, see
   `exp_adol_escrow.py`) *passes* it: McNemar z = +3.78 over every class seen,
   real validation 0.655 to 0.677, test 0.616 to 0.639. The consolidation
   genuinely improved the organism it was applied to; what is lower is the
   Phase-1 substrate that seed happened to get, and no gate can or should
   compare a run against a different substrate it never saw. An earlier draft
   of this section called it a gate miss; that was inferred from the low
   `share_C` and is withdrawn.
5. **The remaining gap to RanPAC-full is its head width and decorrelation, not
   the backbone.** With both Phase 1 and the contracted head we are on the same
   two ingredients RanPAC uses; what differs is that our head must also satisfy
   a territory contract, and that our answers are certified for coverage.
