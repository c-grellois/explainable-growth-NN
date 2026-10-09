# RanPAC versus the contracted organism

*Method-by-method comparison written 2026-09-07, when it was still unknown which backbone checkpoint RanPAC's published columns used. That question was closed on 2026-09-15 by running the ImageNet-1K checkpoint ourselves (`B-in1k-phase1` in [REPORT.md](../REPORT.md)): on ImageNet-R, ImageNet-A and CUB the comparison is now like-for-like and the residual difference is the metric definition. Read this page for the mechanism-by-mechanism account, and REPORT for the current numbers.*

*Written 2026-09-07 from `ANCHORS_PTM.md` (RanPAC v3 Table 1, Zhou et al. v1
supplementary) and `results/claims.json` (`B_cifar100_vitb`, `B_ptm`). Ours =
evaluation seeds 2–4, ViT-B/16-in21k frozen, no tuning.*

## What each method is

| | RanPAC (McDonnell et al., NeurIPS 2023) | contracted organism (this work) |
|---|---|---|
| backbone | frozen ViT-B/16, **after a "Phase 1"**: parameter-efficient tuning (adapter / prompts / SSF) on the *first* task only, then frozen | frozen ViT-B/16, never tuned |
| classifier | random projection of the 768-d feature to M = 10,000 dims, nonlinearity, then a ridge classifier solved in closed form from accumulated Gram (M × M) and class-projection (M × C) matrices; decorrelates class prototypes | nearest contracted prototype: per-class anchor + PCA territory + margin; or a cascade of frozen per-task organs with vetoes, consolidated at adolescence |
| what is stored per task | the running Gram and class-projection sums (order-invariant; no raw data) | one frozen organ (or one anchor + territory per class); no raw data |
| head size | ≈ 100M floats for the Gram matrix + 1M for the weights (CIFAR); ≈ 400 MB fp32 | ≈ 4M floats for the whole CIFAR organism (16 MB); 100 anchors + 100 territories for the consolidated prototype organ |
| forgetting | final weights are order-invariant (same solution whatever the task order), but the ridge solution is re-solved after every task, so old answers change as classes arrive | old organs are never edited; answers already given are bitwise identical after growth (checked every run) |
| abstention | none; always answers | contract-covered answers only; strict accuracy counts abstention as error |
| acceptance of changes | none needed (closed form) | every recombination (consolidation, arbiter, retirement, negotiation) passes a gate on real held-out data |
| training data touched | Phase 1 trains PETL parameters on task 0 with backprop | organs train tiny heads on their own task; contracts and negatives are generated, not stored |

## Numbers, same protocol (B0, 10 tasks; VTAB 5), last-step accuracy

RanPAC's own ablation rows decompose its result: **NCM only** (cosine
nearest-class-mean on the frozen backbone, no head, no Phase 1) → **No Phase 1**
(adds the random-projection ridge head) → **full** (adds first-session PETL).

| dataset | ours, contracted protos strict / **forced** | RanPAC NCM only | RanPAC No Phase 1 (RP head, frozen backbone) | RanPAC full | head alone buys | Phase 1 adds | full − our forced |
|---|---|---|---|---|---|---|---|
| CIFAR-100 | 0.835 / **0.844** | 0.834 | 0.890 | 0.922 | +5.6 | +3.2 | −7.8 |
| ImageNet-R | 0.524 / **0.551** | 0.612 † | 0.718 | 0.779 | +10.6 | +6.1 | −22.8 |
| ImageNet-A | 0.454 / **0.511** | 0.493 † | 0.582 | 0.624 | +8.9 | +4.2 | −11.3 |
| CUB-200 | 0.858 / **0.866** | 0.851 † | 0.887 | 0.903 | +3.6 | +1.6 | −3.7 |
| OmniBenchmark | 0.724 / **0.733** | 0.731 | 0.782 | 0.799 | +5.1 | +1.7 | −6.6 |
| VTAB | 0.888 / **0.893** | 0.884 | 0.922 | 0.922 | +3.8 | 0.0 | −2.9 |
| ObjectNet-200 | 0.506 / **0.543** | not evaluated by RanPAC | | | | | |

**Reading.** Our forced accuracy sits at RanPAC's *NCM-only* level on every
dataset where the two share a backbone (CIFAR −0.1 … +1.0, OmniBenchmark +0.2,
VTAB +0.9): the contract machinery costs nothing in forced accuracy and buys
abstention (strict trails forced by 0.5–5.7 points, the fraction it refuses to
answer). Everything RanPAC has beyond that comes from two additions we do not
have: the random-projection ridge head (+3.6 to +10.6, on frozen features) and
first-session PETL (+0 to +6.1). Neither is a better backbone; both are
classifier-side.

**† Backbone audit.** RanPAC's Table 1 does not state the ViT-B/16 checkpoint per
column. Its NCM-only row reveals it: on ImageNet-R (61.2), ImageNet-A (49.3)
and CUB (85.1) the NCM-only value matches Zhou et al.'s SimpleCIL **IN1K**
block (61.28 / 49.44 / 85.16), not the IN21K block we and Zhou's main results
use (54.55 / 48.91 / 86.73). On OmniBenchmark it matches the IN21K value exactly
(73.15 vs 73.1); on CIFAR and VTAB it sits within a point of *our* IN21K
nearest-class-mean (83.4 vs 84.4; 88.4 vs 89.3), so those three columns are
consistent with IN21K. So the ImageNet-R, ImageNet-A and CUB RanPAC anchors are on the
**ImageNet-1K-fine-tuned** checkpoint, which alone is worth +6.7 on ImageNet-R
and +0.5 on ImageNet-A at the NCM level (and −1.6 on CUB). Of our −22.8 on
ImageNet-R, roughly 7 points is the checkpoint and 16 the head plus Phase 1.
Extracting `vit_base_patch16_224.augreg_in21k_ft_in1k` caches on the GPU box
would remove that confound for three datasets.

## What follows

1. **The lever is a contracted head.** A random-projection ridge readout whose
   output carries a ball contract (radius, margin, territory in the projected
   space) is the one change that could move the certified number by the
   +3.6…+10.6 RanPAC's "No Phase 1" row shows is available on frozen features.
   Nothing about it conflicts with the doctrine: the head is solved in closed
   form from accumulated statistics and never re-trained by gradient.
2. **Phase 1 is compatible too, read correctly.** PETL on the first task then
   frozen is exactly "organ 0 also shapes the substrate, once". It is a birth
   step, not a lifetime of retraining. It is the only place either method
   touches the backbone.
3. **What RanPAC cannot say and we can.** Which answers are covered by a
   contract (all of ours), that no past answer changed under growth (bitwise),
   that a proposed change was accepted only after a real-data test, and what
   the system will abstain on. RanPAC's head is re-solved after every task and
   always answers; its per-task accuracies R_{T,i} are the only retention
   evidence it offers.
4. **Cost.** RanPAC's Gram matrix is 25× the size of our whole organism; the
   organism grows linearly in tasks, RanPAC's head is constant. Both are
   exemplar-free.

## Follow-up (2026-09-08): the contracted head, measured

`exp_head.py` implements the lever named above. On the same in21k features,
seeds 2–4, random order:

| dataset (seeds 2–4) | contracted protos strict / forced | **head M=2000** strict / forced | **head M=10000** strict / forced | coverage (M=10k) | answer-change events under growth, per 100 test points (M=10k) | RanPAC: NCM only / head only / full |
|---|---|---|---|---|---|---|
| CIFAR-100, ViT-B/16-in21k | 0.835 / 0.843 | 0.869 / 0.878 | 0.875 / 0.884 | 0.98 | 15 | 0.834 / 0.890 / 0.922 |
| CIFAR-100, ViT-B full data | 0.832 / 0.842 | 0.870 / 0.879 | 0.879 / 0.888 | 0.98 | 11 | — |
| CIFAR-100, DINOv2-B | 0.865 / 0.873 | 0.890 / 0.898 | 0.901 / 0.909 | 0.98 | 8 | — |
| CIFAR-100, ViT-L/16 | 0.867 / 0.877 | 0.897 / 0.909 | 0.904 / 0.915 | 0.98 | 8 | — |
| ImageNet-R | 0.524 / 0.551 | 0.633 / 0.641 | 0.652 / 0.659 | 0.99 | 66 | 0.612 / 0.718 / 0.779 † |
| ImageNet-A | 0.454 / 0.511 | 0.423 / 0.480 | 0.451 / 0.498 | 0.74 | 155 | 0.493 / 0.582 / 0.624 † |
| CUB-200 | 0.858 / 0.866 | 0.857 / 0.867 | 0.867 / 0.876 | 0.99 | 14 | 0.851 / 0.887 / 0.903 † |
| ObjectNet-200 | 0.506 / 0.543 | 0.516 / 0.547 | 0.531 / 0.561 | 0.84 | 74 | — |
| OmniBenchmark | 0.724 / 0.733 | 0.734 / 0.745 | 0.756 / 0.766 | 0.97 | 19 | 0.731 / 0.782 / 0.799 |
| VTAB | 0.888 / 0.893 | 0.923 / 0.927 | 0.929 / 0.934 | 0.99 | 12 | 0.884 / 0.922 / 0.922 |

The head recovers most of RanPAC's head-only gain while every answer stays
inside a territory and above a margin; what it cannot give is the cascade's
bitwise retention (the ridge weights are re-solved after every task; already-given
answers change 8–155 times per 100 test points over a run). ImageNet-A, with 2–86
training images per class, is where a second-order head is worse than a class
mean and the contract shows it by refusing. Details: `REPORT.md`, B-head.
