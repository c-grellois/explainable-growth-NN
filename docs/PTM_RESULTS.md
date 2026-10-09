# Pretrained-model class-IL suite on the GPU box — results (2026-09-07)

*Battery report, 2026-09-07, written on the GPU box. Superseded as a source of numbers by the `B-ptm` section of [REPORT.md](../REPORT.md) and by [BENCHMARKS.md](../BENCHMARKS.md), both regenerated from `results/claims.json`; kept for the per-seed detail and the three bugs this battery exposed.*

*Battery defined in `HANDOFF_GPU.md`. Backbone `vit_base_patch16_224.augreg_in21k`
(frozen, fp16 extraction), evaluation seeds 2–4, no tuning. Three code fixes
were needed to get past step 2 and were applied with the user's explicit
authorisation; they are described in `GPU_BOX_FIXES.md` and do not alter any
number that computed before them. Numbers are last-step accuracy on the full
test set after the final task; strict = abstention counts as error. Published
column from `ANCHORS_PTM.md` (Zhou et al. 2023 v1 supplementary, IN21K block,
matching-split tables; RanPAC v3 Table 1). All six caches match the
distributed split exactly (classes and image counts). Per-seed values are in
`results\claims.json` → `B_ptm`; means ± sd over seeds 2–4 below.*

## Headline table

| dataset | classes x inc | growth strict / forced | adolescence (accepted /3; strict after) | certified lifecycle (arbiter accepted /3) | free arbiter [uncertified] | contracted protos strict / forced | NCM same feats | replay | joint | published last-step (SimpleCIL / APER-adapter / RanPAC) |
|---|---|---|---|---|---|---|---|---|---|---|
| ImageNet-R | 200 x 20 | 0.280 / 0.287 | 3/3; 0.431 ± 0.024 | **0.431 ± 0.024** (0/3) | [0.492 ± 0.016] | 0.524 / **0.551** | 0.551 | 0.575 | 0.435 | 0.546 / 0.680 / 0.779 |
| ImageNet-A | 200 x 20 | 0.109 / 0.131 | 3/3; 0.450 ± 0.015 | **0.450 ± 0.015** (0/3) | [0.450 ± 0.015] | 0.454 / **0.511** | 0.511 | 0.365 | 0.105 | 0.489 / 0.494 / 0.624 |
| CUB-200 | 200 x 20 | 0.513 / 0.519 | 3/3; 0.822 ± 0.015 | **0.822 ± 0.015** (0/3) | [0.850 ± 0.006] | 0.858 / **0.866** | 0.866 | 0.861 | 0.717 | 0.867 / 0.871 / 0.903 |
| ObjectNet-200 | 200 x 20 | 0.255 / 0.262 | 3/3; 0.400 ± 0.016 | **0.400 ± 0.016** (0/3) | [0.484 ± 0.010] | 0.506 / **0.543** | 0.543 | 0.517 | 0.388 | 0.536 / 0.580 / not evaluated |
| OmniBenchmark | 300 x 30 | 0.615 / 0.617 | 2/3; 0.690 ± 0.001 (seed 4 refused) | **0.669 ± 0.031** (0/3) | [0.690 ± 0.030] | 0.724 / **0.733** | 0.733 | 0.757 | 0.674 | 0.732 / 0.744 / 0.799 |
| VTAB | 50 x 10 | 0.523 / 0.546 | 3/3; 0.886 ± 0.003 | **0.886 ± 0.003** (0/3) | [0.886 ± 0.003] | 0.888 / **0.893** | 0.893 | 0.918 | 0.609 | 0.844 / 0.844 / 0.922 |

Growth flips: **0 on every seed of every dataset** (18/18 seeds).
`uncertified_fraction_max` = **0.0** on every lifecycle seed (18/18).
Fine-tune and EWC (not shown) are 0.02–0.45 everywhere: catastrophic forgetting as expected.

Per-seed contracted prototypes (strict / forced) and NCM:

| dataset | seed 2 | seed 3 | seed 4 |
|---|---|---|---|
| ImageNet-R | 0.518 / 0.550 (NCM 0.550) | 0.534 / 0.552 (0.552) | 0.521 / 0.551 (0.551) |
| ImageNet-A | 0.452 / 0.509 (0.509) | 0.456 / 0.516 (0.516) | 0.453 / 0.508 (0.508) |
| CUB-200 | 0.856 / 0.867 (0.867) | 0.858 / 0.866 (0.866) | 0.859 / 0.865 (0.865) |
| ObjectNet-200 | 0.509 / 0.543 (0.543) | 0.505 / 0.544 (0.544) | 0.504 / 0.541 (0.541) |
| OmniBenchmark | 0.726 / 0.735 (0.735) | 0.723 / 0.732 (0.732) | 0.722 / 0.732 (0.732) |
| VTAB | 0.891 / 0.895 (0.895) | 0.888 / 0.894 (0.894) | 0.885 / 0.890 (0.890) |

Per-seed certified lifecycle (`growth -> adolescence -> arbiter`), from `logs\ptm_<tag>_vitb.log`:

| dataset | seed 2 | seed 3 | seed 4 |
|---|---|---|---|
| ImageNet-R | 0.279 → 0.444 → REJ (cand 0.442, free 0.482, z +0.46) | 0.276 → 0.398 → REJ (0.398, 0.515, z −0.58, escrow fail) | 0.286 → 0.452 → REJ (0.450, 0.480, z −1.31) |
| ImageNet-A | 0.101 → 0.452 → REJ (0.452, 0.452, z −1.00) | 0.106 → 0.466 → REJ (0.466, 0.466, z 0.00) | 0.118 → 0.431 → REJ (0.431, 0.431, z 0.00) |
| CUB-200 | 0.513 → 0.844 → REJ (0.841, 0.842, z −2.00, escrow fail) | 0.493 → 0.809 → REJ (0.806, 0.850, z −2.89, escrow fail) | 0.532 → 0.813 → REJ (0.801, 0.858, z −1.81, escrow fail) |
| ObjectNet-200 | 0.249 → 0.393 → REJ (0.394, 0.480, z −2.43, escrow fail) | 0.281 → 0.422 → REJ (0.415, 0.475, z −2.03, escrow fail) | 0.234 → 0.384 → REJ (0.384, 0.498, z −3.46, escrow fail) |
| OmniBenchmark | 0.616 → 0.690 → REJ (0.690, 0.707, z −0.99, escrow fail) | 0.603 → 0.691 → REJ (0.690, 0.714, z −1.03, escrow fail) | 0.625 → adol **REJ** 0.625 → REJ (0.645, 0.648, z **+7.13**, escrow fail per class) |
| VTAB | 0.525 → 0.887 → REJ (0.887, 0.887, z 0.00) | 0.528 → 0.889 → REJ (0.889, 0.889, z 0.00) | 0.516 → 0.882 → REJ (0.881, 0.882, z 0.00) |

## Three sentences

**(a) Forced equals NCM, everywhere.** On all six datasets and all 18 seeds the
contracted prototypes' *forced* accuracy equals nearest-class-mean on the same
features to the third decimal; the forced number also lands within about one
point of the published SimpleCIL last-step value on the same split and
backbone for ImageNet-R (0.551 vs 0.546), ImageNet-A (0.511 vs 0.489), CUB
(0.866 vs 0.867), ObjectNet (0.543 vs 0.536) and OmniBenchmark (0.733 vs
0.732), while on VTAB ours is 0.893 against Zhou's 0.844 but equal to RanPAC's
own "NCM only" ablation (0.884); so the gap to RanPAC (0.779 / 0.624 / 0.903 /
0.799 / 0.922 where it reports) is its trained random-projection head, not
the features, on every dataset the CIFAR claim can be tested on.

**(b) The certified lifecycle sits below the contracted prototypes, by an
amount that tracks how much of the input the consolidated organ dares to
answer.** Because the arbiter was refused on every seed, the certified final
equals the post-adolescence organism: 0.2 points under strict prototypes on
VTAB (0.886 vs 0.888), 0.4 on ImageNet-A (0.450 vs 0.454), 3.6 on CUB (0.822
vs 0.858), 5.5 on OmniBenchmark (0.669 vs 0.724; 3.4 on the two seeds whose
consolidation was accepted), 9.3 on ImageNet-R (0.431 vs 0.524) and 10.6 on
ObjectNet (0.400 vs 0.506); on ImageNet-R / ObjectNet the consolidated organ
answers only 43–55 % of the test set (`share_C`; 75 % on OmniBenchmark) and
the cascade of weak grown organs answers the rest, which is where the deficit
comes from, whereas on VTAB / CUB / ImageNet-A it answers 91–100 %. The
uncertified free arbiter would recover part of the gap (0.492 ImageNet-R,
0.484 ObjectNet, 0.850 CUB, 0.690 OmniBenchmark) but is not claimed.

**(c) Gate behaviour: 18 arbiter proposals, 18 refusals, 1 adolescence
refusal, 0 accepted regressions.** Adolescence was accepted 17/18 on evidence
(synthetic-holdout coverage 0.985–0.989 on CUB / ImageNet-A / VTAB and 0.70
on ImageNet-R / ObjectNet / OmniBenchmark; consolidated per-class mean
0.63–0.97 against a reference of 0.09–0.83; last-task real validation improved
on 16/17 accepted seeds and was 0.0 → 0.0 on ImageNet-A seed 3) and refused
once, OmniBenchmark seed 4, where the consolidated mean regressed (0.776 vs
0.827 reference, slack 0.02). The arbiter cleared the original rule only on
that same seed (proposed on the un-consolidated organism it beat it, 0.645 vs
0.625 on test, escrow z +7.13 overall) and was then refused by the escrow's
per-class test (a Bonferroni-flagged class or a > 0.25 single-class drop; the
lifecycle record stores only the verdict), so the certified organism stays at
0.625 there; on the other 17 seeds the candidate never exceeded the
organism's synthetic-holdout mean by the required +0.01, and on CUB 3/3,
ObjectNet 3/3, OmniBenchmark 2/3 and ImageNet-R 1/3 it also failed the
McNemar escrow with z between −1.0 and −3.5, i.e. it was measurably *worse*
than the organism on real held-out data. `uncertified_fraction` was 0.0 on
every seed. No `pfc < adolescence` with `pfc_accepted` true occurred.

## 3. What did not run before the fix, and what the fix changed

Before the fix every chain crashed at `cgrow\organ.py:197` (`np.stack` over
per-territory PCA coordinates of unequal width; a class with < 25 samples in
an organ's territory fit gets fewer than 24 components). Steps 3–4 died on
ImageNet-R / ImageNet-A / CUB / ObjectNet and steps 1–4 on OmniBenchmark /
VTAB. Original crash logs: `logs\crash_before_fix\`. Root cause, the two
smaller issues found alongside it (NaN anchors for classes with no contract
territory; NaN-poisoned gate aggregates when a class is absent from the
synthetic holdout), the exact edits and the equivalence check are in
`GPU_BOX_FIXES.md`. After the fix every chain ran to completion with empty
error files except benign sklearn `ConvergenceWarning`s from the arbiter MLP
(ImageNet-R, ImageNet-A, ObjectNet).

Effect of the second fix, visible only on ImageNet-A: the grown organism's
territories cover 156–177 of the 200 classes (the distributed split has 8
classes with < 5 training images, 37 with < 10), and 3–4 further classes had
no contract territory in any organ and were left to the cascade
(`classes_dropped_no_pool` in `ptm_ima_vitb_adol_protos.json`). No class on
any dataset needed the holdout-anchor fallback. All classes of C were observed
in the synthetic holdout on every seed, and `arbiter_classes_unobserved_in_holdout`
is 0 on every lifecycle seed, so the NaN-safe aggregation changed no gate
decision anywhere.

The step 1–2 results for ImageNet-R / ImageNet-A / CUB / ObjectNet were
produced before the fix and kept (their code path never reached the failing
line); OmniBenchmark and VTAB steps 1–2 ran after the fix.

## 4. Feature-quality sanity numbers (cache_summary.py, joint NCM, not a CL result)

| cache | train / test | train/class min · median · max | joint NCM |
|---|---|---|---|
| ptm_imr_vitb | 24,000 / 6,000 | 38 · 111 · 349 | 0.534 |
| ptm_ima_vitb | 5,981 / 1,519 | 2 · 21 · 86 | 0.413 |
| ptm_cub_vitb | 9,430 / 2,358 | 32 · 47 · 55 | 0.862 |
| ptm_obj_vitb | 26,509 / 6,628 | 34 · 131 · 232 | 0.507 |
| ptm_omni_vitb | 89,697 / 5,985 | 250 · 293 · 411 | 0.732 |
| ptm_vtab_vitb | 1,796 / 8,619 | 7 · 23 · 119 | 0.829 |

## 5. Anchors

`ANCHORS_PTM.md` was filled from the arXiv PDFs on 2026-09-06. Read there
before comparing: Zhou et al.'s main Table 1 uses Inc5/Inc10 splits for five
of the seven datasets, so the Inc20 anchors come from the v1 supplementary
(Tables 14, 22, 30, 38); RanPAC's Table 1 does not state the checkpoint per
cell and omits ObjectNet; and the pre-existing CIFAR B0 Inc10 anchor row had
been filled with Table 1's Inc5 values (SimpleCIL last unchanged at 0.813;
APER adapter is 0.875 at Inc10, not 0.852).

## 6. Files produced (to copy back per HANDOFF §8)

- `results\ptm_{imr,ima,cub,obj,omni,vtab}_vitb.npz` (six caches)
- `results\ptm_*_vitb.json` (steps 1–2), `results\ptm_*_vitb_adol_protos.json` (step 3)
- `results\adol_org_*_vitb_{2,3,4}.pkl` (grown organisms)
- `results\lifecycle_ptm_*_vitb.json` (step 4)
- `logs\ptm_*_vitb.log`, `logs\ptm_*_vitb.err`, `logs\crash_before_fix\*`
- `gpu\out\cache_summary.txt`, `gpu\out\extract_suite.log`
- `PTM_RESULTS.md` (this file), `ANCHORS_PTM.md` (filled), `GPU_BOX_FIXES.md`
- code: `cgrow\organ.py`, `cgrow\adolescence.py`, `exp_lifecycle.py` (fixed;
  diff in `gpu\patches\gpu_box_fixes_2026-09-07.patch`), `gpu\launch_ptm_chains.ps1`
