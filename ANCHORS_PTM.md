# Published anchors for the pretrained-model class-IL suite — VERIFIED 2026-09-06

*Nothing on this page is quoted from memory. Every accuracy cell below was read
on 2026-09-06 from the text of the arXiv PDFs (extracted with pypdf on the GPU
box; the PDFs were fetched from arxiv.org the same day), following the same
rule as the 2026-08-31 citation audit: number, table, metric (last-step vs.
average incremental), backbone, protocol, and exemplar budget all recorded at
once. It was filled in before any `B_ptm` result was compared to it.*

Source versions read:

- Zhou, Sun, Ye, Zhan, Liu, "Revisiting Class-Incremental Learning with
  Pre-Trained Models: Generalizability and Adaptivity are All You Need",
  **arXiv:2303.07338v1 (13 Mar 2023)**, 38 pages incl. supplementary. Method
  named ADAM there (renamed Aper in v2, 5 Aug 2024; v2 main-text Table 1 is
  numerically identical to v1 Table 1 but v2 drops the per-split supplementary
  tables, so v1 is the version cited for the Inc20 rows below).
- McDonnell, Gong, Parvaneh, Abbasnejad, van den Hengel, "RanPAC: Random
  Projections and Pre-trained Models for Continual Learning",
  **arXiv:2307.02251v3 (16 Jan 2024)**, NeurIPS 2023.

## Protocols (what our runs use; matches the papers' B0 settings)

| dataset | our cache | classes | inc | tasks | backbone for the anchor row |
|---|---|---|---|---|---|
| CIFAR-100 | `cifar100_vitb.npz` (done) | 100 | 10 | 10 | ViT-B/16-in21k |
| ImageNet-R | `ptm_imr_vitb.npz` | 200 | 20 | 10 | ViT-B/16-in21k |
| ImageNet-A | `ptm_ima_vitb.npz` | 200 | 20 | 10 | ViT-B/16-in21k |
| CUB-200 | `ptm_cub_vitb.npz` | 200 | 20 | 10 | ViT-B/16-in21k |
| ObjectNet-200 | `ptm_obj_vitb.npz` | 200 | 20 | 10 | ViT-B/16-in21k |
| OmniBenchmark | `ptm_omni_vitb.npz` | 300 | 30 | 10 | ViT-B/16-in21k |
| VTAB | `ptm_vtab_vitb.npz` | 50 | 10 | 5 | ViT-B/16-in21k |

Our metric is always last-step accuracy on the full test set after the final
task, strict (abstention = error), evaluation seeds 2–4, class order shuffled
per seed. The papers report **last** (`A_B` / `A_last`) and **average
incremental** (`\bar A`) accuracy; only the *last* column is comparable to ours.

### Protocol audit finding (2026-09-06)

**Zhou et al.'s main-text Table 1 does not use our splits.** Its columns are
CIFAR B0 Inc5, CUB B0 Inc10, IN-R B0 Inc5, IN-A B0 Inc10, ObjNet B0 Inc10,
OmniBench B0 Inc30, VTAB B0 Inc10 (caption: "seven datasets with
ViT-B/16-IN21K as the backbone"). Only the OmniBenchmark and VTAB columns
match our protocol. The B0 Inc20 rows for CUB / ImageNet-R / ImageNet-A /
ObjectNet and the CIFAR B0 Inc10 row exist only in the **v1 supplementary**
(Appendix D, Tables 6–50, one table per split, each with a ViT-B/16-IN1K block
and a ViT-B/16-IN21K block; columns are average-incremental first, then
last). All numbers below come from the IN21K block of the matching-split
table. Note that SimpleCIL's *last* accuracy is split-invariant by
construction (frozen features, class means over all classes), so its last
value coincides between Table 1 and the Inc20 tables for every dataset except
ImageNet-A (Table 1 prints 49.44 for IN21K; the Inc20 table prints 48.91 for
IN21K and 49.44 for IN1K; we record the Inc20 IN21K value).

**RanPAC's Table 1 matches our protocol** (T = 10 equal stages; VTAB T = 5) but
the table does not say which of the two ViT-B/16 checkpoints each cell uses
(Section 5: "two ViT-B/16 models ... one self-supervised on ImageNet-21K, and
another with supervised fine-tuning on ImageNet-1K"; Appendix F.6: the better
backbone is dataset-dependent). Its metric is the final Average Accuracy
`A_T = (1/T) Σ_i R_{T,i}`, i.e. the mean of per-task accuracies after the last
task, which equals overall last-step accuracy only when tasks and per-class
test counts are balanced. RanPAC does not evaluate ObjectNet. Appendix F.10
says Tables 1–3 were updated after submission to match the code repository,
and that seed variability is about ±1 %.

**Correction to the 2026-08-31 CIFAR row.** The previously verified
`0.813 / 0.876` (SimpleCIL) and `0.852 / 0.907` (APER adapter) are the
**B0 Inc5** column of Table 1 (81.26 / 87.57 and 85.15 / 90.65). The B0 Inc10
values, from v1 Table 7 (IN21K block), are SimpleCIL 81.26 / 87.13 and
ADAM w/ Adapter 87.49 / 92.24. The SimpleCIL *last* value is unchanged
(split-invariant), so the +2.2 headline over SimpleCIL stands; the APER
adapter comparison moves from 0.852 to 0.875. The row below now carries the
Inc10 values; the old ones are kept in the note for the audit trail.

## Anchor table (last / avg, fractions; ViT-B/16-IN21K rows; no exemplars)

| dataset | SimpleCIL last / avg | APER (adapter) last / avg | best APER variant last / avg | RanPAC final | source table | verified on | by |
|---|---|---|---|---|---|---|---|
| CIFAR-100 B0 Inc10 | 0.813 / 0.871 ✔ | 0.875 / 0.922 ✔ (was 0.852 / 0.907 = Inc5, see note) | 0.875 / 0.922 (Adapter) | 0.922 ✔ | Zhou v1 Tab. 7 (IN21K block); McDonnell v3 Tab. 1 | 2026-09-06 (SimpleCIL last & RanPAC also 2026-08-31) | GPU-box session, arXiv PDF text |
| ImageNet-R B0 Inc20 | 0.546 / 0.613 ✔ | 0.680 / 0.758 ✔ | 0.695 / 0.771 (VPT-Deep) | 0.779 ✔ | Zhou v1 Tab. 22 (IN21K); McDonnell v3 Tab. 1 "IN-R" | 2026-09-06 | GPU-box session, arXiv PDF text |
| ImageNet-A B0 Inc20 | 0.489 / 0.598 ✔ | 0.494 / 0.605 ✔ | 0.500 / 0.613 (SSF) | 0.624 ✔ | Zhou v1 Tab. 30 (IN21K); McDonnell v3 Tab. 1 "IN-A" | 2026-09-06 | GPU-box session, arXiv PDF text |
| CUB-200 B0 Inc20 | 0.867 / 0.918 ✔ | 0.871 / 0.921 ✔ | 0.871 / 0.921 (Adapter) | 0.903 ✔ | Zhou v1 Tab. 14 (IN21K); McDonnell v3 Tab. 1 "CUB" | 2026-09-06 | GPU-box session, arXiv PDF text |
| ObjectNet-200 B0 Inc20 | 0.536 / 0.646 ✔ | 0.580 / 0.690 ✔ | 0.582 / 0.699 (SSF) | not evaluated in RanPAC | Zhou v1 Tab. 38 (IN21K) | 2026-09-06 | GPU-box session, arXiv PDF text |
| OmniBenchmark B0 Inc30 | 0.732 / 0.793 ✔ | 0.744 / 0.808 ✔ | 0.745 / 0.811 (VPT-Deep) | 0.799 ✔ | Zhou v1 Tab. 45 (IN21K) = v1/v2 Tab. 1 column; McDonnell v3 Tab. 1 "OB" | 2026-09-06 | GPU-box session, arXiv PDF text |
| VTAB B0 Inc10 | 0.844 / 0.860 ✔ | 0.844 / 0.860 ✔ | 0.854 / 0.872 (VPT-Shallow) | 0.922 ✔ | Zhou v1 Tab. 50 (IN21K) = v1/v2 Tab. 1 column; McDonnell v3 Tab. 1 "VTAB" | 2026-09-06 | GPU-box session, arXiv PDF text |

Raw percentages as printed (avg, last) for the IN21K block of each Zhou v1 table:

| split | table | SimpleCIL | ADAM Finetune | ADAM VPT-Shallow | ADAM VPT-Deep | ADAM SSF | ADAM Adapter |
|---|---|---|---|---|---|---|---|
| CIFAR100 B0 Inc10 | 7 | 87.13, 81.26 | 87.12, 81.23 | 90.25, 85.04 | 90.40, 84.62 | 90.61, 85.14 | 92.24, 87.49 |
| CUB200 B0 Inc20 | 14 | 91.82, 86.73 | 91.45, 86.26 | 80.78, 71.59 | 91.33, 86.51 | 91.47, 86.22 | 92.11, 87.11 |
| ImageNet-R B0 Inc20 | 22 | 61.26, 54.55 | 68.54, 58.37 | 68.83, 62.03 | 77.05, 69.47 | 75.47, 67.02 | 75.82, 67.95 |
| ImageNet-A B0 Inc20 | 30 | 59.77, 48.91 | 61.01, 49.57 | 58.39, 47.20 | 58.48, 48.52 | 61.30, 50.03 | 60.47, 49.37 |
| ObjectNet B0 Inc20 | 38 | 64.58, 53.59 | 64.59, 53.58 | 68.37, 57.60 | 69.58, 57.79 | 69.87, 58.24 | 68.97, 58.03 |
| OmniBenchmark B0 Inc30 | 45 | 79.34, 73.15 | 73.02, 65.03 | 79.63, 73.68 | 81.05, 74.47 | 80.53, 74.00 | 80.75, 74.37 |
| VTAB B0 Inc10 | 50 | 85.99, 84.38 | 87.47, 80.44 | 87.15, 85.36 | 86.59, 83.06 | 85.66, 81.92 | 85.95, 84.35 |

RanPAC v3 Table 1 (final `A_T`, %, T = 10 except VTAB T = 5; backbone per cell
not stated, see finding above):

| row | CIFAR100 | IN-R | IN-A | CUB | OB | VTAB |
|---|---|---|---|---|---|---|
| Ours (Algorithm 1) | 92.2 | 77.9 | 62.4 | 90.3 | 79.9 | 92.2 |
| No RPs | 90.6 | 73.9 | 57.7 | 86.6 | 74.3 | 90.0 |
| No Phase 1 | 89.0 | 71.8 | 58.2 | 88.7 | 78.2 | 92.2 |
| No RPs or Phase 1 | 87.0 | 67.6 | 54.4 | 86.8 | 73.1 | 91.9 |
| NCM with Phase 1 | 87.8 | 70.1 | 49.7 | 85.4 | 73.4 | 88.2 |
| NCM only | 83.4 | 61.2 | 49.3 | 85.1 | 73.1 | 88.4 |
| ADaM (best PETL, copied from Zhou) | 87.6 | 72.3 | 52.6 | 87.1 | 74.3 | 84.3 |
| Joint linear probe | 87.9 | 72.0 | 56.6 | 88.7 | 78.5 | 86.7 |

"NCM only" is RanPAC's cosine nearest-class-mean on the frozen backbone with
no PETL and no random projection, the closest published analogue of our
contracted-prototype *forced* configuration. "No Phase 1" is the RanPAC head
(random projection + second-order decorrelation) on the frozen backbone with
no first-session PETL.

Dataset sizes as printed by Zhou v1 Table 4 (train / test / classes):
CIFAR100 50,000 / 10,000 / 100; CUB200 9,430 / 2,358 / 200; ImageNet-R
24,000 / 6,000 / 200; ImageNet-A 5,981 / 1,519 / 200; ObjectNet 26,509 /
6,628 / 200; OmniBenchmark 89,697 / 5,985 / 300; VTAB 1,796 / 8,619 / 50.
RanPAC Table A2 agrees except that it prints 5985 validation samples for
ImageNet-A (Zhou: 1,519); that looks like a copy of the OmniBenchmark figure
and is not used here.

### Checkpoint inference from RanPAC's NCM-only row (2026-09-07, analysis machine)

RanPAC's "NCM only" value equals Zhou's SimpleCIL last (IN1K block) on IN-R
(61.2 vs 61.28), IN-A (49.3 vs 49.44) and CUB (85.1 vs 85.16), and equals the
IN21K value on OmniBenchmark (73.1 vs 73.15); CIFAR and VTAB (83.4, 88.4) are
within a point of our own IN21K nearest-class-mean (84.4, 89.3). Treat the
IN-R / IN-A / CUB RanPAC anchors as **IN1K-checkpoint** numbers; the in21k_ft_in1k
timm checkpoint (`vit_base_patch16_224.augreg_in21k_ft_in1k`) would remove the
confound.


### Addendum (2026-09-08, analysis machine): prompt-based baselines, IN21K block, last / avg

Read visually from the arXiv v1 PDF pages by the analysis session (Tables 7,
22, 30): CIFAR100 B0 Inc10 — L2P 84.57 / 88.34, DualPrompt 84.14 / 89.69;
ImageNet-R B0 Inc20 — L2P 69.77 / 75.46, DualPrompt 67.18 / 73.10;
ImageNet-A B0 Inc20 — L2P 41.71 / 49.39, DualPrompt 41.67 / 53.71. CUB /
ObjectNet / OmniBenchmark / VTAB L2P and DualPrompt cells were not read and are
left blank.

## Checks before comparing

1. Same split as the papers (image counts printed by `extract_ptm.py` must match
   the distributed split; see `gpu/README.md` and the sizes above).
2. Same backbone checkpoint (in21k, not in1k). Satisfied for every Zhou cell
   above (IN21K block). Not verifiable per cell for RanPAC; treat the RanPAC
   column as "best of two ViT-B/16 checkpoints" until shown otherwise.
3. Last-step column only; average-incremental never compared to ours.
4. Our number is the **certified** configuration; the uncertified free arbiter
   goes in brackets and is not claimed.
5. RanPAC's `A_T` is a per-task mean, not an overall test accuracy; on datasets
   with unequal per-class test counts (ImageNet-A, ObjectNet, VTAB) the two can
   differ by a few tenths of a point.
