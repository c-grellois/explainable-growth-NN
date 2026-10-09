# Benchmark comparison: certified organism vs. published results

*All "this work" numbers are the **certified** configuration (territory-respecting
arbiter under the frozen gate, uncertified answer fraction 0.0000, evaluation
seeds 2–4). Uncertified free-arbiter numbers are shown in brackets and are not
claimed. Published anchors were verified against the source papers on
2026-08-31 (see paper §Related work / citation audit); the metric column states
whether an anchor is last-step or average incremental accuracy — ours is always
last-step (strict: abstention counts as error). Regenerated from
`results/claims.json`.*

## A. Settings with standard published protocols

| setting | method | stores data? | metric | accuracy | vs. ours (certified) |
|---|---|---|---|---|---|
| **split-MNIST**, class-IL, 5×2, single head | EWC (van de Ven & Tolias 2019) | no | last | 0.200 | +0.693 |
| | SI (2019) | no | last | 0.200 | +0.693 |
| | LwF (2019) | no | last | 0.239 | +0.654 |
| | DGR (van de Ven et al. 2021 run) | generator | last | 0.913 | −0.020 |
| | Generative classifier (2021) | one VAE / class | last | **0.938** | −0.045 |
| | iCaRL (2019 run, budget 2,000) | 2,000 raw images | last | **0.946** | −0.053 |
| | strengthened replay (ours, 100 raw imgs/class) | 100 / class | last | 0.871 | **+0.022** |
| | joint upper bound (ours, strengthened) | all | — | 0.977 | −0.084 |
| | **this work, certified lifecycle** | **none** | last | **0.893** [0.915 uncert.] | — |
| **CIFAR-100**, ViT-B/16-in21k, 10×10 class-IL | SimpleCIL (Zhou et al. 2023, v1 Tab. 7, B0 Inc10) | no | last / avg | 0.813 / 0.871 | **+0.022** (last) |
| | APER w/ adapter (Zhou et al. 2023, v1 Tab. 7, B0 Inc10) | no | last / avg | 0.875 / 0.922 | −0.040 (last) |
| | RanPAC (McDonnell et al. 2023) | no (random-projection head) | **final** | **0.922** | −0.087 |
| | NCM on the same features (ours) | no | last | 0.844 | +0.000 forced |
| | strengthened replay (ours, 20 / class) | 20 / class | last | 0.862 | −0.027 |
| | **this work, contracted prototypes** | **none** | last | **0.835 strict / 0.844 forced** | — |
| | this work, certified lifecycle (cascade + arbiter) | none | last | 0.794 [0.835 uncert.] | −0.041 vs protos |
| | **this work, contracted head** (RP 10k + ridge under a contract; answers not retention-stable: 15 changes per 100 test points under growth) | none | last | **0.875 strict / 0.884 forced** | +0.040 vs protos |
| | **this work + first-session adapters**, contracted prototypes (rank-16 adapters on task 0 only, then frozen: a birth step) | none | last | **0.867 strict / 0.876 forced** | +0.035 vs frozen |
| | **this work + first-session adapters**, contracted head (RP 10k) (APER-adapter 0.875, RanPAC full 0.922) | none | last | **0.906 strict / 0.915 forced** | +0.027 vs frozen |
| | this work + first-session adapters, certified lifecycle | none | last | 0.858 | +0.033 vs frozen |
| **CIFAR-100**, DINOv2-B, 10×10 (GPU box, 2026-09-06) | no verified published class-IL anchor for this backbone | — | — | — | — |
| | NCM, same features (ours) | no | last | 0.873 | 0.000 forced |
| | strengthened replay (ours, 20 / class) | 20 / class | last | 0.889 | −0.024 |
| | joint on frozen features | all | — | 0.891 | −0.026 |
| | **this work, contracted prototypes** | **none** | last | **0.865 strict / 0.873 forced** | — |
| | **this work, certified lifecycle** | **none** | last | **0.865** [0.873 uncert.] | equals protos |
| | **this work, contracted head** (RP 10k + ridge under a contract; answers not retention-stable: 8 changes per 100 test points under growth) | none | last | **0.901 strict / 0.909 forced** | +0.036 vs protos |
| **CIFAR-100**, ViT-L/16-in21k, 10×10 (GPU box) | no verified published class-IL anchor | — | — | — | — |
| | NCM, same features (ours) | no | last | 0.877 | 0.000 forced |
| | strengthened replay (ours) | 20 / class | last | 0.892 | −0.025 |
| | **this work, contracted prototypes** | **none** | last | **0.867 strict / 0.877 forced** | — |
| | this work, certified lifecycle | none | last | 0.861 [0.869 uncert.] | −0.006 vs protos |
| | **this work, contracted head** (RP 10k + ridge under a contract; answers not retention-stable: 8 changes per 100 test points under growth) | none | last | **0.904 strict / 0.915 forced** | +0.037 vs protos |
| **CIFAR-100**, ViT-B/16-in21k, full 500/class (GPU box) | same anchors as the 250/class row above | | | | |
| | this work, contracted prototypes | none | last | 0.832 / 0.842 forced | data was not the limit |
| | this work, certified lifecycle | none | last | 0.825 [0.836 uncert.] | +3.1 over the 250/class lifecycle |
| **ImageNet-R**, ViT-B/16-in21k, B0 200 × 20 (GPU box, 2026-09-07) | SimpleCIL (Zhou et al. 2023, v1 suppl.) | no | last | 0.546 | +0.005 forced / -0.022 strict |
| | APER w/ adapter (Zhou et al. 2023, v1 suppl.) | no | last | 0.680 | -0.156 |
| | RanPAC (McDonnell et al. 2023) | no (random-projection head) | final | 0.779 | -0.255 |
| | NCM, same features (ours) | no | last | 0.551 | 0.000 forced |
| | strengthened replay (ours, 20 / class) | 20 / class | last | 0.575 | -0.051 |
| | **this work, contracted prototypes** | **none** | last | **0.524 strict / 0.551 forced** | — |
| | this work, certified lifecycle (adolescence 3/3, arbiter 0/3) | none | last | 0.391 [0.425 uncert.] | -0.093 vs protos |
| | **this work, contracted head** (RP 10k + ridge under a contract; answers not retention-stable: 66 changes per 100 test points under growth) | none | last | **0.652 strict / 0.659 forced** | +0.128 vs protos |
| | **this work + first-session adapters**, contracted prototypes (rank-16 adapters on task 0 only, then frozen: a birth step) | none | last | **0.695 strict / 0.702 forced** | +0.171 vs frozen |
| | **this work + first-session adapters**, contracted head (RP 10k) (APER-adapter 0.680, RanPAC full 0.779) | none | last | **0.763 strict / 0.771 forced** | +0.112 vs frozen |
| | this work + first-session adapters, certified lifecycle | none | last | 0.571 | +0.139 vs frozen |
| | *(ImageNet-1K checkpoint, the one behind RanPAC's column)* contracted prototypes / head | none | last | 0.622 / 0.693 strict | joint NCM 0.611 vs RanPAC's NCM-only 0.612/0.493/0.851 |
| **ImageNet-A**, ViT-B/16-in21k, B0 200 × 20 (GPU box, 2026-09-07) | SimpleCIL (Zhou et al. 2023, v1 suppl.) | no | last | 0.489 | +0.022 forced / -0.035 strict |
| | APER w/ adapter (Zhou et al. 2023, v1 suppl.) | no | last | 0.494 | -0.040 |
| | RanPAC (McDonnell et al. 2023) | no (random-projection head) | final | 0.624 | -0.170 |
| | NCM, same features (ours) | no | last | 0.511 | 0.000 forced |
| | strengthened replay (ours, 20 / class) | 20 / class | last | 0.365 | +0.088 |
| | **this work, contracted prototypes** | **none** | last | **0.454 strict / 0.511 forced** | — |
| | this work, certified lifecycle (adolescence 3/3, arbiter 0/3) | none | last | 0.450 [0.450 uncert.] | -0.004 vs protos |
| | **this work, contracted head** (RP 10k + ridge under a contract; answers not retention-stable: 155 changes per 100 test points under growth) | none | last | **0.451 strict / 0.498 forced** | -0.002 vs protos |
| | **this work + first-session adapters**, contracted prototypes (rank-16 adapters on task 0 only, then frozen: a birth step) | none | last | **0.516 strict / 0.553 forced** | +0.063 vs frozen |
| | **this work + first-session adapters**, contracted head (RP 10k) (APER-adapter 0.494, RanPAC full 0.624) | none | last | **0.559 strict / 0.578 forced** | +0.108 vs frozen |
| | this work + first-session adapters, certified lifecycle | none | last | 0.510 | +0.060 vs frozen |
| | *(ImageNet-1K checkpoint, the one behind RanPAC's column)* contracted prototypes / head | none | last | 0.558 / 0.560 strict | joint NCM 0.513 vs RanPAC's NCM-only 0.612/0.493/0.851 |
| **CUB-200**, ViT-B/16-in21k, B0 200 × 20 (GPU box, 2026-09-07) | SimpleCIL (Zhou et al. 2023, v1 suppl.) | no | last | 0.867 | -0.001 forced / -0.009 strict |
| | APER w/ adapter (Zhou et al. 2023, v1 suppl.) | no | last | 0.871 | -0.013 |
| | RanPAC (McDonnell et al. 2023) | no (random-projection head) | final | 0.903 | -0.045 |
| | NCM, same features (ours) | no | last | 0.866 | 0.000 forced |
| | strengthened replay (ours, 20 / class) | 20 / class | last | 0.861 | -0.003 |
| | **this work, contracted prototypes** | **none** | last | **0.858 strict / 0.866 forced** | — |
| | this work, certified lifecycle (adolescence 3/3, arbiter 0/3) | none | last | 0.822 [0.850 uncert.] | -0.036 vs protos |
| | **this work, contracted head** (RP 10k + ridge under a contract; answers not retention-stable: 14 changes per 100 test points under growth) | none | last | **0.867 strict / 0.876 forced** | +0.009 vs protos |
| | **this work + first-session adapters**, contracted prototypes (rank-16 adapters on task 0 only, then frozen: a birth step) | none | last | **0.854 strict / 0.863 forced** | -0.004 vs frozen |
| | **this work + first-session adapters**, contracted head (RP 10k) (APER-adapter 0.871, RanPAC full 0.903) | none | last | **0.891 strict / 0.901 forced** | +0.024 vs frozen |
| | this work + first-session adapters, certified lifecycle | none | last | 0.740 | -0.082 vs frozen |
| | *(ImageNet-1K checkpoint, the one behind RanPAC's column)* contracted prototypes / head | none | last | 0.849 / 0.875 strict | joint NCM 0.855 vs RanPAC's NCM-only 0.612/0.493/0.851 |
| **ObjectNet-200**, ViT-B/16-in21k, B0 200 × 20 (GPU box, 2026-09-07) | SimpleCIL (Zhou et al. 2023, v1 suppl.) | no | last | 0.536 | +0.007 forced / -0.030 strict |
| | APER w/ adapter (Zhou et al. 2023, v1 suppl.) | no | last | 0.580 | -0.074 |
| | RanPAC (McDonnell et al. 2023) | no | final | not evaluated | — |
| | NCM, same features (ours) | no | last | 0.543 | 0.000 forced |
| | strengthened replay (ours, 20 / class) | 20 / class | last | 0.517 | -0.011 |
| | **this work, contracted prototypes** | **none** | last | **0.506 strict / 0.543 forced** | — |
| | this work, certified lifecycle (adolescence 3/3, arbiter 0/3) | none | last | 0.400 [0.484 uncert.] | -0.106 vs protos |
| | **this work, contracted head** (RP 10k + ridge under a contract; answers not retention-stable: 74 changes per 100 test points under growth) | none | last | **0.531 strict / 0.561 forced** | +0.025 vs protos |
| | **this work + first-session adapters**, contracted prototypes (rank-16 adapters on task 0 only, then frozen: a birth step) | none | last | **0.536 strict / 0.569 forced** | +0.030 vs frozen |
| | **this work + first-session adapters**, contracted head (RP 10k) (APER-adapter 0.580) | none | last | **0.639 strict / 0.647 forced** | +0.109 vs frozen |
| | this work + first-session adapters, certified lifecycle | none | last | 0.429 | +0.029 vs frozen |
| **OmniBenchmark**, ViT-B/16-in21k, B0 300 × 30 (GPU box, 2026-09-07) | SimpleCIL (Zhou et al. 2023, v1 suppl.) | no | last | 0.732 | +0.001 forced / -0.008 strict |
| | APER w/ adapter (Zhou et al. 2023, v1 suppl.) | no | last | 0.744 | -0.020 |
| | RanPAC (McDonnell et al. 2023) | no (random-projection head) | final | 0.799 | -0.075 |
| | NCM, same features (ours) | no | last | 0.733 | 0.000 forced |
| | strengthened replay (ours, 20 / class) | 20 / class | last | 0.757 | -0.033 |
| | **this work, contracted prototypes** | **none** | last | **0.724 strict / 0.733 forced** | — |
| | this work, certified lifecycle (adolescence 2/3, arbiter 0/3) | none | last | 0.669 [0.690 uncert.] | -0.055 vs protos |
| | **this work, contracted head** (RP 10k + ridge under a contract; answers not retention-stable: 19 changes per 100 test points under growth) | none | last | **0.756 strict / 0.766 forced** | +0.032 vs protos |
| | **this work + first-session adapters**, contracted prototypes (rank-16 adapters on task 0 only, then frozen: a birth step) | none | last | **0.711 strict / 0.719 forced** | -0.012 vs frozen |
| | **this work + first-session adapters**, contracted head (RP 10k) (APER-adapter 0.744, RanPAC full 0.799) | none | last | **0.766 strict / 0.774 forced** | +0.010 vs frozen |
| | this work + first-session adapters, certified lifecycle | none | last | 0.656 | -0.013 vs frozen |
| **VTAB**, ViT-B/16-in21k, B0 50 × 10 (GPU box, 2026-09-07) | SimpleCIL (Zhou et al. 2023, v1 suppl.) | no | last | 0.844 | +0.049 forced / +0.044 strict |
| | APER w/ adapter (Zhou et al. 2023, v1 suppl.) | no | last | 0.844 | +0.044 |
| | RanPAC (McDonnell et al. 2023) | no (random-projection head) | final | 0.922 | -0.034 |
| | NCM, same features (ours) | no | last | 0.893 | 0.000 forced |
| | strengthened replay (ours, 20 / class) | 20 / class | last | 0.918 | -0.030 |
| | **this work, contracted prototypes** | **none** | last | **0.888 strict / 0.893 forced** | — |
| | this work, certified lifecycle (adolescence 3/3, arbiter 0/3) | none | last | 0.886 [0.886 uncert.] | -0.002 vs protos |
| | **this work, contracted head** (RP 10k + ridge under a contract; answers not retention-stable: 12 changes per 100 test points under growth) | none | last | **0.929 strict / 0.934 forced** | +0.042 vs protos |
| | **this work + first-session adapters**, contracted prototypes (rank-16 adapters on task 0 only, then frozen: a birth step) | none | last | **0.882 strict / 0.888 forced** | -0.006 vs frozen |
| | **this work + first-session adapters**, contracted head (RP 10k) (APER-adapter 0.844, RanPAC full 0.922) | none | last | **0.939 strict / 0.943 forced** | +0.009 vs frozen |
| | this work + first-session adapters, certified lifecycle | none | last | 0.882 | -0.004 vs frozen |
| **ImageNet-R**, ImageNet-1K checkpoint (RanPAC's family for IN-R / IN-A / CUB), + first-session adapters | RanPAC (McDonnell et al. 2023) | no | final A_T | 0.779 | **this work, contracted head** 0.770 strict / 0.777 forced (-0.2 forced vs RanPAC; metric differs) |
| **ImageNet-A**, ImageNet-1K checkpoint (RanPAC's family for IN-R / IN-A / CUB), + first-session adapters | RanPAC (McDonnell et al. 2023) | no | final A_T | 0.624 | **this work, contracted head** 0.602 strict / 0.607 forced (-1.7 forced vs RanPAC; metric differs) |
| **CUB-200**, ImageNet-1K checkpoint (RanPAC's family for IN-R / IN-A / CUB), + first-session adapters | RanPAC (McDonnell et al. 2023) | no | final A_T | 0.903 | **this work, contracted head** 0.886 strict / 0.895 forced (-0.8 forced vs RanPAC; metric differs) |
| **ObjectNet-200**, ImageNet-1K checkpoint (RanPAC's family for IN-R / IN-A / CUB), + first-session adapters | RanPAC (McDonnell et al. 2023) | no | final A_T | not evaluated | **this work, contracted head** 0.631 strict / 0.637 forced |
| **OmniBenchmark**, ImageNet-1K checkpoint (RanPAC's family for IN-R / IN-A / CUB), + first-session adapters | RanPAC (McDonnell et al. 2023) | no | final A_T | 0.799 | **this work, contracted head** 0.764 strict / 0.772 forced (-2.7 forced vs RanPAC; metric differs) |
| **VTAB**, ImageNet-1K checkpoint (RanPAC's family for IN-R / IN-A / CUB), + first-session adapters | RanPAC (McDonnell et al. 2023) | no | final A_T | 0.922 | **this work, contracted head** 0.939 strict / 0.944 forced (+2.2 forced vs RanPAC; metric differs) |
| **all seven PTM datasets, + first-session adapters, gate-confirmation seeds 5–7** (GPU box, battery 4) | same anchors as above | none | last | head strict 0.903 / 0.769 / 0.573 / 0.888 / 0.642 / 0.766 / 0.936 (CIFAR / IN-R / IN-A / CUB / ObjNet / Omni / VTAB) | within 1.4 pt of seeds 2–4 on every dataset; above the best APER variant on 7/7; arbiter accepted 3/21, one accepted regression of −0.4 (CUB s7, sub-resolution), 0 helpful refused; CIFAR arbiter 0/3 here vs 3/3 on seeds 2–4 (at the rule's threshold) — seeds 5–7 confirmation |
| **CIFAR-100**, backbone trained from scratch | DER (Yan et al. 2021), 2,000 exemplars | 2,000 | last | 0.653 | not comparable |
| | FOSTER (Wang et al. 2022) | 2,000 | last | similar | not comparable |
| | this work, frozen ResNet-18 proxy | none | last | 0.470 (ceiling 0.669) | different regime |

**Reading.** Accuracy-wise the certified organism is competitive, not leading:
−2.0 to −5.3 behind the best exemplar-free and buffered methods on split-MNIST,
+2.2 over SimpleCIL-last and −8.7 to RanPAC on CIFAR-ViT. The GPU-box rows
settle where that 8.7 comes from: on four backbones the contracted prototypes'
forced accuracy equals nearest-class-mean on the same features to the third
decimal (0.843 / 0.842 / 0.873 / 0.877), so the certificate rides feature
quality one-for-one but never above the prototype topology it wraps. The gap
to RanPAC is its trained random-projection head, not the features; DINOv2-B
gives the best certified CIFAR-100 number, 0.865, with the cascade route now
equal to the prototype route. The six-dataset pretrained-model suite (GPU box,
2026-09-07; anchors verified against Zhou et al. v1 supplementary Tables 7, 14,
22, 30, 38, 45, 50 and RanPAC v3 Table 1) repeats the CIFAR finding on every
dataset: forced = NCM to the third decimal (18/18 seeds), and forced lands
within about one point of SimpleCIL-last on ImageNet-R, ImageNet-A, CUB,
ObjectNet and OmniBenchmark, while RanPAC's head buys a further 4 to 23
points. The certified *lifecycle* trails the prototypes by 0.2 to 10.6 points
depending on how much of the input the consolidated organ answers (91–100% on
VTAB / CUB / ImageNet-A, 43–55% on ImageNet-R / ObjectNet), and the arbiter
was refused on all 18 seeds, on 11 of them by the escrow (measurably worse on
real held-out data). Anchor correction: the CIFAR APER-adapter value used
before 2026-09-07 (0.852) was the B0 Inc5 column; the B0 Inc10 value is 0.875,
so that comparison moves from −1.7 to −4.0. What it holds against
every published method is the comparison to *same-data* replay with nothing
stored (+2.2 on split-MNIST), and the certificate axis below. The earlier +0.2
edge over DGR came from the uncertified arbiter and is retracted.

## B. All twelve benchmarks vs. same-data baselines (no published class-IL anchors)

| benchmark (tasks) | fine-tune | EWC | replay (strengthened) | **this work, certified** | joint |
|---|---|---|---|---|---|
| split-digits (5×2) | 0.58 | 0.58 | **0.933** | 0.912 | 0.975 |
| split-MNIST (5×2) | 0.25 | 0.39 | 0.871 | **0.893** | 0.977 |
| split-KMNIST (5×2) | 0.27 | 0.34 | 0.773 | **0.805** | 0.957 |
| split-Fashion (5×2) | 0.21 | 0.32 | 0.777 | **0.799** | 0.894 |
| letter (13×2) | 0.04 | 0.10 | 0.502 | **0.782** | 0.949 |
| satimage (3×2) | 0.33 | 0.33 | **0.878** | 0.833 | 0.902 |
| wall-robot (2×2) | 0.23 | 0.36 | 0.722 | **0.800** | 0.909 |
| CIFAR-100, frozen R18 | — | — | **0.591** | 0.470 | 0.669 |
| CIFAR-100, ViT-B/16 | — | — | **0.862** | 0.794 lifecycle / **0.835** protos | 0.844 (NCM) |
| CIFAR-100, ViT-B/16 full data | — | — | **0.859** | 0.825 lifecycle / 0.832 protos | 0.849 |
| CIFAR-100, DINOv2-B | 0.34 | 0.43 | **0.889** | **0.865** lifecycle = protos | 0.891 |
| CIFAR-100, ViT-L/16 | 0.32 | 0.56 | **0.892** | 0.861 lifecycle / 0.867 protos | 0.877 |
| ImageNet-R, ViT-B/16 (200 × 20) | 0.053 | 0.072 | **0.575** | 0.431 lifecycle / 0.524 protos | 0.435 |
| ImageNet-A, ViT-B/16 (200 × 20) | 0.015 | 0.196 | 0.365 | 0.450 lifecycle / **0.454** protos | 0.105 |
| CUB-200, ViT-B/16 (200 × 20) | 0.225 | 0.450 | **0.861** | 0.822 lifecycle / 0.858 protos | 0.717 |
| ObjectNet-200, ViT-B/16 (200 × 20) | 0.078 | 0.078 | **0.517** | 0.400 lifecycle / 0.506 protos | 0.388 |
| OmniBenchmark, ViT-B/16 (300 × 30) | 0.198 | 0.164 | **0.757** | 0.669 lifecycle / 0.724 protos | 0.674 |
| VTAB, ViT-B/16 (50 × 10) | 0.395 | 0.709 | **0.918** | 0.886 lifecycle / 0.888 protos | 0.609 |
| sheet: sweeps / shapes | — | — | ~0.99 | 0.989 / 0.985 | ~1.00 |
| spoken digits, shared-speaker split † | — | — | **0.93** | 0.790 | 0.95 |
| spoken digits, **speaker-disjoint** † | 0.16 | 0.17 | **0.451** | 0.328 | 0.518 |

Ours ≥ replay on **5 of 12** original benchmarks (MNIST, KMNIST, Fashion, letter,
wall-robot); on the three GPU-box CIFAR backbones replay leads by 2.5–3.4
points; on the six-dataset suite contracted prototypes beat same-data replay on
ImageNet-A (+8.9), CUB (−0.3, tie) and ObjectNet (−1.1, tie) and trail it on
ImageNet-R, OmniBenchmark and VTAB (2.9–5.1), while joint training on the
frozen features is *below* the prototypes on every suite dataset but one.
Class-order permutations move the certified numbers by 0.4–4.8 points on the
small-task benchmarks (most on wall-robot and Fashion), so single figures there
carry a ±2–4 point order band. † The standard FSDD split shares speakers across
train/val/test; under a speaker-disjoint split *every* method loses roughly
half its accuracy, so the collapse is the benchmark's, not the organism's.

| **Banking77** (text, frozen all-mpnet-base-v2; 11 × 7) | protos 0.853 / head **0.921** strict (0.939 forced) / certified 0.814 | NCM 0.871 · replay 0.918 · joint 0.881 | head above replay and joint; flips 0 |
| **CLINC150** (text; 10 × 15) | protos 0.922 / head **0.950** (0.965) / certified 0.830 | NCM 0.938 · replay 0.956 · joint 0.928 | head above replay (forced) and joint; flips 0 |
| **NSL-KDD** (tabular, raw flow records; 4 × 2) | protos 0.428 / head **0.935** (0.951) / certified 0.781 (= cascade) | NCM 0.445 · replay 0.951 · joint 0.952 | NCM fails on this space, the head matches replay/joint; every consolidation and arbiter refused, correctly; flips 0 |

New substrates (GPU-box battery 4, 2026-09-15; REPORT.md B-text, B-tabular): no published class-IL anchor has been read from a source PDF for these, so the comparison is to the same-data baselines the chain produces.

## C. The axis no published method occupies

| property | this work (certified configuration) | published continual learning |
|---|---|---|
| forgetting under growth | **exactly 0 answer changes**, verified bitwise, every run, 12 benchmarks, 3 substrates | measured after the fact, never guaranteed |
| answers that never change | the cascade and the **frozen-block head** change exactly 0 answers when a part is added (bitwise, every run); the re-solved contracted head changes 8–155 per 100 test points, and costs 0.482 vs 0.869 strict on CIFAR-ViT if frozen instead | not reported by any published method |
| answers that never change, at 100 updates | contracted prototypes identical to 12 decimals at 10 / 20 / 50 / 100 steps, 0 changes; cascade growth flips **0** over 501 checked steps; the re-solved head ends at the same accuracy but rewrites 13 → 41 answers per 100 test points (CIFAR) and 40 → 141 (ImageNet-R) along the way | no published result beyond Inc5 |
| certified lifecycle vs horizon | 0.795 / 0.717 / 0.747 / 0.729 strict at 10 / 20 / 50 / 100 organs; arbiter refused 9/9, including a −14-point candidate at 100 organs (z -21) and one helpful +3.8 refused by the per-class hard floor | no gate |
| declining out-of-distribution input (certificate slack at the contract's own threshold) | CIFAR-100 ViT-B → CIFAR-10: AUROC 0.841, 79 % accepted; → DTD textures: 0.995, 3 % accepted; → SVHN digits: 0.662, 99 % accepted; CLINC150 → out-of-scope text: 0.955, 42 % accepted. Above kNN / NCM / Mahalanobis on every set; below a joint probe's softmax on CIFAR-10 (0.903) | fine-tuned-encoder OOD numbers, not comparable |
| cost after the encoder (CIFAR-100 ViT-B) | prototypes 1.9 M floats / 2 s per task; cascade 4.9 M / 290 s; head-2k 3.7 M + 4.2 M Gram; head-10k 10.6 M + 101 M Gram / 108 s | RanPAC head-10k 8.7 M + 101 M (arithmetic) |
| size of the grown network, and how much of it is needed | split-MNIST: 5,148 neurons, 1,117,013 connections for 0.893 strict; pruning each trunk at birth holds accuracy to 99 % sparsity (0.8914, trunks 10,445 connections) and collapses by 99.9 %; growth flips 0 at every sparsity, including the collapsed ones | Han et al. 2015 prune a LeNet-300-100 12× to ~22K parameters at 98.4 %, jointly trained (not this protocol) |
| answers covered by a checkable contract | **100%** (uncertified fraction 0.0000 in all 45 evaluation runs) | none |
| acceptance of lifecycle changes | gated; **102 runs on record**: 4/4 known-harmful refused, 0 clearly-helpful refused, 2 sub-resolution slips, 1 miss | no gate |
| stored raw data | **none** (contracts are the generators) | buffers or generators |
| replaceable parts | organ re-derived from its spec alone agrees 0.92–0.93; whole nested organisms 0.75–0.78 | no |
| self-forecast drift bound (SLA) | 5 violations / 54 events vs. a 25% budget | no |

## D. Curricula (a different protocol; not comparable to the anchors above)

| CIFAR-100, ViT-B/16, seeds 2–4 | growth | certified lifecycle | vetoes | contracted prototypes |
|---|---|---|---|---|
| random class order (the protocol) | 0.667 | 0.794 | 250 | 0.835 / 0.843 |
| superclass-grouped tasks | 0.698 | 0.799 | 159 | unchanged |
| easy-first / hard-first / spread | 0.664 / 0.643 / 0.670 | 0.788 / 0.794 / 0.786 | | unchanged |
| containers (20 superclasses) then refinement | coarse 0.774; fine 0.626 strict / 0.696 forced | — (growth only) | 30 | ceiling 0.777 for any coarse-first NCM |
| containers → **prototype** children | coarse 0.774; fine 0.665 strict / 0.714 forced | — | 30 | children 0.08M params |
| containers → two-step children (3 + 2) | coarse 0.774; fine 0.611 / 0.635 | — | 30 + 138 within | within-container flips 0 |
| progressive consolidation (after tasks 2,4,6,8,10) | 0.771 | — | 51 | vs 0.776 end-only, 0.700 mid-life |

Order policies and label hierarchies use information the standard stream does
not provide, so these rows are a separate setting (REPORT.md, B-curriculum).
Prototypes are order-invariant; only the cascade moves.

*Sources for anchors: van de Ven & Tolias, "Three scenarios for continual
learning" (2019, Table 4); van de Ven, Li & Tolias, "Class-incremental learning
with generative classifiers" (CVPR-W 2021, Table 2); Zhou et al., "Revisiting
class-incremental learning with pre-trained models" (2023, Table 1); McDonnell
et al., "RanPAC" (NeurIPS 2023, Table 1, final accuracy A_T); Yan et al., "DER"
(CVPR 2021, Table 1).*
