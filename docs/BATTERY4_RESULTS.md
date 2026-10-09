# Battery 4 on the GPU box — confirmation seeds, far-OOD caches, a text substrate, a tabular substrate (2026-09-14/15)

*Battery report, 2026-09-15, written on the GPU box, merged here unchanged. Its numbers are reproduced by the `B-ptm-confirm`, `B-text`, `B-tabular` and `B-in1k-phase1` sections of [REPORT.md](../REPORT.md) from the same result files. Section 7 is the box's own incident report.*

*Defined in `HANDOFF_GPU3.md`. Rules of `HANDOFF_GPU.md` §2/§9 and `HANDOFF_GPU3.md`
§9 followed: one interpreter (`gpu\venv`), evaluation seeds only, nothing tuned
after seeing a number, no published number quoted from memory (none appears in
this file), no existing result file written. Tables are the output of
`gpu\summarize_battery4.py` over `results\*.json` (per-seed values verbatim;
means over the seeds shown). All numbers are last-step accuracy on the full
test set; strict = abstention counts as error. Every chain's `.err` file is
free of tracebacks (31 chains for A–D). One loader fix was needed (§7).*

## Block A — first-session adapters, confirmation seeds 5, 6, 7

Same recipe as battery 3 (`phase1_extract.py`: rank-16 parallel adapters on
every block's MLP, cosine head on task 0, 20 epochs, then frozen; cache is
seed-specific and was run with its own seed only). 21/21 caches produced, no
failures (`gpu\out\run_phase1_s567.log`, per-run `phase1_<name>_s<seed>.log`);
joint NCM of each cache in `gpu\out\cache_summary_p1s567.txt`.

| dataset | seed | Phase-1 task-0 val acc | tuned params | contracted protos strict / forced | head M=10k strict / forced / coverage / changes per 100 | growth strict (flips) | adolescence (accepted?) | certified lifecycle (arbiter?) | free arbiter | uncertified fraction | NCM |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CIFAR-100 | 5 | 0.996 | 312,000 | 0.851 / 0.858 | 0.904 / 0.912 / 0.986 / 7 | 0.800 (0) | 0.835 (✔) | 0.835 (arb REJ) | [0.860] | 0.000 | 0.858 |
| CIFAR-100 | 6 | 0.988 | 312,000 | 0.861 / 0.871 | 0.903 / 0.912 / 0.984 / 7 | 0.801 (0) | 0.849 (✔) | 0.849 (arb REJ) | [0.873] | 0.000 | 0.871 |
| CIFAR-100 | 7 | 1.000 | 312,000 | 0.865 / 0.875 | 0.904 / 0.915 / 0.981 / 6 | 0.816 (0) | 0.858 (✔) | 0.858 (arb REJ) | [0.875] | 0.000 | 0.875 |
| **CIFAR-100** | mean s5–7 | 0.995 | | 0.859 / 0.868 | 0.903 / 0.913 / 0.984 / 7 | 0.806 (0) | 0.847 (3/3 acc.) | 0.847 (arb 0/3) | [0.869] | 0.000 max | 0.868 |
| ImageNet-R | 5 | 0.934 | 319,680 | 0.698 / 0.705 | 0.774 / 0.780 / 0.988 / 28 | 0.489 (0) | 0.571 (✔) | 0.571 (arb REJ) | [0.656] | 0.000 | 0.705 |
| ImageNet-R | 6 | 0.949 | 319,680 | 0.685 / 0.693 | 0.765 / 0.770 / 0.988 / 33 | 0.487 (0) | 0.575 (✔) | 0.575 (arb REJ) | [0.654] | 0.000 | 0.693 |
| ImageNet-R | 7 | 0.923 | 319,680 | 0.709 / 0.715 | 0.767 / 0.775 / 0.979 / 30 | 0.477 (0) | 0.575 (✔) | 0.581 (arb ✔) | [0.682] | 0.000 | 0.715 |
| **ImageNet-R** | mean s5–7 | 0.935 | | 0.697 / 0.704 | 0.769 / 0.775 / 0.985 / 31 | 0.484 (0) | 0.574 (3/3 acc.) | 0.576 (arb 1/3) | [0.664] | 0.000 max | 0.704 |
| ImageNet-A | 5 | 0.875 | 319,680 | 0.544 / 0.566 | 0.589 / 0.591 / 0.988 / 107 | 0.223 (0) | 0.539 (✔) | 0.539 (arb REJ) | [0.540] | 0.000 | 0.566 |
| ImageNet-A | 6 | 0.862 | 319,680 | 0.542 / 0.562 | 0.558 / 0.575 / 0.886 / 104 | 0.194 (0) | 0.546 (✔) | 0.546 (arb REJ) | [0.546] | 0.000 | 0.562 |
| ImageNet-A | 7 | 0.816 | 319,680 | 0.517 / 0.557 | 0.571 / 0.577 / 0.990 / 98 | 0.166 (0) | 0.544 (✔) | 0.544 (arb REJ) | [0.544] | 0.000 | 0.557 |
| **ImageNet-A** | mean s5–7 | 0.851 | | 0.535 / 0.562 | 0.573 / 0.581 / 0.955 / 103 | 0.194 (0) | 0.543 (3/3 acc.) | 0.543 (arb 0/3) | [0.543] | 0.000 max | 0.562 |
| CUB-200 | 5 | 0.991 | 319,680 | 0.850 / 0.862 | 0.881 / 0.894 / 0.981 / 11 | 0.615 (0) | 0.805 (✔) | 0.805 (arb REJ) | [0.849] | 0.000 | 0.862 |
| CUB-200 | 6 | 0.989 | 319,680 | 0.871 / 0.875 | 0.892 / 0.898 / 0.993 / 11 | 0.659 (0) | 0.795 (✔) | 0.795 (arb REJ) | [0.860] | 0.000 | 0.875 |
| CUB-200 | 7 | 0.949 | 319,680 | 0.855 / 0.863 | 0.891 / 0.899 / 0.989 / 12 | 0.573 (0) | 0.840 (✔) | **0.835 (arb ✔)** | [0.863] | 0.000 | 0.863 |
| **CUB-200** | mean s5–7 | 0.976 | | 0.858 / 0.867 | 0.888 / 0.897 / 0.988 / 11 | 0.616 (0) | 0.813 (3/3 acc.) | 0.812 (arb 1/3) | [0.858] | 0.000 max | 0.867 |
| ObjectNet-200 | 5 | 0.905 | 319,680 | 0.512 / 0.553 | 0.639 / 0.647 / 0.974 / 42 | 0.390 (0) | 0.390 (REJ) | 0.390 (arb REJ) | [0.398] | 0.000 | 0.553 |
| ObjectNet-200 | 6 | 0.921 | 319,680 | 0.543 / 0.566 | 0.643 / 0.651 / 0.971 / 39 | 0.404 (0) | 0.456 (✔) | 0.456 (arb REJ) | [0.544] | 0.000 | 0.566 |
| ObjectNet-200 | 7 | 0.891 | 319,680 | 0.520 / 0.560 | 0.643 / 0.650 / 0.978 / 38 | 0.379 (0) | 0.379 (REJ) | 0.379 (arb REJ) | [0.391] | 0.000 | 0.560 |
| **ObjectNet-200** | mean s5–7 | 0.906 | | 0.525 / 0.560 | 0.642 / 0.649 / 0.974 / 40 | 0.391 (0) | 0.408 (1/3 acc.) | 0.408 (arb 0/3) | [0.444] | 0.000 max | 0.560 |
| OmniBenchmark | 5 | 0.964 | 327,360 | 0.717 / 0.724 | 0.770 / 0.777 / 0.979 / 18 | 0.659 (0) | 0.659 (REJ) | 0.659 (arb REJ) | [0.667] | 0.000 | 0.724 |
| OmniBenchmark | 6 | 0.961 | 327,360 | 0.710 / 0.720 | 0.763 / 0.774 / 0.973 / 17 | 0.659 (0) | 0.659 (REJ) | 0.659 (arb REJ) | [0.672] | 0.000 | 0.720 |
| OmniBenchmark | 7 | 0.909 | 327,360 | 0.710 / 0.716 | 0.765 / 0.772 / 0.979 / 17 | 0.656 (0) | 0.656 (REJ) | 0.666 (arb ✔) | [0.675] | 0.000 | 0.716 |
| **OmniBenchmark** | mean s5–7 | 0.945 | | 0.712 / 0.720 | 0.766 / 0.774 / 0.977 / 17 | 0.658 (0) | 0.658 (0/3 acc.) | 0.661 (arb 1/3) | [0.671] | 0.000 max | 0.720 |
| VTAB | 5 | 1.000 | 312,000 | 0.902 / 0.904 | 0.942 / 0.943 / 0.998 / 9 | 0.522 (0) | 0.894 (✔) | 0.894 (arb REJ) | [0.895] | 0.000 | 0.904 |
| VTAB | 6 | 0.966 | 312,000 | 0.886 / 0.889 | 0.936 / 0.942 / 0.990 / 10 | 0.561 (0) | 0.884 (✔) | 0.884 (arb REJ) | [0.885] | 0.000 | 0.889 |
| VTAB | 7 | 0.975 | 312,000 | 0.889 / 0.889 | 0.931 / 0.935 / 0.992 / 6 | 0.493 (0) | 0.884 (✔) | 0.884 (arb REJ) | [0.884] | 0.000 | 0.889 |
| **VTAB** | mean s5–7 | 0.980 | | 0.892 / 0.894 | 0.936 / 0.940 / 0.993 / 9 | 0.525 (0) | 0.887 (3/3 acc.) | 0.887 (arb 0/3) | [0.888] | 0.000 max | 0.894 |

Growth flips **0 on 21/21 seeds**; uncertified fraction **0.0 on 21/21**.
Contracted-prototype *forced* equals NCM on the same cache on 21/21 seeds.

### Gate record, seeds 5–7

- **Arbiter accepted 3/21**: ImageNet-R seed 7 (0.575 → 0.581, escrow z +2.56),
  CUB seed 7 (see regression below), OmniBenchmark seed 7 (proposed on the
  un-consolidated organism after adolescence was refused: 0.656 → 0.666, z +5.24).
  Refused 18/21: 15 by the original rule (candidate not +0.01 above the
  organism on the synthetic holdout, or last-task real validation not
  improved), 3 also by the escrow (CUB seed 6 z −6.68; ObjectNet seeds 5–7
  z −1.00 to −3.70 with candidates at or below the organism).
- **Adolescence accepted 16/21**; refused 5/21, all "mean regressed" on the
  synthetic holdout: ObjectNet seed 5 (0.579 vs 0.604) and seed 7 (0.578 vs
  0.610), OmniBenchmark seeds 5, 6, 7 (0.762 vs 0.819; 0.766 vs 0.849; 0.766
  vs 0.831). With battery 3, adapter-tuned OmniBenchmark consolidations are
  now refused on 6/6 seeds.
- **Accepted regression: one.** CUB seed 7: the arbiter was accepted with
  candidate test 0.835 against adolescence test 0.840, i.e. **−0.004**
  (−0.0042 unrounded). Its gate record: original rule passed on the synthetic
  holdout and the last-task real validation, territory passed, escrow passed
  with z −0.83 (inside the −1.645 tolerance). The regression is below the
  escrow's resolution on 2,358 test points.
- **Refused candidates whose test accuracy exceeded adolescence by > 0.01: none.**
  The nearest cases are the three CIFAR seeds (candidates +0.007, +0.006,
  +0.002 above adolescence, escrow z +2.9 / +3.5 / +3.5), all refused by the
  original rule's +0.01 synthetic-holdout requirement. On seeds 2–4 the same
  configuration was accepted 3/3 (candidates +0.006 / +0.008 / +0.004); the
  CIFAR arbiter therefore sits right at the rule's threshold and the 3/3 of
  battery 3 does not replicate: **0/3 on seeds 5–7, 3/6 overall.**

### Seeds 5–7 vs. seeds 2–4

Reference = mean ± sd (n = 3) of seeds 2–4 computed from the same result files
on this box (the `+ first-session adapters` rows of `BENCHMARKS.md` §A are the
analysis machine's re-run of those caches and agree to ±0.003 on prototypes
and heads; their certified values differ where a gate decision differed
there: CUB 0.740 vs 0.787 here, OmniBenchmark 0.656 vs 0.651). "inside" =
|Δ| ≤ 2 sd; note that with n = 3 several sd's are ≤ 0.005, so "OUTSIDE" below
often means a shift of 0.003–0.01.

| dataset | protos strict | protos forced | head strict | head forced | certified |
|---|---|---|---|---|---|
| CIFAR-100 | 0.867±0.009 → 0.859 inside | 0.876±0.010 → 0.868 inside | 0.906±0.002 → 0.903 inside | 0.915±0.002 → 0.913 inside | 0.859±0.006 → 0.847 inside |
| ImageNet-R | 0.695±0.005 → 0.697 inside | 0.702±0.005 → 0.704 inside | 0.763±0.003 → 0.769 inside | 0.771±0.003 → 0.775 inside | 0.565±0.005 → 0.576 **outside** (+0.011; one arbiter acceptance) |
| ImageNet-A | 0.516±0.063 → 0.535 inside | 0.553±0.038 → 0.562 inside | 0.559±0.036 → 0.573 inside | 0.578±0.023 → 0.581 inside | 0.510±0.038 → 0.543 inside |
| CUB-200 | 0.854±0.004 → 0.858 inside | 0.863±0.006 → 0.867 inside | 0.891±0.005 → 0.888 inside | 0.901±0.002 → 0.897 **outside** (−0.004) | 0.787±0.023 → 0.812 inside |
| ObjectNet-200 | 0.536±0.013 → 0.525 inside | 0.569±0.004 → 0.560 **outside** (−0.009) | 0.639±0.000 → 0.642 **outside** (+0.003; sd 0.000) | 0.647±0.001 → 0.649 **outside** (+0.002) | 0.435±0.025 → 0.408 inside |
| OmniBenchmark | 0.711±0.007 → 0.712 inside | 0.719±0.007 → 0.720 inside | 0.766±0.003 → 0.766 inside | 0.774±0.004 → 0.774 inside | 0.651±0.003 → 0.661 **outside** (+0.010; one arbiter acceptance) |
| VTAB | 0.882±0.003 → 0.892 **outside** (+0.010) | 0.888±0.002 → 0.894 **outside** (+0.006) | 0.939±0.007 → 0.936 inside | 0.943±0.004 → 0.940 inside | 0.778±0.182 → 0.887 inside |

Reading: 27 of 35 cells are inside two standard deviations; the 8 outside are
all shifts of ≤ 0.011, and 6 of them are "outside" only because the seeds-2–4
sd is ≤ 0.004. The two certified-lifecycle shifts (ImageNet-R, OmniBenchmark)
are each one arbiter acceptance on one seed. The headline configuration
(contracted head + adapters) moves by at most 0.006 on any dataset. VTAB's
certified mean rises from 0.778 to 0.887 because the seed-2 adolescence
refusal of battery 3 did not recur (3/3 accepted here).

## Block B — far-OOD caches (shapes only; no analysis on the box)

| cache | backbone | rows × dims | note |
|---|---|---|---|
| `ood_svhn_vitb.npz` | vit_base_patch16_224.augreg_in21k | 26,032 × 768 | SVHN test split |
| `ood_dtd_vitb.npz` | vit_base_patch16_224.augreg_in21k | 5,640 × 768 | DTD, all splits |
| `cifar10_dinov2b.npz` | vit_base_patch14_dinov2.lvd142m | train 5,000 × 768 (500/class), test 10,000 × 768 | in-distribution partner |
| `ood_svhn_dinov2b.npz` | vit_base_patch14_dinov2.lvd142m | 26,032 × 768 | SVHN test split |
| `ood_clinc_oos_mpnet.npz` (from Block C) | all-mpnet-base-v2 | 1,350 × 768 | CLINC150 out-of-scope queries (train 250 + val 100 + test 1,000); not a class, never read on the box |

Extraction console output: `gpu\out\extract_battery4.txt` (`--batch 64` was
used because the GPU was shared with the Phase-1 runs; batch size does not
change the features).

## Block C — text substrate (frozen `sentence-transformers/all-mpnet-base-v2`, mean-pooled, fp16, never tuned)

Caches: `ptm_banking77_mpnet.npz` (77 classes, 10,003 / 3,080, inc 7 → 11
tasks; joint NCM 0.871) and `ptm_clinc150_mpnet.npz` (150 in-scope intents,
18,000 / 4,500 after folding HF's validation split into train, inc 15 → 10
tasks; joint NCM 0.939). Standard PTM chain, seeds 2–4 in one chain each.

### Banking77

| seed | growth strict / coverage / flips | contracted protos strict / forced | head M=10k strict / forced / coverage / changes per 100 | adolescence (accepted?) | certified lifecycle (arbiter?) | free arbiter | uncertified | NCM | replay | joint | EWC | fine-tune |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.761 / 0.966 / 0 | 0.849 / 0.870 | 0.922 / 0.939 / 0.979 / 6 | 0.792 (✔) | 0.792 (arb REJ) | [0.889] | 0.000 | 0.870 | 0.915 | 0.890 | 0.297 | 0.228 |
| 3 | 0.752 / 0.948 / 0 | 0.850 / 0.871 | 0.915 / 0.939 / 0.969 / 6 | 0.786 (✔) | 0.820 (arb ✔) | [0.885] | 0.000 | 0.871 | 0.923 | 0.881 | 0.388 | 0.182 |
| 4 | 0.750 / 0.963 / 0 | 0.859 / 0.871 | 0.926 / 0.938 / 0.988 / 7 | 0.794 (✔) | 0.830 (arb ✔) | [0.894] | 0.000 | 0.871 | 0.914 | 0.873 | 0.356 | 0.201 |
| **mean** | 0.755 / — / 0 | 0.853 / 0.871 | 0.921 / 0.939 / 0.979 / 7 | 0.790 (3/3 acc.) | 0.814 (arb 2/3) | [0.890] | 0.000 max | 0.871 | 0.918 | 0.881 | 0.347 | 0.204 |

Gate per seed: seed 2 — original rule **failed** (arbiter 0.696 vs organism
0.759 on the last task's real validation split, while the synthetic-holdout
mean improved 0.978 vs 0.932), territory passed, escrow passed z +3.11;
candidate test 0.821 vs adolescence 0.792, i.e. **a refused candidate better
by +0.029 on test**. Seeds 3, 4 — all three checks passed (z +1.61, +3.79),
accepted, certified 0.820 / 0.830.

### CLINC150 (in-scope)

| seed | growth strict / coverage / flips | contracted protos strict / forced | head M=10k strict / forced / coverage / changes per 100 | adolescence (accepted?) | certified lifecycle (arbiter?) | free arbiter | uncertified | NCM | replay | joint | EWC | fine-tune |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.766 / 0.932 / 0 | 0.923 / 0.938 | 0.950 / 0.964 / 0.982 / 4 | 0.799 (✔) | 0.799 (arb REJ) | [0.931] | 0.000 | 0.938 | 0.957 | 0.924 | 0.629 | 0.283 |
| 3 | 0.791 / 0.954 / 0 | 0.916 / 0.939 | 0.945 / 0.966 / 0.974 / 3 | 0.819 (✔) | 0.850 (arb ✔) | [0.933] | 0.000 | 0.939 | 0.956 | 0.930 | 0.639 | 0.246 |
| 4 | 0.789 / 0.973 / 0 | 0.926 / 0.937 | 0.955 / 0.966 / 0.988 / 4 | 0.819 (✔) | 0.840 (arb ✔) | [0.936] | 0.000 | 0.937 | 0.956 | 0.929 | 0.648 | 0.229 |
| **mean** | 0.782 / — / 0 | 0.922 / 0.938 | 0.950 / 0.965 / 0.981 / 4 | 0.813 (3/3 acc.) | 0.830 (arb 2/3) | [0.933] | 0.000 max | 0.938 | 0.956 | 0.928 | 0.639 | 0.253 |

Gate per seed: seed 2 — original rule **failed** (last-task real validation),
territory and escrow passed (z +6.97); candidate test 0.839 vs adolescence
0.799, **a refused candidate better by +0.040 on test**. Seeds 3, 4 — accepted
(z +5.74, +4.95), certified 0.850 / 0.840.

### Four sentences (Block C)

1. **The certificate transfers to text unchanged**: growth flips 0 and
   uncertified fraction 0.0 on all 6 seeds, contracted-prototype forced equals
   NCM on 6/6 seeds, and no script or configuration was touched (the caches are
   768-d and carry the `ptm_` prefix; only the dataset loader was repaired, §7).
2. **Against same-data replay and NCM**: the contracted head sits above replay
   (Banking77 0.939 vs 0.918 forced, 0.921 vs 0.918 strict; CLINC150 0.965 vs
   0.956 forced, 0.950 vs 0.956 strict) and above the joint MLP (0.881, 0.928);
   the contracted prototypes equal NCM (0.871, 0.938) and beat the joint MLP on
   CLINC150; the certified lifecycle (0.814, 0.830) sits 6–11 points under the
   prototypes because the consolidated organ answers only 51–59 % (Banking77)
   and 39–40 % (CLINC150) of the test set and the grown cascade answers the rest.
3. **The gate accepted the arbiter on 4 of 6 seeds — the first acceptances
   outside CIFAR** — each time raising the certified number by 3–4 points with
   escrow z between +1.6 and +5.7 and no accepted regression; the two refusals
   (seed 2 on both datasets) came from the last-task real-validation clause of
   the original rule while the escrow and synthetic holdout both favoured the
   candidate, and in both cases the refused candidate would have scored higher
   on test (+0.029, +0.040) — the clause is conservative on these streams.
4. **Nothing failed to run.** The HF loader needed two fixes to run under
   `datasets` 5 (§7); the resulting texts, splits and label ids are those of
   the original `banking77` / `clinc_oos` scripts.

## Block D — tabular substrate (NSL-KDD, raw rows, CPU only)

Protocol as printed by `extract_tabular.py` (verbatim; fixed preprocessing:
identifier columns dropped, protocol/service/flag one-hot, log1p on non-negative
numerics, keep a class iff ≥ 300 train and ≥ 30 test rows, 5,000-row cap per
class, inc = 2):

```
nslkdd: 122 features | 8 classes kept (>= 300 train, >= 30 test):
   0 back                         train    956 test    359
   1 ipsweep                      train   3599 test    141
   2 neptune                      train   5000 test   4657
   3 nmap                         train   1493 test     73
   4 normal                       train   5000 test   9711
   5 portsweep                    train   2931 test    157
   6 satan                        train   3633 test    735
   7 smurf                        train   2646 test    665
  dropped (32): apache2 (0/737), buffer_overflow (30/20), ftp_write (8/3), guess_passwd (53/1231), httptunnel (0/133), imap (11/1), land (18/7), loadmodule (9/2), mailbomb (0/293), mscan (0/996), multihop (7/18), named (0/17), perl (3/2), phf (4/2), pod (201/41), processtable (0/685), ps (0/15), rootkit (10/13), saint (0/319), sendmail (0/14), snmpgetattack (0/178), snmpguess (0/331), spy (2/0), sqlattack (0/2), teardrop (892/12), udpstorm (0/2), warezclient (890/0), warezmaster (20/944), worm (0/2), xlock (0/9), xsnoop (0/4), xterm (0/13)
wrote out\ptm_nslkdd_raw.npz  train (25258, 122) test (16498, 122)  inc 2 -> 4 tasks
```

Joint NCM 0.763 (`gpu\out\extract_battery4_tabular.txt`). Organ configuration:
the 768-d CIFAR one, unchanged. Seeds 2–4, one chain.

| seed | growth strict / coverage / flips | contracted protos strict / forced | head M=10k strict / forced / coverage / changes per 100 | adolescence (accepted?) | certified lifecycle (arbiter?) | free arbiter | uncertified | NCM | replay | joint | EWC | fine-tune |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 0.682 / 0.814 / 0 | 0.428 / 0.444 | 0.936 / 0.952 / 0.979 / 5 | 0.682 (REJ: regressed on last task's real val) | 0.682 (arb REJ) | [0.722] | 0.000 | 0.444 | 0.942 | 0.951 | 0.742 | 0.382 |
| 3 | 0.885 / 0.987 / 0 | 0.424 / 0.443 | 0.934 / 0.951 / 0.976 / 0 | 0.885 (REJ: regressed on last task's real val) | 0.885 (arb REJ) | [0.671] | 0.000 | 0.443 | 0.959 | 0.952 | 0.919 | 0.877 |
| 4 | 0.777 / 0.903 / 0 | 0.433 / 0.450 | 0.934 / 0.950 / 0.978 / 0 | 0.777 (REJ: regressed on last task's real val) | 0.777 (arb REJ) | [0.814] | 0.000 | 0.450 | 0.953 | 0.952 | 0.746 | 0.612 |
| **mean** | 0.781 / — / 0 | 0.428 / 0.445 | 0.935 / 0.951 / 0.978 / 2 | 0.781 (0/3 acc.) | 0.781 (arb 0/3) | [0.735] | 0.000 max | 0.445 | 0.951 | 0.952 | 0.802 | 0.624 |

Gate per seed: adolescence refused 3/3 on the last task's real validation;
arbiter refused 3/3 by the original rule and by the escrow (z −8.44, −14.63,
−9.43; candidates 0.556 / 0.631 / 0.764 against organisms 0.682 / 0.885 /
0.777), territory passed 3/3.

### Four sentences (Block D)

1. **The certificate transfers to raw tabular rows**: growth flips 0 and
   uncertified fraction 0.0 on 3/3 seeds, forced prototypes equal NCM on 3/3,
   and the 768-d organ configuration ran unmodified on 122-d inputs; it is not
   "visibly wrong" in the handoff's sense (growth strict 0.68–0.89 is far
   *above* NCM, coverage 0.81–0.99), but its seed variance is large.
2. **Against same-data replay and NCM**: nearest-class-mean is a poor readout
   on this space (0.445 forced; the class means of one-hot/log1p flow records
   are not separable by Euclidean distance), replay and the joint MLP reach
   0.95, and the contracted head reaches the same level (0.951 forced / 0.935
   strict, coverage 0.98, 0–5 answer changes per 100 under growth), while the
   certified lifecycle equals the grown cascade at 0.781 mean (0.68 / 0.89 /
   0.78) because every consolidation and every arbiter was refused.
3. **The gate refused everything, correctly**: the consolidated prototype organ
   inherits NCM's weakness on this space and regressed on real validation on
   every seed, and the arbiter candidates were worse than the organism by 1–25
   points with escrow z from −8 to −15 — the strongest refusals of any battery.
4. **Nothing failed to run**; NSL-KDD downloaded from the first GitHub mirror,
   the whole block took 10 minutes of CPU, and no second tabular dataset was run
   because the first was neither trivially easy (NCM 0.445) nor broken (joint
   0.952).

## Block E — ImageNet-1K-fine-tuned checkpoint (`vit_base_patch16_224.augreg_in21k_ft_in1k`), frozen and + first-session adapters

Run 2026-09-14 20:20 → 2026-09-15 11:20 on the user's instruction to run E
after everything else. GPU: `gpu\run_blockE_gpu.ps1` (the three in1k caches
not made in battery 2 — ObjectNet, OmniBenchmark, VTAB — then Phase 1 on the
in1k checkpoint, same recipe as batteries 3/4, six datasets × seeds 2–4; 21/21
caches, no failures; log `gpu\out\extract_battery4_E.txt`, which also holds the
`cache_summary.py` output). A session restart on 2026-09-15 07:00 killed the
driver during OmniBenchmark seed 4 and two chains; the driver is idempotent and
was relaunched, the two chains were relaunched, nothing was lost or duplicated.
CPU: steps 1, 2, 5 only (baselines, contracted prototypes, contracted head), as
HANDOFF_GPU3 §6 specifies; 24 chains, no tracebacks. The ImageNet-R / ImageNet-A
/ CUB frozen-in1k rows are battery 2's (seeds 2–4), repeated here for the
comparison.

RanPAC "full" column: `ANCHORS_PTM.md` (RanPAC v3 Tab. 1, read from the PDF on
2026-09-06). **Two differences remain named**: RanPAC's `A_T` is the mean of
per-task accuracies after the last task, ours is overall last-step accuracy on
the full test set (B0 protocol; identical only when tasks and per-class test
counts are balanced); and RanPAC's Table 1 does not state its checkpoint per
column — its NCM-only row matches Zhou et al.'s IN1K block on ImageNet-R /
ImageNet-A / CUB and the IN21K values on CIFAR / OmniBenchmark / VTAB
(`RANPAC_VS_US.md`), so "same checkpoint family" below is established for the
first three datasets and assumed not to hold for OmniBenchmark / VTAB, where
the in21k rows of battery 3 are the like-for-like ones.

| dataset | config | seed | Phase-1 task-0 val acc | contracted protos strict / forced | head M=10k strict / forced / coverage / changes per 100 | growth strict (flips) | NCM | RanPAC full (A_T) |
|---|---|---|---|---|---|---|---|---|
| ImageNet-R | in1k frozen | 2 | — | 0.621 / 0.628 | 0.692 / 0.698 / 0.989 / 53 | 0.373 (0) | 0.628 | 0.779 |
| ImageNet-R | in1k frozen | 3 | — | 0.621 / 0.627 | 0.696 / 0.703 / 0.988 / 50 | 0.367 (0) | 0.627 | 0.779 |
| ImageNet-R | in1k frozen | 4 | — | 0.623 / 0.627 | 0.693 / 0.700 / 0.989 / 50 | 0.389 (0) | 0.627 | 0.779 |
| **ImageNet-R** | **in1k frozen** | mean | — | 0.622 / 0.628 | 0.693 / 0.700 / 0.988 / 51 | 0.377 (0) | 0.628 | 0.779 |
| ImageNet-R | in1k + adapters | 2 | 0.953 | 0.705 / 0.712 | 0.766 / 0.773 / 0.983 / 25 | 0.526 (0) | 0.712 | 0.779 |
| ImageNet-R | in1k + adapters | 3 | 0.958 | 0.702 / 0.710 | 0.770 / 0.778 / 0.981 / 29 | 0.554 (0) | 0.710 | 0.779 |
| ImageNet-R | in1k + adapters | 4 | 0.959 | 0.693 / 0.699 | 0.773 / 0.780 / 0.984 / 30 | 0.526 (0) | 0.699 | 0.779 |
| **ImageNet-R** | **in1k + adapters** | mean | 0.957 | 0.700 / 0.707 | **0.770 / 0.777** / 0.982 / 28 | 0.536 (0) | 0.707 | 0.779 |
| ImageNet-A | in1k frozen | 2 | — | 0.546 / 0.575 | 0.544 / 0.571 / 0.853 / 120 | 0.128 (0) | 0.575 | 0.624 |
| ImageNet-A | in1k frozen | 3 | — | 0.577 / 0.580 | 0.567 / 0.575 / 0.974 / 131 | 0.201 (0) | 0.580 | 0.624 |
| ImageNet-A | in1k frozen | 4 | — | 0.552 / 0.572 | 0.568 / 0.577 / 0.972 / 116 | 0.176 (0) | 0.572 | 0.624 |
| **ImageNet-A** | **in1k frozen** | mean | — | 0.558 / 0.576 | 0.560 / 0.575 / 0.933 / 122 | 0.169 (0) | 0.576 | 0.624 |
| ImageNet-A | in1k + adapters | 2 | 0.855 | 0.482 / 0.516 | 0.581 / 0.584 / 0.979 / 98 | 0.190 (0) | 0.516 | 0.624 |
| ImageNet-A | in1k + adapters | 3 | 0.962 | 0.508 / 0.554 | 0.604 / 0.612 / 0.976 / 94 | 0.218 (0) | 0.554 | 0.624 |
| ImageNet-A | in1k + adapters | 4 | 0.928 | 0.598 / 0.603 | 0.621 / 0.625 / 0.980 / 85 | 0.267 (0) | 0.603 | 0.624 |
| **ImageNet-A** | **in1k + adapters** | mean | 0.915 | 0.529 / 0.558 | **0.602 / 0.607** / 0.978 / 92 | 0.225 (0) | 0.558 | 0.624 |
| CUB-200 | in1k frozen | 2 | — | 0.841 / 0.859 | 0.869 / 0.880 / 0.985 / 13 | 0.547 (0) | 0.859 | 0.903 |
| CUB-200 | in1k frozen | 3 | — | 0.849 / 0.858 | 0.873 / 0.882 / 0.984 / 13 | 0.599 (0) | 0.858 | 0.903 |
| CUB-200 | in1k frozen | 4 | — | 0.856 / 0.860 | 0.883 / 0.888 / 0.995 / 13 | 0.562 (0) | 0.860 | 0.903 |
| **CUB-200** | **in1k frozen** | mean | — | 0.849 / 0.859 | 0.875 / 0.883 / 0.988 / 13 | 0.570 (0) | 0.859 | 0.903 |
| CUB-200 | in1k + adapters | 2 | 0.982 | 0.856 / 0.863 | 0.892 / 0.902 / 0.987 / 9 | 0.640 (0) | 0.863 | 0.903 |
| CUB-200 | in1k + adapters | 3 | 0.969 | 0.840 / 0.849 | 0.880 / 0.891 / 0.983 / 12 | 0.612 (0) | 0.849 | 0.903 |
| CUB-200 | in1k + adapters | 4 | 0.975 | 0.847 / 0.850 | 0.888 / 0.891 / 0.994 / 12 | 0.603 (0) | 0.850 | 0.903 |
| **CUB-200** | **in1k + adapters** | mean | 0.975 | 0.848 / 0.854 | **0.886 / 0.895** / 0.988 / 11 | 0.619 (0) | 0.854 | 0.903 |
| ObjectNet-200 | in1k frozen | 2 | — | 0.491 / 0.542 | 0.551 / 0.573 / 0.910 / 70 | 0.261 (0) | 0.542 | n.a. |
| ObjectNet-200 | in1k frozen | 3 | — | 0.511 / 0.543 | 0.555 / 0.572 / 0.905 / 58 | 0.308 (0) | 0.543 | n.a. |
| ObjectNet-200 | in1k frozen | 4 | — | 0.511 / 0.543 | 0.546 / 0.575 / 0.838 / 65 | 0.266 (0) | 0.543 | n.a. |
| **ObjectNet-200** | **in1k frozen** | mean | — | 0.504 / 0.543 | 0.551 / 0.574 / 0.884 / 64 | 0.278 (0) | 0.543 | n.a. |
| ObjectNet-200 | in1k + adapters | 2 | 0.878 | 0.516 / 0.553 | 0.630 / 0.637 / 0.975 / 40 | 0.385 (0) | 0.553 | n.a. |
| ObjectNet-200 | in1k + adapters | 3 | 0.892 | 0.512 / 0.549 | 0.627 / 0.634 / 0.969 / 38 | 0.398 (0) | 0.549 | n.a. |
| ObjectNet-200 | in1k + adapters | 4 | 0.842 | 0.536 / 0.559 | 0.635 / 0.642 / 0.978 / 41 | 0.388 (0) | 0.559 | n.a. |
| **ObjectNet-200** | **in1k + adapters** | mean | 0.870 | 0.521 / 0.554 | **0.631 / 0.637** / 0.974 / 39 | 0.390 (0) | 0.554 | n.a. |
| OmniBenchmark | in1k frozen | 2 | — | 0.722 / 0.729 | 0.761 / 0.768 / 0.977 / 19 | 0.646 (0) | 0.729 | 0.799 † |
| OmniBenchmark | in1k frozen | 3 | — | 0.723 / 0.731 | 0.764 / 0.773 / 0.975 / 19 | 0.661 (0) | 0.731 | 0.799 † |
| OmniBenchmark | in1k frozen | 4 | — | 0.722 / 0.729 | 0.761 / 0.767 / 0.978 / 18 | 0.649 (0) | 0.729 | 0.799 † |
| **OmniBenchmark** | **in1k frozen** | mean | — | 0.722 / 0.730 | 0.762 / 0.769 / 0.977 / 19 | 0.652 (0) | 0.730 | 0.799 † |
| OmniBenchmark | in1k + adapters | 2 | 0.968 | 0.708 / 0.717 | 0.765 / 0.775 / 0.975 / 18 | 0.662 (0) | 0.717 | 0.799 † |
| OmniBenchmark | in1k + adapters | 3 | 0.965 | 0.712 / 0.721 | 0.766 / 0.774 / 0.974 / 18 | 0.658 (0) | 0.721 | 0.799 † |
| OmniBenchmark | in1k + adapters | 4 | 0.916 | 0.702 / 0.709 | 0.760 / 0.767 / 0.976 / 18 | 0.666 (0) | 0.709 | 0.799 † |
| **OmniBenchmark** | **in1k + adapters** | mean | 0.950 | 0.707 / 0.716 | **0.764 / 0.772** / 0.975 / 18 | 0.662 (0) | 0.716 | 0.799 † |
| VTAB | in1k frozen | 2 | — | 0.886 / 0.890 | 0.932 / 0.936 / 0.992 / 10 | 0.496 (0) | 0.890 | 0.922 † |
| VTAB | in1k frozen | 3 | — | 0.883 / 0.890 | 0.931 / 0.938 / 0.983 / 11 | 0.464 (0) | 0.890 | 0.922 † |
| VTAB | in1k frozen | 4 | — | 0.886 / 0.888 | 0.933 / 0.937 / 0.992 / 9 | 0.537 (0) | 0.888 | 0.922 † |
| **VTAB** | **in1k frozen** | mean | — | 0.885 / 0.889 | 0.932 / 0.937 / 0.989 / 10 | 0.499 (0) | 0.889 | 0.922 † |
| VTAB | in1k + adapters | 2 | 1.000 | 0.862 / 0.868 | 0.939 / 0.941 / 0.995 / 8 | 0.546 (0) | 0.868 | 0.922 † |
| VTAB | in1k + adapters | 3 | 0.951 | 0.886 / 0.892 | 0.941 / 0.948 / 0.990 / 7 | 0.564 (0) | 0.892 | 0.922 † |
| VTAB | in1k + adapters | 4 | 0.980 | 0.866 / 0.873 | 0.937 / 0.943 / 0.990 / 7 | 0.551 (0) | 0.873 | 0.922 † |
| **VTAB** | **in1k + adapters** | mean | 0.977 | 0.871 / 0.878 | **0.939 / 0.944** / 0.992 / 7 | 0.554 (0) | 0.878 | 0.922 † |

† RanPAC's OmniBenchmark and VTAB columns are consistent with the **in21k**
checkpoint (`RANPAC_VS_US.md`); the like-for-like rows for those two are the
in21k + adapters rows of `PHASE1_RESULTS.md` / Block A (head 0.766 / 0.774 and
0.939 / 0.943), which the in1k rows here happen to equal within 0.002.

Growth flips 0 on 36/36 seeds; contracted-prototype forced = NCM on 36/36.
Joint NCM of the caches (`cache_summary.py`, feature quality only): in1k
frozen 0.611 / 0.513 / 0.855 / 0.525 / 0.730 / 0.848 (ImageNet-R / -A / CUB /
ObjectNet / OmniBenchmark / VTAB); in1k + adapters 0.710 · 0.705 · 0.703 /
0.506 · 0.536 · 0.598 / 0.861 · 0.851 · 0.848 / 0.552 · 0.550 · 0.559 / 0.716 ·
0.721 · 0.712 / 0.856 · 0.878 · 0.856 (seeds 2 · 3 · 4).

**Reading.** (i) With the checkpoint asterisk removed, the contracted head +
first-session adapters sits **0.2 under RanPAC full on ImageNet-R** (0.777 vs
0.779 forced; 0.770 strict), **1.7 under on ImageNet-A** (0.607 vs 0.624;
0.602 strict), **0.8 under on CUB** (0.895 vs 0.903; 0.886 strict) — the three
datasets where the checkpoint family is now the same — with the metric
difference (`A_T` vs B0 overall) still unresolved and worth a few tenths on
ImageNet-A's unbalanced test set. On OmniBenchmark (2.7 under, 0.772 vs 0.799)
and VTAB (2.2 above, 0.944 vs 0.922) the like-for-like rows are the in21k ones
and say the same thing. (ii) The 1k checkpoint alone (frozen, no adapters) is
worth +7.7 / +6.5 / −0.7 / 0.0 / −0.3 / −0.4 points on the prototypes
(ImageNet-R / -A / CUB / ObjectNet / OmniBenchmark / VTAB) versus the frozen
in21k caches of battery 1, and on the head (forced) +4.1 / +7.7 / +0.7 / +1.3 / +0.3 /
+0.3 — so outside ImageNet-R and ImageNet-A the checkpoint is worth less than
a point either way, and on ObjectNet the 1k head is markedly less stable under
growth (64 answer changes per 100 test points against 39 for the in21k head). (iii) Adapters on the 1k checkpoint move the
head by +7.7 / +3.2 / +1.2 / +6.3 / +0.3 / +0.7 but the prototypes by +7.9 /
−1.8 / −0.5 / +1.1 / −1.4 / −1.1: on four of six datasets first-session tuning
on the 1k checkpoint makes the prototype space slightly *worse* while the ridge
head still gains, the same divergence seen on OmniBenchmark in battery 3.
(iv) Per-seed spread of the adapter caches follows task-0 validation accuracy,
most visibly on ImageNet-A (val 0.855 / 0.962 / 0.928 → head 0.584 / 0.612 /
0.625) — the first task's class draw matters more than the checkpoint there.

## 7. Process notes and the one code change

- **Payload overlay.** Payload 3 (code + docs) was robocopied onto the
  battery-2/3 folder (`ASDplusContracts2`) excluding `results\`; the 14
  result files new in the payload were added; existing result files, caches,
  `gpu\venv` / `gpu\data` (junctions) untouched.
- **Code change (loader only): `gpu\extract_text.py`.** Under the installed
  `datasets` 5.0.1, `load_dataset("banking77")` / `("clinc_oos","plus")` fail
  twice: short repo ids are rejected ("Repository id must be 'namespace/name'"),
  and `PolyAI/banking77` is a script-based repo with no parquet conversion
  ("Dataset scripts are no longer supported"). Fix: CLINC loads from
  `clinc/clinc_oos` (parquet files on `main`); Banking77 reads the two CSVs the
  original `banking77.py` itself downloads
  (`raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/{train,test}.csv`)
  and maps categories with the script's `ClassLabel` names from the repo's
  `dataset_infos.json`, so texts, splits (10,003 / 3,080) and integer labels are
  identical to what `load_dataset` produced. Encoder, pooling, batch, inc
  untouched. Pristine copy `gpu\patches\extract_text.py.orig`, diff
  `gpu\patches\extract_text_hub_ids_2026-09-14.patch`.
- `setup_env_text.ps1` installed datasets 5.0.1, transformers 5.17.0, pandas 3.0.5.
- Launch details: all `.ps1` were run as child `powershell.exe -Command`
  processes (in-process, the first Python stderr line becomes a terminating
  error under `$ErrorActionPreference = "Stop"` on PowerShell 5.1);
  `launch_chains.ps1` was invoked with `OMP/MKL/OPENBLAS_NUM_THREADS = 2` in
  the environment (inherited by the chains; performance only) because the
  Phase-1 GPU runs are data-loader-bound and were starved by unbounded BLAS
  threads in battery 3. Cosmetic: `launch_chains.ps1` prints "seeds 2..22" for
  a 3-seed chain (string concatenation in its message); the Python command
  lines carry `--first-seed 2 --seeds 3` correctly (checked).
- Phase-1 wall time per (dataset, seed) was 14–96 min (ObjectNet the slowest),
  i.e. 5–10× the handoff estimate, as in battery 3 (JPEG decoding bound).
- **Incident (logs only, found while staging the send-back).** `run_phase1.ps1`
  names its logs `out\phase1_<name>_s<seed>.log/.err` without the backbone tag,
  so the Block E runs (`-Tag vitb1k`, seeds 2–4) overwrote the battery-3
  in21k Phase-1 training logs for `imr, ima, cub, obj, omni, vtab` seeds 2–4
  (`cifar100` was not in Block E; its logs are intact). The files now under
  those names are the **Block E (in1k) logs**. No cache, json, head, lifecycle
  or pkl file was touched (all carry the tag in their names); the battery-3
  driver log `out\run_phase1_box.log` survives, and the battery-3 logs were in
  the battery-3 send-back. If a re-run of Phase 1 on a second tag is ever
  planned, the log name in `run_phase1.ps1` should include `$Tag` (not
  changed here: no code change after the fact).

## 8. Files to send back (HANDOFF_GPU3 §8)

- caches: `results\cifar100_vitb_p1s{5,6,7}.npz`, `results\ptm_{imr,ima,cub,obj,omni,vtab}_vitb_p1s{5,6,7}.npz` (21);
  `results\ood_svhn_vitb.npz`, `ood_dtd_vitb.npz`, `ood_svhn_dinov2b.npz`, `cifar10_dinov2b.npz`;
  `results\ptm_banking77_mpnet.npz`, `ptm_clinc150_mpnet.npz`, `ood_clinc_oos_mpnet.npz`; `results\ptm_nslkdd_raw.npz`;
  Block E: `results\ptm_{obj,omni,vtab}_vitb1k.npz` (3), `results\ptm_{imr,ima,cub,obj,omni,vtab}_vitb1k_p1s{2,3,4}.npz` (18)
- results: for every `*_p1s5/6/7` cache and for `ptm_banking77_mpnet`, `ptm_clinc150_mpnet`, `ptm_nslkdd_raw`:
  `<cache>.json`, `<cache>_adol_protos.json`, `head_<cache>.json`, `lifecycle_<bench>[_s<seed>].json`, `adol_org_<tag>_<seed>.pkl`;
  Block E (steps 1, 2, 5 only): `ptm_*_vitb1k.json` (obj/omni/vtab), `ptm_*_vitb1k_p1s*.json` (18), `head_ptm_*_vitb1k*.json` (21 new + 3 from battery 2)
- `gpu\out\phase1_*_s5.log … _s7.log` (+ `.err`), `gpu\out\run_phase1_s567.log`, `gpu\out\cache_summary_p1s567.txt`,
  `gpu\out\extract_battery4.txt`, `gpu\out\extract_battery4_tabular.txt`, `gpu\out\extract_battery4_E.txt`
  (contains the Block E extract_suite output and the `cache_summary.py` table for all `*_vitb1k*` caches), `logs\chain_*`;
  Block E Phase-1 logs: `gpu\out\phase1_{imr,ima,cub,obj,omni,vtab}_s{2,3,4}.log/.err` (in1k runs; see the §7 incident about the name collision)
- Staged copy of everything above (relative paths preserved) in `..\sendback_battery4\`, with `SENDBACK_battery4_filelist.txt` (path, size, SHA-256)
- `BATTERY4_RESULTS.md` (this file); helpers `gpu\summarize_battery4.py`, `gpu\run_blockE_gpu.ps1`
- code change: `gpu\extract_text.py` with `gpu\patches\extract_text.py.orig` and `gpu\patches\extract_text_hub_ids_2026-09-14.patch`
