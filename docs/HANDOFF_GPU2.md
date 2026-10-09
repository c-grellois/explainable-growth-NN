# Handoff 2: the ImageNet-1K checkpoint, and first-session tuning (Phase 1)

*Operational history. Battery 2 and 3 instructions (the ImageNet-1K checkpoint and first-session adapters, 2026-09-08). Results: `docs/PHASE1_RESULTS.md` and the `B-phase1` / `B-in1k` sections of [REPORT.md](../REPORT.md).*

*Written 2026-09-08 on the analysis machine for the Claude session on the RTX
3070 box. This is the second battery. Everything in `HANDOFF_GPU.md` still
applies (sections 2 and 9 especially: one interpreter, no tuning of the CPU
pipeline, evaluation seeds only, no published numbers from memory, never write
into existing result files, report refusals as results). Read that file first
if this is a fresh session; then this one. The payload zip you received is the
current project state: it already contains every script named below and the
reference result files.*

## 1. Why these two batteries

The comparison table (`LITERATURE_COMPARISON.md`) now has three of our
configurations against SimpleCIL, L2P, DualPrompt, APER and RanPAC on the same
frozen ViT-B/16-in21k features. Two confounds remain, and both are GPU work:

- **Battery 2, checkpoint.** RanPAC's published ImageNet-R, ImageNet-A and
  CUB numbers are on the ImageNet-1K-fine-tuned ViT-B/16 (its "NCM only" row
  matches Zhou et al.'s IN1K SimpleCIL block: 61.28 / 49.44 / 85.16). Ours are
  in21k. Extract those three datasets on the other checkpoint.
- **Battery 3, Phase 1.** Every published method that beats us on the same
  checkpoint (APER on ImageNet-R / ObjectNet, RanPAC full on CIFAR /
  OmniBenchmark, L2P / DualPrompt on ImageNet-R) tunes the backbone once on the
  first task with a parameter-efficient module, then freezes it. In this
  project's terms it is a birth step (organ 0 also shapes the substrate, once),
  so it is compatible with the doctrine. `phase1_extract.py` does it: parallel
  bottleneck adapters (r = 16) on every block's MLP, trained with a cosine
  classifier on the first task's classes for 20 epochs, then frozen, then
  features for the whole dataset are extracted. Everything downstream (contracted
  prototypes, contracted head, certified lifecycle) then runs unchanged on the
  new caches, so the effect of Phase 1 is isolated.

## 2. Environment and data

Same venv (`gpu\setup_env.ps1`, already built) and the same six dataset folders
under `gpu\data` from battery 1, plus CIFAR-100 which `torchvision` downloads
into `gpu\data` on first use. Nothing new to download.

```powershell
$PY = "<payload>\ASDplusContracts\gpu\venv\Scripts\python.exe"
$W  = "<payload>\ASDplusContracts"
cd "$W\gpu"
```

## 3. Battery 2: in1k caches for ImageNet-R, ImageNet-A, CUB (≈ 15 min)

```powershell
.\extract_suite.ps1 -Backbone vit_base_patch16_224.augreg_in21k_ft_in1k -Tag vitb1k -Names imr,ima,cub
& $PY cache_summary.py out\ptm_*_vitb1k.npz
Copy-Item "$W\gpu\out\ptm_*_vitb1k.npz" "$W\results\"
```

Check: the joint NCM printed by `cache_summary.py` should be near 0.61 (ImageNet-R),
0.49 (ImageNet-A) and 0.85 (CUB), the values RanPAC prints for "NCM only". If it
is not, say so; do not proceed to compare against RanPAC on that dataset.

## 4. Battery 3: Phase 1 caches (≈ 2–3 h for 7 datasets × 3 seeds)

```powershell
.\run_phase1.ps1                       # seeds 2,3,4; CIFAR-100 + every suite dataset present
& $PY cache_summary.py out\*_p1s*.npz
Copy-Item "$W\gpu\out\*_p1s*.npz" "$W\results\"
```

Per (dataset, seed) the script prints the first task's classes, the number of
Phase-1 training images, per-epoch loss and accuracy, and the task-0 validation
accuracy of the adapters. Record those. A cache is **seed-specific**: it was
tuned on that seed's first task, so on the CPU side it must be run with
`--first-seed <seed> --seeds 1` only. Expected orders of magnitude: 1–4 min of
tuning per (dataset, seed) except OmniBenchmark (≈ 15–20 min), plus extraction
(CIFAR ≈ 3 min, OmniBenchmark ≈ 5 min).

If the GPU runs out of memory, add `--batch 32` to the `phase1_extract.py` call
in `run_phase1.ps1`. If a run prints `WARNING: no CUDA`, stop: wrong Python.

## 5. Analysis on the new caches (CPU; run on the box, in parallel)

Three configurations per cache, exactly as in battery 1 plus the head. Use the
launcher pattern from `HANDOFF_GPU.md` section 6 (PowerShell `Start-Process`,
`-u`, logs in `logs\`). Below, `<c>` is a cache stem and `<s>` its seed.

**Battery 2 caches** (`ptm_imr_vitb1k`, `ptm_ima_vitb1k`, `ptm_cub_vitb1k`;
seeds 2–4 in one chain each, as in battery 1):

```powershell
& $PY -u exp_cifar.py --feats results\<c>.npz --tag <name>_vitb1k --prefix ptm --first-seed 2 --seeds 3
& $PY -u exp_cifar.py --feats results\<c>.npz --tag <name>_vitb1k --prefix ptm --first-seed 2 --seeds 3 --only-cncm
& $PY -u exp_head.py --cache <c> --M 2000 10000 --first-seed 2 --seeds 3
& $PY -u exp_cifar.py --feats results\<c>.npz --tag <name>_vitb1k --prefix ptm --adolescence --adol-mode protos --first-seed 2 --seeds 3
& $PY -u exp_lifecycle.py --benchmarks ptm_<name>_vitb1k --arbiter gated --gate repaired --first-seed 2 --seeds 3 --out-name lifecycle_ptm_<name>_vitb1k.json
```

**Battery 3 caches** (one chain per cache, ONE seed each):

```powershell
# CIFAR-100, seed s:
& $PY -u exp_cifar.py --feats results\cifar100_vitb_p1s<s>.npz --tag vitb_p1s<s> --first-seed <s> --seeds 1
& $PY -u exp_cifar.py --feats results\cifar100_vitb_p1s<s>.npz --tag vitb_p1s<s> --first-seed <s> --seeds 1 --only-cncm
& $PY -u exp_head.py --cache cifar100_vitb_p1s<s> --M 2000 10000 --first-seed <s> --seeds 1
& $PY -u exp_cifar.py --feats results\cifar100_vitb_p1s<s>.npz --tag vitb_p1s<s> --adolescence --adol-mode protos --first-seed <s> --seeds 1
& $PY -u exp_lifecycle.py --benchmarks cifar_vitb_p1s<s> --arbiter gated --gate repaired --first-seed <s> --seeds 1 --out-name lifecycle_cifar_vitb_p1s<s>.json
# suite dataset <name>, seed s:
& $PY -u exp_cifar.py --feats results\ptm_<name>_vitb_p1s<s>.npz --tag <name>_vitb_p1s<s> --prefix ptm --first-seed <s> --seeds 1
& $PY -u exp_cifar.py --feats results\ptm_<name>_vitb_p1s<s>.npz --tag <name>_vitb_p1s<s> --prefix ptm --first-seed <s> --seeds 1 --only-cncm
& $PY -u exp_head.py --cache ptm_<name>_vitb_p1s<s> --M 2000 10000 --first-seed <s> --seeds 1
& $PY -u exp_cifar.py --feats results\ptm_<name>_vitb_p1s<s>.npz --tag <name>_vitb_p1s<s> --prefix ptm --adolescence --adol-mode protos --first-seed <s> --seeds 1
& $PY -u exp_lifecycle.py --benchmarks ptm_<name>_vitb_p1s<s> --arbiter gated --gate repaired --first-seed <s> --seeds 1 --out-name lifecycle_ptm_<name>_vitb_p1s<s>.json
```

Priority if time is short: `--only-cncm` (contracted prototypes) and
`exp_head.py` first on every cache (minutes each); adolescence and lifecycle
(the slow steps) afterwards. The head with `--M 10000` needs about 1 GB per
process; run at most eight in parallel.

## 6. Write-up on the box: `PHASE1_RESULTS.md`

One table per battery, one row per (dataset, seed) and a mean row per dataset:

| dataset | seed | Phase-1 task-0 val acc | tuned params | contracted protos strict / forced | contracted head (M=10k) strict / forced / coverage / changes per 100 | growth | adolescence | certified lifecycle | free arbiter |

plus, for battery 2, the joint NCM of each in1k cache next to RanPAC's NCM-only
value (0.612 / 0.493 / 0.851). Then four sentences, stated plainly with the
numbers: (a) does Phase 1 move contracted prototypes and by how much per
dataset (the reference in21k values are in `BENCHMARKS.md` section A);
(b) does it move the contracted head, and how does the head-after-Phase-1
compare to RanPAC's *full* row (0.922 / 0.779 / 0.624 / 0.903 / 0.799 / 0.922;
the ImageNet-R / ImageNet-A / CUB cells are 1k-checkpoint numbers, compare those
to the battery-2 + Phase-1 caches only if you also ran Phase 1 on the 1k caches,
otherwise say so); (c) growth flips and uncertified fraction (must be 0 and
0.0); (d) anything that did not run. Do not compute deltas against published
numbers whose protocol or checkpoint differs; name the difference instead.

## 7. What to send back

- `results\ptm_*_vitb1k.npz`, `results\*_p1s*.npz` (caches; the expensive part)
- every new `results\*.json` and `results\*.pkl` your runs produced (`ptm_*_vitb1k*`,
  `*_p1s*`, `lifecycle_*_vitb1k*`, `lifecycle_*_p1s*`, `head_*_vitb1k*`, `head_*_p1s*`,
  `adol_org_*_vitb1k_*`, `adol_org_*_p1s*_*`)
- `gpu\out\phase1_*.log`, `gpu\out\cache_summary*.txt` if you saved one, `logs\*`
- `PHASE1_RESULTS.md`
- if you changed any code, the diff and pristine copies as in `GPU_BOX_FIXES.md`

Not `claims.json` or `MANIFEST.json`; those are regenerated on the analysis machine.

## 8. Things that would be wrong to do

- Tuning Phase 1 hyper-parameters against test accuracy, or against any seed
  other than the one being run. The defaults (r = 16, 20 epochs, AdamW 1e-3,
  cosine) are fixed in advance; if they fail visibly (loss not decreasing,
  task-0 val accuracy below the frozen backbone's), report it and stop.
- Using a `_p1s<s>` cache with a different seed.
- Comparing a Phase-1 cache's numbers to the in21k anchors as if nothing
  changed: they are a new row ("+ first-session adapters"), directly comparable
  to APER-adapter and RanPAC full, and no longer to SimpleCIL.
- Reading the CPU pipeline's validation split into Phase 1: the script already
  excludes those rows; do not change that.
