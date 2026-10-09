# gpu/ — feature extraction on a CUDA machine

The GPU is needed for one thing: extracting features from frozen pre-trained
encoders (and, for the first-session adapters, training those adapters on the
first task). Everything else — organisms, contracts, the gate, the lifecycle —
runs on a CPU and reads the `.npz` caches from `results/`.

The scripts here were written for Windows PowerShell on a machine with one
RTX 3070 (8 GB), and that is where every cache in this project was built. The
Python extractors themselves are plain argparse scripts and run anywhere with
CUDA torch. This folder is self-contained: copy it to the GPU machine, nothing
else from the project is needed there, and copy `out\*.npz` back into the
project's `results\` afterwards.

## Setup

Python 3.10+, an NVIDIA driver recent enough for CUDA 12.4 wheels (≥ 550),
about 15 GB of disk for weights, datasets and the environment, and internet on
the first run. In PowerShell, from this folder:

    Set-ExecutionPolicy -Scope Process Bypass
    .\setup_env.ps1          # venv\ with CUDA torch, torchvision, timm, plus the CPU pipeline's packages
    .\setup_env_text.ps1     # adds transformers, datasets and pandas (text and tabular extractors)

`setup_and_extract.ps1` is the original one-shot version: it builds a smaller
venv (CUDA torch + timm) and runs the four CIFAR extractions below. If torch
reports `cuda available: False`, the driver is too old for the cu124 wheels;
update it or change `cu124` to `cu121` in the setup script.

## Extractors and the caches they write

Every extractor is resumable (an existing output file is skipped) and stores
`n_classes` / `inc` metadata in the cache, which the CPU side reads.

| script | writes | what |
|---|---|---|
| `extract_gpu.py` (via `setup_and_extract.ps1`) | `cifar100_<tag>.npz`, `cifar10_<tag>.npz` | CIFAR through torchvision; `vitb_full` (ViT-B/16-in21k, 500 per class), `dinov2b`, `vitl` |
| `extract_ptm.py` (via `extract_suite.ps1`) | `ptm_<name>_<tag>.npz` | the six image-folder datasets below |
| `phase1_extract.py` (via `run_phase1.ps1`) | `<prefix>_<tag>_p1s<seed>.npz` | first-session adapters: rank-16 bottleneck adapters trained on that seed's first task only, then frozen, then the whole dataset extracted |
| `extract_ood.py` | `ood_<set>_<tag>.npz` | out-of-distribution test sets (SVHN, DTD; Places365 optional) for `exp_ood_ptm.py` |
| `extract_text.py` | `ptm_banking77_mpnet.npz`, `ptm_clinc150_mpnet.npz`, `ood_clinc_oos_mpnet.npz` | text streams on a frozen sentence encoder (all-mpnet-base-v2), mean-pooled |
| `extract_tabular.py` | `ptm_nslkdd_raw.npz` | NSL-KDD flow records as raw rows, with a fixed preprocessing decided before any run |

`run_blockE_gpu.ps1` chains the ImageNet-1K-checkpoint extractions and their
adapters. `cache_summary.py` prints shapes, class counts and a joint
nearest-class-mean accuracy for any cache, as a sanity check before analysis.

Note that `results\cifar100_vitb.npz`, the reference cache behind the CIFAR-100
numbers, is 250 images per class; `cifar100_vitb_full` is the 500-per-class
version.

### The suite datasets

The six datasets SimpleCIL / APER (Zhou et al. 2023) and RanPAC (McDonnell et
al. 2023) report with ViT-B/16-in21k, in their B0 protocols:

| folder under `gpu\data\` | dataset | classes | classes / task | tasks | train / test images (as distributed) |
|---|---|---|---|---|---|
| `imagenet-r`    | ImageNet-R      | 200 | 20 | 10 | 24,000 / 6,000 |
| `imagenet-a`    | ImageNet-A      | 200 | 20 | 10 | 5,981 / 1,519 |
| `cub`           | CUB-200-2011    | 200 | 20 | 10 | 9,430 / 2,358 |
| `objectnet`     | ObjectNet-200   | 200 | 20 | 10 | 26,509 / 6,628 |
| `omnibenchmark` | OmniBenchmark   | 300 | 30 | 10 | 89,697 / 5,985 |
| `vtab`          | VTAB (5 sets)   | 50  | 10 | 5  | 1,796 / 8,619 |

Each must be laid out as `<folder>\train\<class>\*` and `<folder>\test\<class>\*`
(the split the RevisitingCIL / LAMDA-PILOT authors distribute; their README
links the archives — download them yourself, and note that ObjectNet has its
own license). The extractor prints what it actually finds; a differently sized
copy still runs, but the comparison with published numbers is only valid on
the distributed split.

    .\extract_suite.ps1                                    # ViT-B/16-in21k -> out\ptm_<name>_vitb.npz
    .\extract_suite.ps1 -Backbone vit_base_patch14_dinov2.lvd142m -Tag dinov2b

## Analysis chains

Once a cache is in `results\`, the CPU analysis is the same for every cache.
`launch_chains.ps1` starts one detached chain per cache (and per seed block),
using `gpu\venv`, with logs in the project's `logs\`:

    .\launch_chains.ps1 -Caches ptm_imr_vitb                          # evaluation seeds 2-4, all five steps
    .\launch_chains.ps1 -Caches ptm_imr_vitb -Seeds 5,6,7 -OnePerSeed  # confirmation seeds, one chain each
    .\launch_chains.ps1 -Caches cifar100_vitb_p1s2 -PerCacheSeed       # a seed-specific adapter cache

The five steps, which can also be run by hand from the project root with any
environment built from `requirements.txt`:

    python exp_cifar.py --feats results/ptm_imr_vitb.npz --tag imr_vitb --prefix ptm --first-seed 2 --seeds 3                 # 1 baselines: growth, fine-tune, EWC, replay, NCM, joint
    python exp_cifar.py --feats results/ptm_imr_vitb.npz --tag imr_vitb --prefix ptm --first-seed 2 --seeds 3 --only-cncm     # 2 contracted prototypes
    python exp_cifar.py --feats results/ptm_imr_vitb.npz --tag imr_vitb --prefix ptm --adolescence --adol-mode protos --first-seed 2 --seeds 3   # 3 adolescence (caches the organism)
    python exp_lifecycle.py --benchmarks ptm_imr_vitb --arbiter gated --gate repaired --first-seed 2 --seeds 3 --out-name lifecycle_ptm_imr_vitb.json   # 4 certified lifecycle
    python exp_head.py --cache ptm_imr_vitb --M 2000 10000 --first-seed 2 --seeds 3                                             # 5 contracted head
    python claims.py

Any benchmark named `cifar_<tag>` or `ptm_<tag>` resolves to
`results/cifar100_<tag>.npz` or `results/ptm_<tag>.npz`; class count and
increment are read from the cache, and `--inc` overrides them.
`launch_ptm_chains.ps1` and `launch_battery23.ps1` are the earlier,
battery-specific versions of the same launcher.

Two PowerShell lessons are baked into these scripts. Launch long jobs with
`Start-Process` and separate stdout/stderr files. Never pipe a native
executable through `2>&1 | Tee-Object` under `$ErrorActionPreference = "Stop"`:
on Windows PowerShell 5.1 the first line torch writes to stderr becomes a
terminating error and the whole battery aborts.

## History

The batteries these scripts served, what was asked of the GPU machine and what
came back are in [docs/](../docs/README.md): the instructions (`HANDOFF_GPU*.md`),
the machine's reports, and the code fixes it found (`GPU_BOX_FIXES.md`, with
the two diffs in [`patches/`](patches/)). `summarize_phase1.py` and
`summarize_battery4.py` rebuild those reports' tables from `results/`.
