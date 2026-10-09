# Handoff: the pretrained-model class-IL suite on the GPU box

*Operational history. The instructions this analysis machine sent the GPU box for battery 1 (the six-dataset pretrained-model suite, 2026-09-06). The battery ran; its results are `docs/PTM_RESULTS.md` and the `B-ptm` section of [REPORT.md](../REPORT.md). Kept because the protocol it fixes is the protocol every later battery inherited.*

*Written 2026-09-06 by the Claude session on the analysis machine for the Claude
session on the RTX 3070 box. You are receiving a self-contained payload of the
"contracted growth" project (folder `ASDplusContracts`). Your job is one
experiment battery: extract frozen ViT-B/16-in21k features for six standard
class-incremental datasets, run the existing certified pipeline on them
unchanged, aggregate, and hand the result files back. Read this file fully
before touching anything. `APPROACH.md` explains the project in plain English
if you need context; `REPORT.md` sections "B-vit" and "B-backbones" are the two
results this battery extends.*

## 1. What this is about (two minutes)

The project grows an **organism**: a cascade of frozen **organs**, each a small
prototype classifier wrapped in a **ball contract** (anchors, margin, radius,
PCA territory) that says which inputs it may answer. Growth adds organs at
abstention and never edits old ones, so forgetting is exactly zero, checked
bitwise. **Adolescence** consolidates the organs into one contracted prototype
organ. An **acceptance gate** (territory check + McNemar escrow on the
validation split + the original contract rule) decides whether a proposed
change to the organism is accepted; the whole thing is "certified" when every
answer comes from inside a contract.

On CIFAR-100 with a frozen ViT-B/16-in21k backbone the certified contracted
prototypes score 0.835 strict / 0.844 forced, which is +2.2 over SimpleCIL's
last-step number and -8.7 under RanPAC's 0.922. On 2026-09-06 we showed on
three more backbones (ViT-B full data, DINOv2-B, ViT-L) that the contracted
prototypes' forced accuracy equals nearest-class-mean on the same features to
the third decimal: the gap to RanPAC is its trained random-projection head,
not the features.

**This battery** asks whether that picture holds on the other five datasets
those papers report on with the *same* backbone (ImageNet-R, ImageNet-A,
CUB-200, ObjectNet-200, OmniBenchmark) plus VTAB. Published anchors exist for
all of them, so this turns one anchored comparison into seven.

## 2. Ground rules (not negotiable; they are why the numbers are trustworthy)

1. **One interpreter.** Every Python command uses the venv you build in step 3:
   `gpu\venv\Scripts\python.exe`. `cgrow/__init__.py` refuses the conda *base*
   interpreter; if you ever see that RuntimeError, you launched the wrong
   Python. Never set `CGROW_ALLOW_ANY_PYTHON`.
2. **No tuning.** Do not change `CFG`, `CIFAR_CFG`, gate thresholds, escrow
   settings, seeds, class-order rule, or any hyper-parameter. The pipeline is
   frozen; the whole point is running it unchanged on new data. If something
   looks wrong, report it; do not fix it by tuning.
3. **Evaluation seeds only: `--first-seed 2 --seeds 3`** (seeds 2, 3, 4).
   Seeds 0-1 are development seeds and 5-7 are reserved gate-confirmation
   seeds; do not run them here.
4. **No published numbers from memory.** `ANCHORS_PTM.md` has blank cells on
   purpose. Fill a cell only from the source paper's table (arXiv PDF), and
   record table number, metric (last-step vs. average), backbone checkpoint and
   protocol in the same row. Last-step accuracy is the only column comparable
   to ours. If you cannot open the source, leave the cell blank.
5. **Do not write into or delete existing result files.** New files only:
   `results\ptm_*`, `results\lifecycle_ptm_*`, `results\adol_org_<tag>_<seed>.pkl`,
   `logs\*`. Every `results\*.json` in the payload is a reference copy from
   the analysis machine (the CIFAR and MLP results `claims.py` aggregates; the
   heavy `.npz` caches and `.pkl` organisms were left out). `claims.py` needs
   them present and may regenerate `claims.json` here for a look, but the
   canonical `claims.json` is rebuilt on the analysis machine after your files
   are copied back.
6. **Report outcomes faithfully.** A refused gate, a crashed seed, a dataset
   with the wrong image count: say so with the numbers. Nothing here is a
   failure to hide; a refusal is a result.
7. **Launching long jobs on Windows.** Use PowerShell `Start-Process` with the
   venv Python and `-u` (unbuffered) so logs stream; see step 6. The venv's
   `python.exe` is a launcher stub, so the working process shows the *base*
   interpreter's path in Task Manager: that is normal. Do not use `nohup` from
   Git Bash (it picks up whatever `python` is on PATH). Heredocs in Git Bash
   mangle backslashes on this setup; write helper scripts with a file-writing
   tool instead.

## 3. Environment (once)

```powershell
cd <payload>\ASDplusContracts\gpu
Set-ExecutionPolicy -Scope Process Bypass
.\setup_env.ps1
```

Expected last lines: `cuda: True NVIDIA GeForce RTX 3070` and
`cgrow guard OK under ...\gpu\venv\Scripts\python.exe`. If CUDA is False, stop
and tell the user (driver too old for the cu124 wheels, or wrong Python).
Define once per shell:

```powershell
$PY = "<payload>\ASDplusContracts\gpu\venv\Scripts\python.exe"
$W  = "<payload>\ASDplusContracts"
```

## 4. Datasets (with the user; you cannot download these yourself)

Layout required, one folder per dataset under `gpu\data\`:

```
gpu\data\<folder>\train\<class_name>\*.jpg
gpu\data\<folder>\test\<class_name>\*.jpg
```

| folder | dataset | classes | inc | tasks | train / test images in the distributed split |
|---|---|---|---|---|---|
| `imagenet-r` | ImageNet-R | 200 | 20 | 10 | 24,000 / 6,000 |
| `imagenet-a` | ImageNet-A | 200 | 20 | 10 | 5,981 / 1,519 |
| `cub` | CUB-200-2011 | 200 | 20 | 10 | 9,430 / 2,358 |
| `objectnet` | ObjectNet-200 | 200 | 20 | 10 | 26,509 / 6,628 |
| `omnibenchmark` | OmniBenchmark | 300 | 30 | 10 | 89,697 / 5,985 |
| `vtab` | VTAB (5 sub-datasets) | 50 | 10 | 5 | 1,796 / 8,619 |

The splits are the ones distributed with the RevisitingCIL / LAMDA-PILOT code
(Zhou et al.); their GitHub README links the archives. Ask the user to download
them (ObjectNet has its own license) and unpack into the folders above. Some
archives unpack as `train/` `test/` directly, others one level deeper; the
requirement is exactly `<folder>\train` and `<folder>\test` containing class
sub-folders with identical names in both. Check counts before extracting:

```powershell
foreach ($d in Get-ChildItem "$W\gpu\data" -Directory) { foreach ($s in "train","test") { $p="$($d.FullName)\$s"; if (Test-Path $p) { $c=(Get-ChildItem $p -Directory).Count; $n=(Get-ChildItem $p -Recurse -File).Count; "$($d.Name)\$s  classes=$c  images=$n" } else { "$($d.Name)\$s  MISSING" } } }
```

If a count differs from the table, tell the user before extracting: the anchor
comparison is only valid on the distributed split. A missing dataset is fine;
the extractor skips it and the battery runs on whatever is present.

## 5. Extraction (GPU, 5-40 minutes per dataset)

```powershell
cd "$W\gpu"; .\extract_suite.ps1
& $PY cache_summary.py out\ptm_*.npz
```

Outputs `gpu\out\ptm_<name>_vitb.npz` with metadata (`n_classes`, `inc`,
`dataset`, `backbone`, `classes`). The summary prints shapes, per-class counts
and a joint nearest-class-mean accuracy (a feature-quality number only, not a
continual-learning result). Record that table in your notes. Then:

```powershell
Copy-Item "$W\gpu\out\ptm_*.npz" "$W\results\"
```

Optional second pass later, only if asked: `.\extract_suite.ps1 -Backbone
vit_base_patch14_dinov2.lvd142m -Tag dinov2b`.

## 6. Analysis (CPU; run the datasets in parallel, one job chain per dataset)

Per cache `ptm_<name>_vitb.npz` the chain is four steps that must run in this
order (the third caches the organisms the fourth reuses):

1. baseline run: growth organism, fine-tune, EWC, replay, NCM, joint
2. `--only-cncm`: the contracted prototypes (the headline configuration)
3. `--adolescence --adol-mode protos`: consolidation, caches `adol_org_<name>_vitb_<seed>.pkl`
4. certified lifecycle: `exp_lifecycle.py --benchmarks ptm_<name>_vitb --arbiter gated --gate repaired`

Launch one chain per dataset, detached, logging to `logs\`:

```powershell
New-Item -ItemType Directory -Force "$W\logs" | Out-Null
foreach ($name in @("imr","ima","cub","obj","omni","vtab")) {
  $tag = "${name}_vitb"
  if (-not (Test-Path "$W\results\ptm_$tag.npz")) { "skip $tag (no cache)"; continue }
  $cmd = "Set-Location '$W'; " +
    "& '$PY' -u exp_cifar.py --feats results\ptm_$tag.npz --tag $tag --prefix ptm --first-seed 2 --seeds 3; " +
    "& '$PY' -u exp_cifar.py --feats results\ptm_$tag.npz --tag $tag --prefix ptm --first-seed 2 --seeds 3 --only-cncm; " +
    "& '$PY' -u exp_cifar.py --feats results\ptm_$tag.npz --tag $tag --prefix ptm --adolescence --adol-mode protos --first-seed 2 --seeds 3; " +
    "& '$PY' -u exp_lifecycle.py --benchmarks ptm_$tag --arbiter gated --gate repaired --first-seed 2 --seeds 3 --out-name lifecycle_ptm_$tag.json"
  Start-Process powershell.exe -ArgumentList @("-NoProfile","-Command",$cmd) -WorkingDirectory $W `
    -RedirectStandardOutput "$W\logs\ptm_$tag.log" -RedirectStandardError "$W\logs\ptm_$tag.err" -WindowStyle Hidden
  "launched $tag"
}
```

Each job uses about one CPU core and 1-2 GB RAM. Durations scale with train
size: CIFAR (45k x 768) took ~25 min per seed for step 1 and ~25 min per seed
for step 3 on a 20-core machine; expect ImageNet-A / CUB / VTAB to be quick,
ImageNet-R / ObjectNet similar to CIFAR, OmniBenchmark roughly twice that.
Check progress with `Get-Content logs\ptm_<tag>.log -Tail 5`; a non-empty
`.err` file with a traceback means that chain died: read it, report it, and do
not restart with changed settings.

Each chain produces `results\ptm_<tag>.json` (with `contracted_ncm` after step
2), `results\ptm_<tag>_adol_protos.json`, `results\adol_org_<tag>_{2,3,4}.pkl`,
`results\lifecycle_ptm_<tag>.json`.

## 7. Aggregate and write up

```powershell
& $PY claims.py
& $PY manifest.py
```

`claims.json` gains `B_ptm[<name>_vitb]` with: growth strict/forced and
flips, fine-tune / EWC / replay / NCM / joint, `contracted_ncm_strict` and
`_forced`, `adol_accepted`, `adol_after_strict`, and a `lifecycle` block
(`certified_final`, `free_arbiter`, `arbiter_accepted`, `uncertified_fraction_max`,
`escrow_z`). Growth flips must be 0 and `uncertified_fraction_max` must be
0.0 everywhere; if not, that is the first thing to report.

Write `PTM_RESULTS.md` at the project root with one row per dataset:

| dataset | classes x inc | growth | adolescence | certified lifecycle (arbiter accepted /3) | free arbiter [uncertified] | contracted protos strict / forced | NCM same feats | replay | joint | published last-step (from ANCHORS_PTM.md, or "unverified") |

and three sentences under it: (a) does contracted-prototypes *forced* equal NCM
on every dataset (the CIFAR finding), (b) where does the certified lifecycle
sit relative to the prototypes, (c) gate behaviour (how many arbiter
acceptances, any accepted regression, i.e. `pfc < adolescence` with
`pfc_accepted` true). State plainly anything that did not run.

Then fill `ANCHORS_PTM.md` from the source PDFs per rule 4, marking each cell
with date and table. Do not compute "vs. ours" deltas against unverified cells.

## 8. What to send back to the analysis machine

Copy these into the analysis machine's `ASDplusContracts\results\` (caches
first: they are the expensive part) and the two markdown files to the root:

- `results\ptm_*_vitb.npz`
- `results\ptm_*_vitb.json`, `results\ptm_*_vitb_adol_protos.json`
- `results\adol_org_*_vitb_*.pkl`
- `results\lifecycle_ptm_*.json`
- `logs\ptm_*.log`, `logs\ptm_*.err`
- `PTM_RESULTS.md`, `ANCHORS_PTM.md` (filled)

Do not send back `claims.json`, `MANIFEST.json`, or anything else under
`results\`; those are regenerated there.

## 8b. Battery 2 (2026-09-08): the ImageNet-1K-fine-tuned checkpoint on three datasets

RanPAC's "NCM only" row matches Zhou et al.'s **IN1K** SimpleCIL numbers on
ImageNet-R, ImageNet-A and CUB (61.28 / 49.44 / 85.16), so those three RanPAC
anchors are on the ImageNet-1K-fine-tuned ViT-B/16, not the in21k checkpoint
we used. Extract those three caches on the other checkpoint and run the same
chain; nothing else changes. The datasets are already in `gpu\data`.

```powershell
cd "$W\gpu"
.\extract_suite.ps1 -Backbone vit_base_patch16_224.augreg_in21k_ft_in1k -Tag vitb1k -Names imr,ima,cub
& $PY cache_summary.py out\ptm_*_vitb1k.npz
Copy-Item "$W\gpu\out\ptm_*_vitb1k.npz" "$W\results\"
```

Then section 6 with `$tag = "${name}_vitb1k"` for `imr`, `ima`, `cub` (three
chains), section 7, and send back the same file set as section 8 plus
`results\head_ptm_*_vitb1k.json` if you also run
`exp_head.py --cache ptm_<name>_vitb1k --M 2000 10000 --first-seed 2 --seeds 3`
(the contracted head; fast, CPU). Expected check: the in1k caches' joint NCM in
`cache_summary.py` should be near 0.61 (ImageNet-R), 0.49 (ImageNet-A) and 0.85
(CUB), the values RanPAC prints for "NCM only". Anchors do not change; the
comparison row for these three datasets gains a same-checkpoint entry.

## 9. Troubleshooting

- `RuntimeError: cgrow: refusing to run under the conda BASE interpreter`: you
  used the wrong Python. Re-launch with `$PY`.
- `missing <root>/train`: dataset folder layout is one level off; move the
  `train`/`test` folders up so they sit directly under `gpu\data\<folder>`.
- `train/test class folders differ`: the two splits do not have the same class
  names; the archive is incomplete or mixed. Report to the user.
- CUDA out of memory during extraction: add `--batch 32` to the failing line
  in `extract_suite.ps1` (edit the `& $PY ... extract_ptm.py` call).
- A lifecycle run prints `pfc REJ`: that is the gate refusing the arbiter, a
  normal outcome (it refused 8 of 9 times on the strong-backbone CIFAR caches).
  Record it; do not change the gate.
- `ImportError: sheet benchmarks need ASDbrain`: you asked for a sheet
  benchmark; those are not part of this battery and need a sibling project.
- Logs empty while a job runs: you forgot `-u`; the output arrives at exit.
