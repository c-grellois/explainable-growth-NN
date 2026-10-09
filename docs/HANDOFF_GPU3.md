# Handoff 3: confirmation seeds, far-OOD sets, a text substrate, a tabular substrate

*Operational history. Battery 4 instructions (confirmation seeds 5-7, far-OOD caches, a text substrate, a tabular substrate, 2026-09-14). Results: `docs/BATTERY4_RESULTS.md` and the `B-ptm-confirm` / `B-text` / `B-tabular` / `B-in1k-phase1` sections of [REPORT.md](../REPORT.md).*

*Written 2026-09-14 on the analysis machine for the Claude session on the RTX
3070 box. This is the fourth battery. `HANDOFF_GPU.md` sections 2 and 9 still
apply word for word (one interpreter; evaluation seeds only; nothing tuned
against test; no published number quoted from memory; never write into an
existing result file; a refusal is a result). Read that file first if this is a
fresh session, then this one. The payload zip overlays the project folder you
already have: it carries code and documents only, no caches, so unpack it ON
TOP of your existing copy (your `results\*.npz`, `results\*.pkl`, `gpu\data`,
`gpu\venv` and `gpu\out` stay where they are).*

## 0. What is new in the code (read before running)

- `exp_cifar.py`, `exp_head.py`, `exp_lifecycle.py` take `--inc <k>` (a longer
  stream on the same cache; result and organism files carry `_inc<k>`). You
  will not need it here; it is the analysis machine's B-horizon battery.
- The synthetic pools of the consolidation and the arbiter are bounded by a
  total budget equal to the inc-10 values (10 × 4,000 and 10 × 2,500), so
  every run with ≤ 10 organs is bit-for-bit what it was.
- `exp_head.py` has `return_model=True` for the OOD script; `exp_ood_ptm.py`
  and `exp_cost.py` are new (CPU, analysis machine).
- `gpu\extract_ood.py`, `gpu\extract_text.py`, `gpu\extract_tabular.py`,
  `gpu\launch_chains.ps1`, `gpu\setup_env_text.ps1` are new and are what you
  run. `gpu\run_phase1.ps1` is unchanged and already takes `-Seeds`.

## 1. Order of work, and what each block is for

| block | GPU time | CPU time | purpose |
|---|---|---|---|
| A. first-session-adapter caches, seeds 5–7, seven datasets | ≈ 2–3 h | ≈ 6–10 h of chains in parallel | the headline table (contracted head + adapters, `B_phase1`) gets its own gate-confirmation block, like the MLP benchmarks already have |
| B. far-OOD caches: SVHN, DTD on ViT-B; CIFAR-10 on DINOv2-B | ≈ 15 min | none | the analysis machine measures how the certificate declines them (B-ood) |
| C. text substrate: Banking77, CLINC150 on a frozen sentence encoder | ≈ 10 min | ≈ 1–2 h | first non-vision stream; the standard chain unchanged |
| D. tabular substrate: an intrusion-detection stream, raw rows | none (CPU) | ≈ 1 h | "new attack categories appear" — the partner document's use case, first pass |
| E. optional: ImageNet-1K checkpoint + adapters, all six RanPAC datasets | ≈ 3 h | ≈ 6 h | removes the last checkpoint asterisk on the RanPAC column |

Do A first (start it, it runs unattended), then B and C while A extracts,
then D. E only if everything else is back and time remains.

```powershell
$PY = "<payload>\ASDplusContracts\gpu\venv\Scripts\python.exe"
$W  = "<payload>\ASDplusContracts"
cd "$W\gpu"
Set-ExecutionPolicy -Scope Process Bypass
.\setup_env_text.ps1          # adds datasets, transformers, pandas to the venv (once)
```

## 2. Block A — adapter caches for confirmation seeds 5, 6, 7

```powershell
.\run_phase1.ps1 -Seeds 5,6,7                      # all seven datasets present under data\
& $PY cache_summary.py out\*_p1s5.npz out\*_p1s6.npz out\*_p1s7.npz
Copy-Item "$W\gpu\out\*_p1s5.npz","$W\gpu\out\*_p1s6.npz","$W\gpu\out\*_p1s7.npz" "$W\results\"
```

Same expectations as battery 3 (HANDOFF_GPU2.md §4): task-0 validation
accuracy printed per (dataset, seed), 1–4 min tuning each except OmniBenchmark
(15–20 min). The class order for seed s is `default_rng(s + 400)`; the cache
is seed-specific and is used with that seed only.

Then the standard chain, one seed per cache, all five steps:

```powershell
$p1 = Get-ChildItem "$W\results\*_p1s5.npz","$W\results\*_p1s6.npz","$W\results\*_p1s7.npz" | ForEach-Object { $_.BaseName }
.\launch_chains.ps1 -Caches ($p1 -join ",") -PerCacheSeed -Steps 1,2,3,4,5
```

That is 21 chains (7 datasets × 3 seeds); each uses about one core and 1–2 GB
(the M = 10,000 head briefly ~1 GB more). If the box has fewer than ~24 GB free,
launch in two waves (seeds 5–6 first). Check with
`Get-Content logs\chain_<cache>_s<seed>.log -Tail 3`; a non-empty `.err` is a
dead chain — read it, report it, do not restart with changed settings.

Outputs per cache `<c>` (seed s): `results\<prefix>_<tag>.json`
(+ `contracted_ncm`), `results\<prefix>_<tag>_adol_protos.json`,
`results\adol_org_<tag>_<s>.pkl`, `results\lifecycle_<bench>_s<s>.json`,
`results\head_<c>.json`.

## 3. Block B — far-OOD caches (15 minutes)

```powershell
& $PY extract_ood.py --dataset svhn --out out\ood_svhn_vitb.npz
& $PY extract_ood.py --dataset dtd  --out out\ood_dtd_vitb.npz
& $PY extract_gpu.py --backbone vit_base_patch14_dinov2.lvd142m --dataset cifar10 --per-class 500 --out out\cifar10_dinov2b.npz
& $PY extract_ood.py --dataset svhn --backbone vit_base_patch14_dinov2.lvd142m --out out\ood_svhn_dinov2b.npz
Copy-Item "$W\gpu\out\ood_*.npz","$W\gpu\out\cifar10_dinov2b.npz" "$W\results\"
```

torchvision downloads SVHN (≈ 65 MB) and DTD (≈ 600 MB) into `data\`. No
analysis is run on these on the box; they are inputs for `exp_ood_ptm.py` on
the analysis machine. Print the shapes.

## 4. Block C — text substrate (frozen sentence encoder)

```powershell
& $PY extract_text.py --dataset banking77 --out out\ptm_banking77_mpnet.npz
& $PY extract_text.py --dataset clinc150  --out out\ptm_clinc150_mpnet.npz --ood-out out\ood_clinc_oos_mpnet.npz
& $PY cache_summary.py out\ptm_banking77_mpnet.npz out\ptm_clinc150_mpnet.npz
Copy-Item "$W\gpu\out\ptm_*_mpnet.npz","$W\gpu\out\ood_clinc_oos_mpnet.npz" "$W\results\"
.\launch_chains.ps1 -Caches ptm_banking77_mpnet,ptm_clinc150_mpnet -Steps 1,2,3,4,5
```

Encoder: `sentence-transformers/all-mpnet-base-v2`, mean-pooled, fp16, never
tuned — fixed in advance, do not swap it for a better one after seeing numbers.
Banking77: 77 classes, inc 7 (11 tasks). CLINC150: the 150 in-scope intents,
inc 15 (10 tasks); HF's validation split is folded into train because the
pipeline draws its own validation rows; the 1,350 out-of-scope queries go to a
separate OOD cache. Record the joint NCM `cache_summary.py` prints (feature
quality, not a CL number). The chain is the ordinary PTM chain: the caches are
768-d and the `ptm_` prefix is what makes every script find them.

## 5. Block D — tabular substrate (CPU only; the download is the only risk)

```powershell
& $PY extract_tabular.py --dataset nslkdd --out out\ptm_nslkdd_raw.npz
# if the GitHub mirrors are unreachable, try:  --dataset cicids2017  (auto-download, ~230 MB zip)
# or place UNSW_NB15_training-set.csv / UNSW_NB15_testing-set.csv under data\unswnb15\ and use --dataset unswnb15
& $PY cache_summary.py out\ptm_*_raw.npz
Copy-Item "$W\gpu\out\ptm_*_raw.npz" "$W\results\"
.\launch_chains.ps1 -Caches ptm_nslkdd_raw -Steps 1,2,3,4,5
```

The script prints the classes kept (≥ 300 train and ≥ 30 test rows, 5,000-row
cap per class) and the ones dropped with their counts: copy that block into
your write-up verbatim, it is the protocol. Preprocessing is fixed (one-hot
categoricals, log1p numerics); the pipeline's own min-max scaling does the
rest. inc = 2. The organ configuration is the 768-d CIFAR one, unchanged: if
it is visibly wrong for ~100-d rows (growth strict far below NCM, or the
cascade covering almost nothing), say so with the numbers; do not tune it on
these seeds. One tabular dataset is enough; run a second only if the first
was trivially easy (NCM ≥ 0.98) or trivially broken.

## 6. Block E (optional) — ImageNet-1K checkpoint + adapters, six datasets

```powershell
.\extract_suite.ps1 -Backbone vit_base_patch16_224.augreg_in21k_ft_in1k -Tag vitb1k -Names obj,omni,vtab   # the three 1k caches not yet made
.\run_phase1.ps1 -Backbone vit_base_patch16_224.augreg_in21k_ft_in1k -Tag vitb1k -Names imr,ima,cub,obj,omni,vtab
& $PY cache_summary.py out\ptm_*_vitb1k*.npz
Copy-Item "$W\gpu\out\ptm_*_vitb1k*.npz" "$W\results\"
$c = Get-ChildItem "$W\results\ptm_*_vitb1k_p1s*.npz" | ForEach-Object { $_.BaseName }
.\launch_chains.ps1 -Caches ($c -join ",") -PerCacheSeed -Steps 2,5      # prototypes + head are what the RanPAC column needs
.\launch_chains.ps1 -Caches ptm_obj_vitb1k,ptm_omni_vitb1k,ptm_vtab_vitb1k -Steps 1,2,5
```

(`-Steps 2` needs step 1's json: for the `_p1s` caches run `-Steps 1,2,5`
instead if step 1 has not run; step 1 is the slow one, so prefer 1,2,5 only
when there is time.)

## 7. Write-up on the box: `BATTERY4_RESULTS.md`

One section per block. Tables, one row per (dataset, seed) plus a mean row:

- **A**: same columns as `PHASE1_RESULTS.md` (task-0 val acc, tuned params,
  contracted protos strict/forced, head M = 10k strict/forced/coverage/changes
  per 100, growth, adolescence accepted?, certified lifecycle, free arbiter,
  uncertified fraction), then the gate record for seeds 5–7: arbiter accepted
  n/21, adolescence accepted n/21, any accepted regression (candidate test
  below adolescence test) with its size, any refused candidate whose test
  accuracy was above adolescence by > 0.01. Compare each dataset's seeds-5–7
  mean to its seeds-2–4 mean from `BENCHMARKS.md` section A ("+ first-session
  adapters" rows) and say whether it is inside two standard deviations.
- **B**: shapes only.
- **C**, **D**: growth strict / coverage / flips, contracted protos strict /
  forced, head strict / forced / coverage / changes per 100, adolescence
  accepted?, certified lifecycle, NCM, replay, joint, EWC, fine-tune — all
  from the chain's own json files. No published number. Then four sentences:
  does the certificate transfer to this substrate (flips 0, uncertified 0.0),
  where does it land against same-data replay and NCM, what did the gate do,
  and anything that did not run.
- **E**: protos / head per (dataset, seed) next to RanPAC's Table 1 row, with
  the note that this is now the same checkpoint family RanPAC used; the
  `B0`-vs-`A_T` metric difference still stands and is named.

## 8. What to send back

- `results\*_p1s5.npz`, `*_p1s6.npz`, `*_p1s7.npz`; `results\ood_*.npz`;
  `results\cifar10_dinov2b.npz`; `results\ptm_*_mpnet.npz`; `results\ptm_*_raw.npz`;
  block E caches if run
- every new `results\*.json` and `results\*.pkl` the chains produced
  (`*_p1s5*`, `*_p1s6*`, `*_p1s7*`, `*mpnet*`, `*_raw*`, `lifecycle_*_s5.json`
  etc., `head_*`, `adol_org_*`)
- `gpu\out\phase1_*_s5.log` … `_s7.log`, the `extract_*` console output saved
  to `gpu\out\extract_battery4.txt`, `logs\chain_*`
- `BATTERY4_RESULTS.md`
- if you changed any code: the diff and pristine copies, as in `GPU_BOX_FIXES.md`

Not `claims.json` / `MANIFEST.json`.

## 9. Things that would be wrong to do

- Re-tuning the adapters, the encoder choice, the tabular preprocessing or the
  organ configuration after seeing a number on these seeds.
- Running a `_p1s<k>` cache with any seed other than k.
- Fetching a published Banking77 / CLINC150 / NSL-KDD continual-learning
  number from memory. If you have the PDF, record table, metric and protocol
  in `ANCHORS_PTM.md` style; otherwise leave the column empty.
- Dropping a class from the tabular stream because it scores badly. The
  inclusion rule is the row counts, fixed above.
- Interpreting the CLINC out-of-scope cache as a class: it is an OOD set for
  the analysis machine, nothing on the box reads it.
