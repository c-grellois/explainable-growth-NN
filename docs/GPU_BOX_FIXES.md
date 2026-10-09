# Code changes made on the GPU box (2026-09-07) — read before merging results

*Changelog of code changes made on the second machine, kept because each entry is a bug this code had. The 2026-09-07 entry is the most instructive: NaN-poisoned gate aggregates would have made every comparison false and the gate would have silently **accepted** a consolidation it should have refused.*

*Written by the GPU-box Claude session for the analysis-machine session. The
handoff said "pipeline frozen, report don't fix"; the user (c.grellois) then
explicitly authorised applying the fixes below after every PTM chain had
crashed on the same line. Nothing here is tuning: no CFG, threshold, seed,
class order or gate rule changed. Unified diff: `gpu\patches\gpu_box_fixes_2026-09-07.patch`;
pristine copies of the three files as shipped in the payload:
`gpu\patches\{organ,adolescence,exp_lifecycle}.py.orig`.*

## 1. `cgrow/organ.py` — `Organ._proj_nearest` and its two consumers (the crash)

**Symptom.** Every PTM chain died with

```
File "cgrow\organ.py", line 197, in _proj_nearest
    P = np.stack([projs[ci[i]][i] for i in range(len(X))])
ValueError: all input arrays must have the same shape
```

reached from `Organ.evaluate -> _veto_judge` (steps 1, 3, 4) and from
`Organ.add_veto` (VTAB step 1). Original crash logs are preserved in
`logs\crash_before_fix\`.

**Cause.** `PCAProx.fit` (`cgrow\proximity.py:100`) gives each class
territory `q = min(24, d-1, n_c-1)` components, so a class with fewer than 25
samples in an organ's fit gets a narrower territory. `_proj_nearest`
computed every sample's whitened coordinates in every territory and then
stacked, per sample, the row from its *nearest* territory into one `(n, q)`
array, which presumes one common `q`. On CIFAR (250 train/class, 45k train)
every territory had q = 24 and the stack never failed. On the PTM suite the
organs grown on residual (abstained) samples routinely have classes with
< 25 samples: the cached CUB seed-2 organism has territory widths such as
`[2, 13, 16, 21, 24]` in 8 of its 10 organs.

**Change.** `_proj_nearest` now returns `(projs, ci)` where `projs[j]` is the
`(n, q_j)` coordinate array of *all* samples in territory `j` and `ci[i]` the
nearest-territory index; no stacking. The two consumers select
`projs[j][ci == j]` — which is exactly the set of rows the old stack held for
those samples. Three call sites updated: `_veto_judge` (one), `add_veto`
(negatives and the `X_self` verification set). No other module calls
`_proj_nearest` (grepped `cgrow/*.py`, `exp_*.py`).

**Equivalence.** On the cached CUB organism, for the 2 organs whose
territories all have q = 24 the old stack and the new per-territory selection
agree with max |difference| = 0.0 over 2000 test samples. For mixed-width
organs the old code raised, so there is no prior number to preserve. Each
veto net was trained on `neg.shape[1]` = q_j inputs in `add_veto`, so feeding
it `projs[j]` (width q_j) is the intended pairing; the old stack would have
fed it the wrong width had it not raised first.

## 2. `cgrow/adolescence.py` — `consolidate_protos` anchors for classes with no pool sample

**Symptom.** `RuntimeWarning: Mean of empty slice` at `adolescence.py:220`
(ImageNet-A), i.e. a NaN anchor row carried into the consolidated
`ProtoOrgan`.

**Cause.** Anchors were the mean of each class's synthetic samples in the 80 %
`tr` split of the contract pool. A class has no such sample when (a) the
random split left none in `tr`, or (b) no organ's territory covers it at
all: `PCAProx.fit` skips classes with fewer than 3 samples ("no territory
yet") and `PCAProx.sample` only emits classes that have a territory.
ImageNet-A's distributed split has classes with 2 training images.

**Change.** Per class, in order: mean over its `tr` samples (unchanged
behaviour when present); else mean over all its pool samples (`tr` ∪ holdout);
else the class is **not represented in C** — it is removed from C's class
list, so the cascade below C keeps answering it. The report gains
`classes_anchored_from_holdout` and `classes_dropped_no_pool` (both lists,
empty on CIFAR-like data). When every class has ≥ 1 `tr` sample the anchors
are numerically identical to before (same samples, same normalisation).

## 3. `cgrow/adolescence.py` `_strict_per_class` / `_gate_synthetic`, and `exp_lifecycle.py` line ~155 — NaN-safe gate aggregation

**Symptom.** `Mean of empty slice` at `adolescence.py:107` (ImageNet-A):
`_strict_per_class` returned NaN for a class absent from the synthetic
holdout.

**Cause / consequence.** With plain `min`/`mean`, one NaN makes the aggregate
NaN and *every* comparison False. In `_gate_synthetic` that meant the
"mean regressed" and "worst class regressed" checks both passed → the
consolidation was **silently accepted**. In `exp_lifecycle.py` the arbiter
rule `mean_new >= mean_ref + 0.01 and ...` became False → the arbiter was
**refused by arithmetic**, not by evidence. Neither can happen on CIFAR
(every class has hundreds of pool samples).

**Change.** `_strict_per_class` returns NaN explicitly (no warning) for a
class with no holdout sample; a helper `_agg` aggregates min/mean over the
observed classes and `_gate_synthetic` uses it, records
`classes_observed_in_holdout` / `classes_total`, and refuses outright if no
class is observed. `exp_lifecycle.py` uses `np.nanmean` / `np.nanmin` in the
same rule (unchanged constants +0.01 / −0.05) and records
`arbiter_classes_unobserved_in_holdout`. When all classes are observed the
values are identical to before. Other scripts that import
`_strict_per_class` (`exp_arbiter_cifar`, `exp_gate_audit`, `exp_isolate`,
`exp_negotiate`, `exp_pfc`, `exp_sleep`) still receive the same NaN they
received before (the old empty-slice mean was already NaN); they were not run
here and were not touched.

## 4. Auxiliary files (not pipeline code)

- `gpu\launch_ptm_chains.ps1` — the section-6 launcher as a script, with
  `-Names` and `-Steps` so steps 3–4 can be rerun without redoing 1–2.
- `gpu\patches\` — originals, diff.
- `logs\crash_before_fix\` — the six pre-fix `.log`/`.err` pairs.

## 5. What was rerun with the fixed code

See `PTM_RESULTS.md` §3 (updated after the reruns). Steps 1–2 results
(`results\ptm_{imr,ima,cub,obj}_vitb.json`) were produced *before* the fix
and are kept: their code path never reached the failing line except on
OmniBenchmark / VTAB, which are rerun from step 1. Growth organisms cached
before the fix (`adol_org_*_2.pkl`) are reused by step 3: growth code that
did not raise produced exactly what the fixed code produces.


## Battery 4 (2026-09-14/15) — one loader fix, one logging incident

**Loader fix (adopted verbatim, `gpu/extract_text.py`; pristine copy and diff
under `gpu/patches/`).** `datasets` 5 refuses short Hub ids and script-based
repos. CLINC150 now loads from `clinc/clinc_oos` (parquet); Banking77 reads the
two CSVs the original `banking77.py` itself downloaded and maps categories
with the script's own `ClassLabel` names from `dataset_infos.json`, so texts,
splits (10,003 / 3,080) and integer labels are identical. Encoder, pooling,
batch and inc untouched.

**Incident (logs only).** `run_phase1.ps1` named its logs
`out\phase1_<name>_s<seed>.log` without the backbone tag, so the block-E runs
(`-Tag vitb1k`, seeds 2–4) overwrote the battery-3 in21k Phase-1 training
logs for six datasets on the box. No cache, json, head, lifecycle or pkl was
affected (all carry the tag); the battery-3 logs exist in the battery-3
send-back on this machine (`GPUOutput2/`). Fixed here on 2026-09-15: the log
name now includes `$Tag`. The box, correctly, did not change the script after
the fact.

**Reproducibility note.** The box's re-run of the battery-2 `ptm_*_vitb1k`
chains reproduces prototypes and NCM to the last digit and growth / replay to
within 0.8 / 0.3 points (torch training under a different thread count); one
CUB and one OmniBenchmark gate decision on seeds 2–4 differed between the two
machines. The battery-2 originals remain the files of record here.
