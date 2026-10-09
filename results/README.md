# results/ — the per-seed records

Every number the project reports is regenerated from the JSON files in this
folder. `claims.py` reads them and writes `claims.json`; `manifest.py` writes
`MANIFEST.json`, the sha256, size and modification time of every result
artefact. The documents quote `claims.json` and nothing else.

## Tracked

- **`<experiment>_<benchmark or cache>[...].json`** — what each script wrote:
  one record per seed, with the scores, the growth-flip count, the gate's
  verdicts and whatever else that experiment measures. The script that wrote a
  file is named in the REPORT.md section that uses it.
- **`claims.json`** — the aggregate: one family per battery (`B0_toy`,
  `B2_continual_<dataset>`, `B_ptm`, `B_head`, `B_phase1`, `B_size_mnist`, …),
  each with means, standard deviations and per-seed values, plus
  `_provenance`, a hash of every JSON it read.
- **`MANIFEST.json`** — content hashes of every result artefact, caches and
  organisms included when present.
- **`*_smoke.json`** — short runs that checked a script before its battery.
  Nothing quotes them.
- **`archive/`** — a superseded result kept for the record.
- `toy_anatomy.png` — the toy organism's anatomy, drawn by `exp_toy.py`.

## Not tracked (about 8 GB)

- **Feature caches, `*.npz`**: `cifar100_<tag>.npz`, `ptm_<dataset>_<tag>.npz`
  and `ood_<set>_<tag>.npz`, built by the extractors in [`gpu/`](../gpu/README.md)
  (and, for the two local CIFAR caches, `extract_cifar_feats.py` and
  `extract_cifar_vit.py`). Each stores its class count and classes per task.
- **Grown organisms, `adol_org_<tag>_<seed>.pkl`**: cached by
  `exp_cifar.py --adolescence` and reused by the later steps on the same cache;
  rebuilt on first use.

## Naming

| pattern | meaning |
|---|---|
| `<tag>` | the encoder behind a cache: `vitb` (ViT-B/16-in21k, 250 per class on CIFAR), `vitb_full` (500 per class), `vitb1k` (the ImageNet-1K checkpoint), `dinov2b`, `vitl`, `r18`, `mpnet` (sentence encoder), `raw` (tabular rows) |
| `_p1s<seed>` | a cache extracted after first-session adapters trained on that seed's first task; it is used with that seed only |
| `_inc<k>` | the same cache cut into tasks of `k` classes (the horizon runs) |
| `_s<seed>` | a single-seed file, aggregated with its siblings by `claims.py` |
| `cifar_<tag>`, `ptm_<tag>` | benchmark names inside `lifecycle_*.json`; each resolves to the matching cache |
| `_preescrow` | the lifecycle as it ran before the adolescence gate gained the real-data escrow (`B-adol-escrow` in REPORT.md); kept for the record, not quoted |

The development-seed records (seeds 0–1) are in [`../results_dev/`](../results_dev/README.md).
