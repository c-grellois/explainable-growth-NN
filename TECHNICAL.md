# Technical report: how the contracted-growth code works

*Companion to [DESIGN.md](DESIGN.md) (the why) and [REPORT.md](REPORT.md) (the
results). This document is the how: the architecture of the code, every file,
and what the main functions do. Environment: `requirements.txt` (the results
were produced with Python 3.12, torch 2.13 CPU, numpy 2.5, scikit-learn 1.9,
scipy 1.18; torchvision 0.28 and timm 1.0.28 for the local extractors). After
the first download of each dataset everything runs offline from local
caches.*

---

## 1. Architecture in one page

The system has a small core library, `cgrow/` (~115 KB of Python), and a layer
of experiment scripts (`exp_*.py`) that compose it. The core defines four
objects:

```
Proximity (cgrow/proximity.py)    "is this input inside known territory?"
     ▲  fitted per organ, at birth, on real data; doubles as a data GENERATOR
Organ (cgrow/organ.py)            trunk + anchors + thresholds + vetoes
     ▲  frozen at birth; only monotone tightening afterwards
Organism (cgrow/organism.py)      first-accept cascade + growth gate + audit
     ▲
Lifecycle ops (cgrow/adolescence.py)   consolidation + gates (+ exp_pfc arbiter)
```

Data flows one way: an experiment builds an `Organism`, grows `Organ`s on the
abstained mass, tightens them against incoming data, optionally runs the
lifecycle operators, and audits everything bitwise. Nothing in the core ever
mutates a frozen parameter; the only sanctioned changes are (a) appending
organs, (b) monotone threshold/veto tightening, (c) the two gated
consolidations. Every experiment writes per-seed JSON records to `results/`,
and `claims.py` regenerates every reported number from them.

Two conventions run through all code:

- **Solve, never penalize** (inherited from NNcontracts §7): every certificate
  parameter — thresholds ρ, τ, ρ_in, veto cutoffs θ, arbiter confidence,
  consolidation acceptance — is chosen by direct search or closed form against
  held-out data, never as a loss term.
- **Controls that must fail**: experiments carry sabotage arms (swapped specs,
  unverified growth, fine-tuned parents) whose failure is asserted.

---

## 2. The core library, file by file

### 2.1 `cgrow/organ.py` — the contracted part (≈24 KB)

**`SpectralMLP(dims, cap, seed)`** — an MLP whose layer spectral norms are
projected to ≤ `cap` after every optimizer step.
- `project(n_iter=2)`: persistent-vector power iteration per layer; rescales
  a weight matrix when its estimated top singular value exceeds the cap.
- `lipschitz_exact()`: exact product of layer spectral norms (via `svdvals`
  in float64), computed once at freeze — the certificate uses the exact value,
  training uses the cheap projection.

**`Organ(in_dim, feat_dim, hidden, cap, class_ids, seed, k_centroids,
q_cov_in, prox)`** — one part, one contract. Key state: `trunk`
(SpectralMLP), `anchors` (one per served class, in feature space), thresholds
`rho`/`tau`/`rho_in`, the proximity object `prox`, the veto list, `frozen`,
and `stats` (birth-validation scalars licensing future tightening).

- `train_proto(X, y, epochs, lr, batch, prune, prune_rounds, prune_epochs)`:
  prototypical training — cross-entropy over `−‖φ(x) − a_c‖²`, spectral
  projection every step, anchors initialized at class means of initial
  features; filters samples to the organ's own classes; ends by fitting the
  proximity geometry (`make_prox(...).fit`) on the same real data. `prune > 0`
  adds magnitude pruning **at birth**: drop the smallest-|w| fraction of each
  trunk layer, retrain the survivors with the dropped weights pinned at zero,
  and only then solve thresholds and freeze, so the contract describes the
  network that ships. Pruning a frozen organ would invalidate its contract and
  is not offered. `prune = 0` leaves the unpruned path bit for bit unchanged.
- `_features(X)` / `_lipschitz_exact()`: the two override hooks. Everything
  contract-side (thresholds, vetoes, tightening) is geometry-agnostic;
  subclasses replace only the feature map. `SheetOrgan` and `ProtoOrgan`
  exploit this.
- `evaluate(X) → (d1, margin, pred, d_in, vetoed, veto_r)`: nearest-anchor
  distance, half-gap to the runner-up, predicted global class, proximity
  score, veto verdicts and their certified slack. When the organ has vetoes on
  a PCA territory, the per-class projection (`prox.project`) is computed once
  and shared by the proximity score and the veto routing.
- `accept(d1, margin, d_in, vetoed)`: the contract —
  `d1 ≤ ρ ∧ margin > τ ∧ d_in ≤ ρ_in ∧ ¬vetoed`.
- `robust_radius(...)`: certified input-space radius for accepted points —
  the minimum of the feature-margin slack scaled by `1/L`, the proximity slack
  scaled by the proximity's Lipschitz bound, and each applicable veto's slack.
  Pure arithmetic, no bound propagation.
- `solve_thresholds(Xval, yval, p_target, require, q_cov)`: proximity radii
  are set to own-correct quantiles ("cover my own data, nothing more" — a
  precision objective cannot see future negatives, so proximity must not be
  maximized against it); τ is then grid-searched to maximize coverage subject
  to precision ≥ `p_target`. `require=False` (ungated founders) falls back to
  the best-precision point.
- `freeze(Xval, yval)`: computes the exact Lipschitz constant and stores the
  birth statistics `(d1, margin, d_in, correct)` — the organ's license for
  later tightening decisions without retaining raw data.
- `tighten(ρ′, τ′, ρ_in′)` / `solve_tightening(...)`: monotone by assertion
  (accept set can only shrink); the solver grid-searches the exclusions of
  measured false accepts subject to an estimated self-coverage budget from the
  stored stats.
- `_proj_nearest(X, parts)`: whitened PCA coordinates of each sample in its
  nearest class territory (routes veto judgment); mixture-aware when `prox` is
  a `MixProx`; reuses a precomputed projection when the caller has one.
- `add_veto(X_fa, theta_q, X_self, max_self_loss)`: **learned tightening** —
  per class territory with enough routed negatives, trains a small capped net
  on exact-spec synthetic positives (`prox.sample_whitened`, `N(0, I)` for the
  PCA model) vs the real falsely-accepted samples; threshold θ solved as a
  quantile of positive scores; certified input-space Lipschitz recorded via
  the whitening map's spectral norm. When real own data is supplied (birth
  vetoes), each veto is verified against it and discarded if it costs more
  than `max_self_loss` of own coverage. Purely conjunctive → retention-safe.

### 2.2 `cgrow/proximity.py` — territories and generators (≈20 KB)

One interface, several geometries. Every scorer returns a dimensionless
`score(X)` calibrated so own data sits ≤ 1 near the `CAL_Q=0.9` quantile,
publishes `lipschitz()` (so acceptance stays a certificate), `sample(n, rng)`
(the spec as data generator — the load-bearing property for vetoes,
adolescence, the arbiter, and replacement), and `spec_size()`.

- **`PCAProx(n_components=24)`** — the workhorse. Per class: mean `mu`, top-q
  axes `V`, floored eigenvalues `lam`, and two calibration scalars (`r_q` for
  the subspace residual, `w_q` for the whitened in-subspace norm).
  `score = min over classes of max(residual/r_q, whitened/w_q)`.
  `project(X)`: per class, the coordinates along the principal axes and the
  distance from the subspace — the one expensive step, shared by `score` and
  an organ's veto routing.
  `sample`: gaussian in the subspace + isotropic residual;
  `sample_whitened`: `N(0, I)` — exactly the model the spec asserts, which is
  what makes veto positives faithful *by construction*.
- **`MixProx(k_mix=5)`** — subclass of PCAProx; a diagonal GMM on the whitened
  in-subspace coordinates per class. Measured tie-or-loss vs PCAProx
  (`exp_sampler.py`), kept for completeness; `pca` remains the default.
- **`KMeansProx`** — v1 geometry: input-space k-means centroids with shrunk
  per-cluster radii; still the default for small non-image data configs.
- **`AEProx`** — spectrally capped autoencoder; score = reconstruction error.
- **`ComboProx`**, **`DiscProx`** — shoot-out contenders (intersection of
  territories; discriminator vs synthetic corruptions). Both measured
  non-superior.
- **`UnionProx(members)`** — opaque union (min over members); superseded for
  the consolidated organ by:
- **`merge_pca_family(members)`** — the veto-capability fix: a union of PCA
  territories *is* a PCA territory with more classes, so members' per-class
  lists are concatenated into one first-class prox that supports vetoes,
  whitened projection, and sampling.

### 2.3 `cgrow/organism.py` — the cascade and its audit (≈8 KB)

**`Organism`** holds `organs` (ordered by birth), a structured event `log`,
and retention snapshots.

- `predict(X)`: first-accept cascade. Returns `pred` (global class or
  `ABSTAIN`), the answering `organ` index, the certified `radius`, and a
  `forced` fallback (nearest-anchor across organs, normalized by ρ) for
  abstention-free scoring.
- `scores(X, y)`: strict (abstention = error), forced, coverage, precision,
  mean certified radius.
- `try_grow(candidate, Xval, yval, p_target, gate)`: the growth gate. Solves
  the candidate's thresholds **on the validation mass the organism currently
  declines** (its future jurisdiction), accepts only if it answers
  net-positively there. `p_target` may be a declining ladder (static-task
  growth takes the highest-precision slice still available). The founder runs
  ungated — there is nothing to protect yet.
- `tighten_against(X_foreign, self_budget, veto, veto_theta_q)`: incoming data
  whose classes no organ serves. Two monotone stages: scalar threshold
  tightening under a self-coverage budget (measured **net-harmful** once
  vetoes exist — default budget 0 in all final configs), then `add_veto` on
  the residual false accepts. Each organ is evaluated on the foreign data once:
  the scalar stage moves only thresholds that `evaluate` does not read.
  Everything logged.
- `retention_snapshot` / `retention_check`: bitwise verification of the
  invariant — no previously answered input may carry a different answer;
  experiments additionally audit growth events and tightening events
  separately (growth flips must be exactly zero; tightening re-routes are
  logged with their to-right/to-wrong balance).

### 2.4 `cgrow/sheet_organ.py` — the spatial substrate (≈12 KB)

**`SheetOrgan(Organ)`** — an organ made of recurrent units embedded in
`[0,1]²`, faithful to ASDbrain's conventions: `tanh` dynamics over `T=8`
steps, input entering at a fixed site (`ENTRY`), the hinge wiring cost
`c(d) = d + β·max(0, d−d₀)²` on lateral *and* input wiring, cost-weighted
pruning `|W|/c(d)`, and a spectral cap on the recurrent block (echo state
preserved; input→feature Lipschitz bound `σ(U)·Σₜσ(W)ᵗ`).

- `place_center(existing)`: organ placement by the wiring economy — cheapest
  input access on a grid, crowding-penalized.
- Directional growth: `W_prev` lets a newborn listen to frozen predecessors'
  units; **no edge ever enters a frozen row** — the retention theorem on a
  shared recurrent sheet, verified causally by the lesion experiments
  (lower-triangular damage matrices).
- `_run` / `_chain_states`: unrolls the organ's own dynamics given the frozen
  predecessors' trajectories (recomputed on demand; organs reference their
  prefix). `_features` = mean state over time of the organ's own units.
- `train_proto`: prototypical loss + wiring penalties + spectral projection,
  then `prune(frac_keep=0.35)` and proximity fitting (on flattened inputs, so
  vetoes work unchanged on the sheet).
- `edges()` / `n_params()`: anatomy accounting (per-edge weights and lengths)
  consumed by `exp_sheet.anatomy` and `lesion_matrix`.

### 2.5 `cgrow/adolescence.py` — the lifecycle operators (≈20 KB)

- `sample_pool(org, n_per_organ, rng)`: labeled synthetic data from every
  organ's contract — the resource that powers everything below.
- `_gate_synthetic(C, X_ho, y_ho, classes, cfg, ref_org)`: the consolidation
  gate, with the two repairs the CIFAR diagnosis forced: a **usefulness
  floor** on holdout coverage (vacuous consolidations pass every safety
  check), and a **relative** per-class comparison against the organism being
  replaced on identical data (absolute floors implicitly harden with class
  count).
- `consolidate_mlp(org, in_dim, cfg, seed, Xva_last, yva_last, prox_policy)`:
  trains one trunk-organ on the pooled synthetic data; territory inherited via
  `_inherit_territory` (merged, veto-capable) or refit (`prox_policy="refit"`,
  the measured-vacuous renegotiation arm); gates on synthetic holdout + the
  last task's real validation; swaps in ahead of the old cascade, which stays
  as fallback so coverage cannot drop. **Known limit:** the trained trunk
  collapses on 100-class pools.
- `ProtoOrgan(Organ)` + `consolidate_protos(...)`: the high-class-count
  consolidator — anchors are closed-form normalized class means of the pool
  (no training at all), features are L2-normalized inputs
  (`_lipschitz_exact = 2/min‖x‖`), territory merged. This is the variant the
  lifecycle uses everywhere.
- `consolidate_sheet(...)`: the sheet version (units re-placed, local wiring
  re-learned from scratch under the hinge cost — the literal "massive
  recombination of local connections").
- `metamorphosis_audit(before, after, Xte, yte)`: real-test accounting of the
  one sanctioned invariant exception — flips and their direction, gained/lost
  answers, share answered by the consolidated organ (the vacuity detector).

### 2.6 `cgrow/baselines.py` and `cgrow/data.py`

- `baselines.py`: a plain `MLP`, a shared `_epochs` trainer (with optional
  EWC penalty closure and replay-buffer mixing), `_fisher_diag`, `train_plain`,
  `accuracy`, and `run_sequential` for fine-tune / EWC / replay streams.
- `data.py`: `toy_modes` (multi-modal 2-D classes — multimodality is what
  makes growth necessary in the toy), `load_real` (digits, mnist, kmnist,
  fashion, letter/letter10, satimage, wall-robot; digits ships with
  scikit-learn, the rest are fetched from OpenML once and the converted arrays
  cached as `.npz` under `<sklearn data home>/cgrow_cache/`, which must be
  deleted if the conversion in `_load_real` ever changes), `prepare`
  (train/val/test split + min-max scaling to `[0,1]^d` on train statistics),
  `split_tasks` (class-incremental task lists).

### 2.7 `cgrow/gates.py` and `cgrow/curriculum.py`

- `gates.py` (≈13 KB): the repaired acceptance gate, shared by every gated
  lifecycle step. Three sub-checks, each reported by name in the verdict:
  `territory_check` (data-free), `escrow_check` (real held-out data) and
  `contract_check` (the original rule), combined by `certify`; a candidate
  passes only if every enabled sub-check passes, and `certified` is reported
  separately from `accepted`. `detectable_drop` states the smallest regression
  an escrow of a given size can resolve, `escrow_split` / `escrow_sample` carve
  the escrow out of the validation split, and `DriftLedger` keeps the lifetime
  budget on accepted drift (`B-ledger`). §4 below states what each check does.
- `curriculum.py` (≈6 KB): class-order policies behind
  `exp_lifecycle.py --order-policy` (`random`, `easy`, `hard`, `grouped`,
  `spread`), with difficulty measured by a nearest-class-mean fit before any
  organ exists. A curriculum changes the protocol, so its results are never
  placed in the same table as the random-order anchors.

### 2.8 The prefrontal tier (`exp_pfc.py`, reused by `exp_lifecycle.py`)

- `evidence(org, X)`: the executive's input — for every organ, `(d1, margin,
  d_in, accept)`, concatenated; plus the per-organ forced predictions.
- `Arbiter` / `train_arbiter(F, correct)`: a two-layer net trained with
  multi-label BCE to predict, per organ, "would this organ answer this sample
  correctly" — trained **only on contract-sampled data** (labels known, each
  organ's correctness computable).
- `arbiter_predict(net, F, preds, theta)`: pick the most-trusted organ, take
  its prediction; abstain below a solved confidence threshold.
- Ultimate consolidation: the arbiter passes the same relative gate
  (synthetic per-class vs the pre-arbiter organism, plus the last task's real
  validation, plus — in `exp_lifecycle` — a strict-improvement requirement),
  its answer changes are audited as one logged event, and it is then frozen.

---

## 3. The experiment scripts

Each writes per-seed JSON to `results/` and prints a per-seed line. `claims.py`
aggregates all of them into `results/claims.json`, which is what the documents
and the paper quote; adding an experiment means adding a block there.

**The substrate ladder**
- `exp_toy.py` — B0: growth by abstention on 2-D multimodal data; monotone
  coverage, zero flips; two controls that must fail (anchor injection inside
  the parent = NNcontracts' closed mechanism; fine-tuning the parent =
  ASDbrain's interference mechanism).
- `exp_static.py` — B1: growth vs a single organ vs a plain MLP on static
  tasks, with the declining-precision growth ladder.
- `exp_continual.py` — B2: class-incremental streams (`run_ours` = tighten →
  grow → birth-veto → audit per task; `run_baseline` = fine-tune / EWC /
  replay; joint bound).
- `exp_sheet.py`, `exp_sheet_audit.py` — the spatial substrate: sheet organs,
  anatomy statistics, the lesion matrix; streams `sweeps` / `shapes` / `fsdd`.
- `exp_cifar.py` — the pretrained-backbone harness and the workhorse of every
  later battery: cascade organism, `contracted_ncm` (the prototype topology
  wrapped in contract machinery), NCM, replay and joint baselines; flags for
  the B50 protocol, `--adolescence --adol-mode protos`, `--inc k` for longer
  horizons, and organism caching to `results/adol_org_*.pkl`.

**Mechanism studies** (each one redirected the design)
- `exp_ood.py` — the proximity-geometry shoot-out; the unsupervised frontier
  saturates.
- `exp_veto.py`, `exp_ablate_tighten.py` — the veto measurement and the
  four-cell tightening ablation: veto-only wins, scalar is net-harmful on top.
- `exp_sampler.py` — sampler fidelity (real-vs-synthetic AUROC).
- `exp_drift.py`, `exp_meta.py` — the drift forecast and the second-order SLA.
- `exp_isolate.py` — the feature ledger: remove each mechanism through the
  full pipeline and measure what it was worth.

**Lifecycle**
- `exp_adolescence.py`, `exp_renegotiate.py`, `exp_midlife.py` — consolidation,
  contract renegotiation, and consolidating mid-stream then growing on.
- `exp_pfc.py`, `exp_arbiter_cifar.py`, `exp_arbiter_source.py` — the arbiter,
  and where its advantage comes from.
- `exp_lifecycle.py` — the full gated lifecycle (growth → gated adolescence →
  gated arbiter) on every benchmark; `--order-policy` for curricula, `--inc`
  for horizons.
- `exp_gatefix.py`, `exp_gate_audit.py`, `exp_adol_escrow.py` — the gate's own
  audit: the two blind spots found in it, the repair, and a three-arm replay of
  every consolidation decision on record.
- `exp_apoptosis.py`, `exp_redundant.py`, `exp_ensemble.py`, `exp_vote.py`,
  `exp_fractal.py`, `exp_sleep.py`, `exp_market.py`, `exp_negotiate.py` —
  organ death, redundant growth, voting, nesting, sleep cycles, and the two
  market mechanisms.
- `exp_replace.py` — the replaceable organ: re-derive one from its published
  spec alone; the swapped-spec control must invert.

**Accuracy configurations**
- `exp_head.py` — the contracted head: a random-projection ridge readout under
  the same contract. The accuracy lever, and the configuration whose answers
  move as the ridge is re-solved.
- `exp_head_blocks.py` — a head that never re-solves (one frozen ridge block
  per task, first-accept). Delivers the invariant exactly and loses the
  accuracy; a negative result with an identified mechanism.

**Curricula** (a different protocol: they use label information the stream
does not give)
- `exp_hier.py`, `exp_progressive.py`, and `cgrow/curriculum.py` behind
  `exp_lifecycle.py --order-policy`.

**Properties of the deployed system**
- `exp_ood_ptm.py` — the certificate slack as an out-of-distribution and
  selective-prediction score, against same-feature baselines.
- `exp_cost.py` — state to answer, state to keep learning, MACs per sample and
  seconds to absorb one task, per configuration.
- `exp_size.py` — how big the grown organism is: neurons, connections and
  contract floats, counted on an organism built through the ordinary
  `exp_lifecycle.build` path and scored as the same object, against the single
  network the baselines train.
- `exp_prune.py` — magnitude pruning at birth swept over sparsity (preregistered
  in its docstring): accuracy, size and growth flips per sparsity.

**Feature and data builders**
- `extract_cifar_feats.py`, `extract_cifar_vit.py`, `build_fsdd.py` — local
  caches.
- `gpu/` — everything that needs a GPU: `extract_gpu.py`, `extract_ptm.py`,
  `phase1_extract.py` (first-session adapters), `extract_ood.py`,
  `extract_text.py`, `extract_tabular.py`, with PowerShell setup scripts and
  launchers (`launch_chains.ps1` is the general one) and read-only summaries
  (`cache_summary.py`, `summarize_phase1.py`, `summarize_battery4.py`).
  Written for a second machine that an AI assistant session drove from the
  battery instructions in `docs/`; [gpu/README.md](gpu/README.md) is the
  runbook.

**Reporting**
- `claims.py` — the single aggregation point; `manifest.py` — content hashes of
  every result file; `gen_overview.py` — the three-page overview in
  `overview/`; `make_figures.py` — the paper figures; `trace_anatomy.py` — the
  interactive explainer in `explainer/`; `summarize_indexed.py`,
  `summarize_market_maker.py` — read-only tables for the market experiments.
- `paper/main.tex` (+ a stand-in `neurips_2025.sty`, `references.bib`).

---

## 4. Invariants the code enforces (and how they are checked)

1. **Frozen means frozen.** `Organ.train_proto` asserts `not frozen`;
   `tighten` asserts monotonicity; nothing else writes to a frozen organ.
2. **Growth changes no answer.** Checked empirically in every experiment by
   comparing `predict` outputs on the full test set before/after each growth
   event (`growth_flips`, must be 0 — and is, in every run of the study). At
   the long horizons, where the harness does not keep the live counter, the
   same invariant is measured afterwards over every prefix of the cached
   anatomy: 501 additions on bodies of up to 100 organs, zero changes.
3. **Tightening only shrinks accepted sets.** Scalar moves are
   assertion-monotone; vetoes are conjunctive by construction; re-routes are
   counted and direction-audited.
4. **Consolidations are gated, never assumed.** Every structural change faces
   `cgrow/gates.py`: a data-free **territory check** (every answer must come
   from an organ that accepted the input), a **real-data escrow** (a one-sided
   McNemar test on the held-out validation split, per-class flags under a
   Bonferroni correction, and a hard floor that requires both a large fraction
   and enough lost samples), and the **original rule** on contract samples plus
   the last task's validation. The old cascade always remains as fallback. The
   gate was audited against its own record, two blind spots were found and
   repaired, and it was then confirmed on seeds it had never seen; its
   measured record, including the refusals it got wrong, is in the `B-gatefix`,
   `B-adol-escrow` and `B-ptm-confirm` sections of REPORT.md.
5. **No raw data retained.** Contracts store prototypes, axes, eigenvalues,
   thresholds, birth-stat scalars, and small nets — never samples. Every
   synthetic-data pathway (vetoes, adolescence, arbiter, replacement,
   refinement audits) flows through `prox.sample`/`sample_whitened`.
6. **Every number regenerates.** `results/*.json` are the ground truth;
   `claims.py` is the single aggregation point; the paper and report quote
   only from it.
