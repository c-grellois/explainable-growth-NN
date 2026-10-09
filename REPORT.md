# Contracted growth: results

*Prototype and benchmarks for the unification of ASDbrain ("Never Go Dense") and
NNcontracts ("Interval contracts at neural network interfaces"). Design in
[DESIGN.md](DESIGN.md); every number below is regenerated from per-seed records by
`claims.py` → `results/claims.json`.*

*How to read this record.* It is the project's lab book, kept in full: each
`B-…` section is one battery of experiments, written when it ran and amended
in place (never silently) when a later result overturned it. Sections are
labelled with the working session that produced them (sessions 1–44,
2026-08-20 to 2026-10-02). They are not in date order: the rigor pass and the
development-seed sections come first, the later batteries follow, and the
newest are in the middle of the file. For a summary read
[APPROACH.md](APPROACH.md); for the current comparison tables,
[BENCHMARKS.md](BENCHMARKS.md). "GPU box" is the second machine on which the
frozen-encoder features were extracted.

## The system in one paragraph

An **organism** is a first-accept cascade of frozen **organs**. Each organ is a
spectrally-capped prototypical trunk behind a **ball contract**: feature anchors with
margin slack τ and radius ρ, plus a learned input-space **proximity geometry** —
per-class PCA territories (means, axes, eigenvalues, calibration scalars; no raw
samples) with a cap ρ_in. Growth happens where the organism **abstains** (the
certified need field); the parent is never touched. Incoming out-of-distribution
data triggers monotone **tightening** — accepted sets can only shrink, every change
is logged. The two methodological inheritances are applied throughout: certificate
parameters are *solved, never penalized* (NNcontracts §7), and no stage ever trains
a monolith (ASDbrain's never-go-dense).

## The rigor pass: evaluation-seed verdict (read this first)

The sections from here down to "Final scoreboard vs published results" report
**development-seed** numbers (seeds 0–1, archived in `results_dev/`), on which
every knob was tuned. Every section after them states its own seeds: mostly
evaluation seeds 2–4, with gate confirmation on seeds 5–7. The final protocol
froze all configurations, switched thresholds to **split-conformal calibration**
(finite-sample coverage guarantees under exchangeability), **strengthened the
baselines** (z-scored features — which cured the suspicious ViT-head collapse —
doubled training, doubled-width joint), and reran everything on untouched
**evaluation seeds 2–4**. The verdict, honestly:

**What replicated.** The organism's own numbers reproduce within ~1 point on
every benchmark (mnist 0.892 vs dev 0.896; fashion 0.799 vs 0.798; cifar-ViT
cascade 0.667 vs 0.660; contracted prototypes 0.844 forced, exactly) — the
dev-tuning bias was small. Growth flips remain **exactly zero** in every
evaluation run. The CIFAR-ViT lifecycle replicates in full (0.667 → 0.776 →
0.835, 3/3 seeds, all gates passing), as do the arbiter's gains on digits
(→0.948), mnist (→0.915), cifar-r18 (→0.483), shapes (→1.000) and real audio
(→0.872).

**What died under stronger baselines.** (1) *Every "beats joint" claim*: the
dev joints were undertrained; the strengthened joints score 0.90–0.98
everywhere (letter 0.949 vs our 0.782 — the celebrated dev gap of +0.02 was an
artifact of a weak joint). (2) *The replay scoreboard shrank*: strengthened
replay jumped (digits 0.69→0.93, cifar-ViT 0.25→0.86, satimage 0.71→0.88), and
the honest count is now **ours ≥ replay on 6 of 12** (mnist, kmnist, fashion,
letter, robot, and digits-after-lifecycle) rather than 7 of 12 — and **5 of 12
once the lifecycle is certified**, since certified digits (0.912) no longer
clears replay's 0.933 — and replay
slightly leads even contracted prototypes on cifar-ViT (0.862 vs 0.844).
(3) *The wall-robot gate false-accept is systematic*, not a fluke: 3/3
evaluation seeds accept an arbiter that regresses ~12 points. Data-free gating
has a real, reproducible boundary where contract fidelity fails. **Partly
revised** (B-gatefix): a data-free *territory* check refuses that arbiter on all
three seeds — but it refuses every other free arbiter too, including eleven
beneficial ones, so it prevents the failure as a matter of policy rather than
detecting it. Detecting the regression itself still needs held-out data, and
even then our escrow missed one of the three seeds.

| eval seeds 2–4 (strict) | ours | + lifecycle | replay (strengthened) | joint (strengthened) |
|---|---|---|---|---|
| split-digits | 0.877 | **0.948** | 0.933 | 0.975 |
| split-MNIST | **0.892** | 0.915 | 0.871 | 0.977 |
| split-KMNIST | **0.805** | 0.812 | 0.773 | 0.957 |
| split-Fashion | **0.799** | refused | 0.777 | 0.894 |
| letter (13 tasks) | **0.782** | refused | 0.502 | 0.949 |
| satimage | 0.832 | 0.842 | 0.878 | 0.902 |
| wall-robot | **0.800** | (false-accept: 0.685) | 0.722 | 0.909 |
| CIFAR-100 R18 | 0.395 | 0.483 | 0.592 | 0.669 |
| CIFAR-100 ViT | 0.667 | 0.835 *(uncertified; **0.794** certified)* | 0.862 | 0.844 (NCM) |
| sheet sweeps / shapes / fsdd | 0.979 / 0.985 / 0.772 | 0.989 / **1.000** / 0.872 | ~0.99 / ~0.99 / ~0.93 | ~1.00 |

The durable claims after the rigor pass: **verified zero forgetting
everywhere; own numbers stable across unseen seeds; the lifecycle's gains are
real (arbiter improves 8 of 12, gated refusals cost nothing except robot); the
guarantee is competitive with honest replay on half the board and cheaper than
believed on strong features — but nothing here beats a properly trained joint
model, and the robot case bounds what data-free gates can promise.**

## The retention invariant, checked bitwise (development seeds)

Growth caused **exactly 0 answer changes** in every run of every experiment (toy,
split-digits/MNIST/KMNIST/Fashion, all four tightening-ablation cells, and all 12
spatial-sheet organisms). Tightening re-routes are rare (digits: 11 over 3 seeds;
MNIST: 509 over 3 seeds ≈ 1% of test, 54 wrong→right vs 118 right→wrong — slightly
net-negative and reported as such; they are logged events, not drift). ASDbrain
*measured* that training touches destroy retention (86–95% of the loss); here
touching is structurally impossible and the zero is verified, not trusted.

## B0 — toy (2-D, 5 seeds)

Development works and is visible ([toy_anatomy.png](results/toy_anatomy.png)):
coverage grows monotonically (e.g. 0.15 → 0.83, 0.23 → 0.88, 0.36 → 0.98) while
precision holds ≈ 0.94 and answers never flip. Growth was accepted on 4/5 seeds
(one founder was never improved on — reported, not hidden).

Both parents' failure modes are reproduced by controls that **must fail, and did
(5/5 seeds each)**: growth *inside* the parent (anchor injection — the mechanism
NNcontracts closed) causes 50–255 retention violations per seed; fine-tuning the
parent on the need set (touching weights — ASDbrain's mechanism) collapses covered
accuracy, e.g. 1.000 → 0.000.

## B2 — class-incremental learning (the flagship)

Single head, no task oracle, 5 tasks × 2 classes; abstention counted as error
("strict"). One organ per task. Final numbers (learned OOD geometry + veto-only
tightening; the path to them is the next section):

| method | split-digits (3 seeds) | split-MNIST (3 seeds) | split-KMNIST (2 seeds) | split-Fashion (2 seeds) |
|---|---|---|---|---|
| fine-tune | 0.168 ± 0.040 | 0.193 ± 0.003 | 0.199 | 0.201 |
| EWC | 0.307 ± 0.034 | 0.220 ± 0.022 | 0.199 | 0.221 |
| replay (100/class raw) | 0.688 ± 0.007 | 0.889 ± 0.008 | 0.772 | 0.805 |
| **organism (strict)** | **0.860 ± 0.026** | **0.896 ± 0.005** | **0.809 ± 0.001** | **0.798 ± 0.003** |
| joint upper bound | 0.935 ± 0.003 | 0.975 ± 0.001 | 0.949 | 0.882 |

The organism **beats raw-data replay on digits (+17 points) and KMNIST (+3.7), ties
it on MNIST and Fashion** — with zero stored raw samples, growth flips exactly 0 in
every run, and forgetting +0.04 to +0.11 (all of it logged coverage movement, none
of it silent drift). Placement against the published class-IL landscape (van de Ven
& Tolias 2019: EWC ≈ 0.20, SI ≈ 0.20, LwF ≈ 0.23, DGR ≈ 0.91, iCaRL ≈ 0.95): at
replay level, without a generator, without exemplars, with a certificate none of
them has.

## The learned OOD geometry (how the replay gap closed)

The path matters more than the destination; each step is a measurement.

**1. The unsupervised frontier is saturated.** A shoot-out of five proximity
geometries as pure OOD detectors (fit on one task's data, threshold at own-coverage
q, measure foreign-accept rate; `exp_ood.py`) shows raw-pixel centroids, per-class
PCA distance, a capped autoencoder, and their combination all plateau at the same
frontier on MNIST (FAR ≈ 0.23 at coverage 0.90, ≈ 0.61 at 0.99), and a discriminator
against synthetic corruptions fails outright (AUROC 0.69). No unsupervised input
geometry gets close to what class-IL needs.

**2. The contract is its own negative sampler.** The organism's unused resource is
supervision from itself: a contract's spec is a data generator (contracts note
§3.7.2). When task B arrives, each old organ gets a **veto** — a small capped net
trained on contract-sampled positives vs the real falsely-accepted B samples,
conjoined with the accept rule (accepted sets only shrink; the retention invariant
survives untouched). The naive version — train on sampled *pixels* — is a **vacuous
certificate**: it learns "synthetic vs real" and rejects everything real (self-loss
1.000). Training in the territory's **whitened PCA coordinates**, where z ~ N(0, I)
is exactly the model the spec asserts, collapses the synthetic/real gap: exclusion
0.986/0.955/0.960 (MNIST/KMNIST/Fashion) at 6–10% self-loss, with a measured
synthetic→real calibration gap of ~3 points. The whitening is load-bearing.

**3. The error decomposition redirected the design.** With old→new closed, 93.7% of
all remaining errors were **new organs accepting older classes** — and the samples
they caught were ones their own organ had *released through scalar-tightening
self-loss*. Each released sample becomes a downstream **error**, not an abstention.
Two consequences, both confirmed: birth vetoes (tightening newborns against old
contracts' samples, verified on real val data) cannot raise strict accuracy — they
only convert wrong to abstain — and scalar tightening became net-harmful.

**4. The ablation (split-MNIST, 2 seeds, identical geometry):**

| tightening | strict | coverage | precision |
|---|---|---|---|
| none | 0.408 ± 0.011 | 0.996 | 0.410 |
| scalar only | 0.742 ± 0.003 | 0.947 | 0.784 |
| **veto only** | **0.897 ± 0.005** | 0.984 | 0.911 |
| scalar + veto | 0.819 ± 0.007 | 0.941 | 0.871 |

Adding scalar tightening *on top of* vetoes costs 8 points — the release mechanism,
shown causally. The final system tightens by learned vetoes only; the false-accept
rate on incoming tasks falls from 0.46–0.87 to 0.00–0.09 across the ladder.

## B-sheet — the organism on ASDbrain's spatial substrate

The port ([cgrow/sheet_organ.py](cgrow/sheet_organ.py), [exp_sheet.py](exp_sheet.py)):
organs become spatially placed recurrent populations on one 2-D sheet, faithful to
the parent substrate — positions in [0,1]², tanh dynamics over T=8 steps, a fixed
input entry site, the hinge wiring cost c(d) = d + β·max(0, d−d₀)² on lateral *and*
input wiring, cost-weighted pruning (|W|/c(d)), and spectral capping of the
recurrent block (echo-state preserved). Growth is **directional**: a newborn organ
may wire *from* frozen predecessors' units (listening), but no edge ever points
into a frozen row — the retention theorem on a shared recurrent sheet. Organ
placement is decided by the wiring economy (cheapest input access, crowding
penalized). The contract side — PCA proximity on the input, vetoes, thresholds —
is inherited unchanged; the feature map is the only override.

Class-incremental (4 tasks × 2 classes) on ASDbrain's own stream generators,
3 seeds:

| | sweeps (temporal) | shapes (static) |
|---|---|---|
| fine-tune (mono sheet) | 0.255 ± 0.007 | 0.247 ± 0.009 |
| EWC (mono sheet) | 0.525 ± 0.049 | 0.247 ± 0.009 |
| **organism (strict)** | **0.960 ± 0.011** | **0.976 ± 0.003** |
| organism, listening off | 0.873 ± 0.050 | 0.973 ± 0.001 |
| replay (mono, 100 raw/class) | 0.999 ± 0.001 | 0.988 ± 0.004 |
| joint upper bound | 1.000 | 1.000 |

Growth flips 0 in all 12 organisms; false accepts 0.00 at every task boundary
(these stream families are well separated in input space); the organism is
*smaller* than the monolith (19k vs 30k params on sweeps — no global n²
recurrence). These tasks are easy (joint saturates), so the accuracy story is
parity-with-replay; the port's real content is what only anatomy can offer:

**Anatomy is real and priced.** Wiring is local and modular by measurement:
within-organ weight fraction 0.93 (shapes) / 0.58 (sweeps) against a 0.25 null;
mean weight-weighted edge length 0.10–0.21 vs 0.45 under random pairing.

**Lesions are causal and lower-triangular.** Silencing organ k destroys its own
tasks (diagonal damage 0.96–0.98) and — across all six listening organisms —
damages *no earlier task* (lower-triangle damage 0.000): what organs < k compute
cannot depend on organ k by construction, and the residual empirical channel (an
early task's sample caught by a later organ) never fired here. The upper triangle
prices **forward transfer**: listening is worth +8.7 points on the temporal stream
(0.960 vs 0.873 deaf) and ~nothing on the static one, and the fragility it buys is
visible — on one seed, lesioning organ 0 cost task 1 a 0.59 drop because organ 1
had leaned on organ 0's dynamics. Forward transfer through frozen anatomy:
measurable benefit, measurable dependency, zero retention risk.

Example lesion damage matrix (sweeps, seed 0; row = silenced organ, col = task):

```
0.983  0.590  -0.007  0.030
0.000  0.967   0.008  0.000
0.000  0.000   0.963  -0.005
0.000  0.000   0.000   0.962
```

ASDbrain's paper asked for anatomy where "where is this function?" is well-posed;
the ported organism answers it with an intervention, and adds "and nothing you do
later can move it."

## B-adolescence — the verified metamorphosis

After development ends, the organism can afford one biological move its invariant
otherwise forbids: **adolescence** — a massive recombination of its machinery
([cgrow/adolescence.py](cgrow/adolescence.py)). The recipe uses only what the
organism owns: sample labeled data from every organ's contract, train one
consolidated organ jointly on the pooled synthetic set (the joint problem the
cascade never solved — where frozen errors live), on the sheet re-place its units
and re-learn all local wiring from scratch under the hinge cost, then **verify**
(per-class synthetic holdout + the last task's real val) and swap it in ahead of
the old cascade, which remains as fallback so coverage cannot drop. Adolescence is
the single sanctioned exception to the no-drift invariant: every answer change is
logged and audited.

Results (2 seeds each; strict, before → after):

| benchmark | before | after | audit |
|---|---|---|---|
| digits | 0.824 / 0.871 | **0.878 / 0.915** | flips 15 right : 2 wrong |
| mnist | 0.902 / 0.891 | — | **rejected by the gate** (would regress) |
| kmnist | 0.810 / 0.808 | 0.813 / 0.806 | accepted, net ≈ 0 (flips balanced) |
| fashion | 0.795 / 0.800 | — | **rejected by the gate** |
| sheet sweeps | 0.969 / 0.944 | **0.980 / 0.984** | flips 24 right : 0 wrong |
| sheet shapes | 0.978 / 0.978 | **0.986 / 0.992** | wiring: 5600 → 2240 edges |

Three readings. Where the contract sampler is faithful (low-dimensional inputs,
the sheet streams), adolescence **fixes frozen errors data-free** (+4 to +5 points
on digits, flips overwhelmingly to-right) and **compresses**: the consolidated
sheet organ carries 60% fewer edges, halves its mean wire length (shapes
0.088 → 0.045), and standing alone is ~5× smaller than the cascade at 0.87–0.95
strict. Where the sampler is not faithful (MNIST/Fashion pixels), the consolidated
organ would regress — and the **verification gate correctly rejects the
metamorphosis** (4/4 rejections, organism unchanged). Solve-and-verify, again.

## B-renegotiate — can the contracts themselves change at adolescence?

A contract has two halves, and they behave oppositely. The **guarantee side**
(anchors, margins, thresholds) is renegotiated freely at adolescence — the
consolidated organ's are brand new — and a **refinement audit** certifies backward
compatibility by sampling the *old* contracts' territories: coverage kept 1.000,
answer agreement 0.93–1.00, and accuracy on the promised regions *improves*
(e.g. kmnist 0.79 → 0.94, sheet 0.99 → 1.00) in every accepted cell.

The **assumption/territory side** is the opposite. Head-to-head, 2 seeds
([exp_renegotiate.py](exp_renegotiate.py)): territories **inherited** from the old
contracts (union, calibrated on real data at each organ's birth) vs territories
**renegotiated** by refitting on the contract-sampled pool:

| benchmark (strict, seed 0/1) | inherit (union) | renegotiate (refit) | refit share answered by C |
|---|---|---|---|
| digits | **0.878 / 0.915** | 0.855 / 0.902 | 0.48 / 0.46 |
| kmnist | **0.813 / 0.806** | 0.811 / 0.804 | 0.33 / 0.38 |
| sheet sweeps | **0.980 / 0.984** | 0.969 / 0.945 | **0.00 / 0.00** |
| sheet shapes | **0.986 / 0.992** | 0.978 / 0.978 | 0.07 / 0.05 |
| mnist / fashion | rejected | rejected | (gate binds before policy) |

A **per-organ audit on the sheet** ([exp_sheet_audit.py](exp_sheet_audit.py),
16 organ-territories across 2 streams × 2 seeds) shows the metamorphosis keeps
its promises *uniformly*: coverage kept 1.000 in every cell, agreement 0.94–1.00
— and every sub-1.0 agreement is a **correction concentrated on the youngest
organ** (accuracy on its territory 0.938 → 0.999). One audit lesson with teeth:
the vacuous refit policy passes this audit *perfectly* (its consolidated organ
answers nothing, so the fallback reproduces old behavior verbatim). Refinement
audits certify safety, not usefulness; `share_C` is the vacuity detector, and
both are needed.

Inheritance wins every cell. The failure mode of renegotiation is the contracts
note's signature one: **quiet vacuity**. A territory refitted from its own
generated data under-accepts real data — on the sheet the renegotiated organ
answers 0–7% of test, so the organism silently falls back to its old self. Nothing
breaks, the refinement audit still passes, the gate sees no regression — the
renegotiated contract is simply *true and useless*. So the measured answer to
"can contracts change at adolescence?": **guarantees yes (certified as a
refinement by sampling), territories no — not from generated data**. Renegotiating
territories would need real data, i.e. post-adolescence recalibration as new data
flows — future work, and the one place the data-free discipline genuinely binds.

## B-cifar — the pretrained-backbone CIFAR-100 test (the wall, measured)

The make-or-break regime: class-IL on frozen ImageNet-pretrained ResNet-18
features (extracted once, cached), 10 tasks × 10 classes, class order shuffled
per seed, single head, no exemplars for ours; 2 seeds
([exp_cifar.py](exp_cifar.py)). All contestants on identical features:

| method | guarantee | strict |
|---|---|---|
| fine-tune / EWC | none | 0.07 / 0.06 |
| **organism (cascade)** | **answers never flip, bitwise** | **0.392 ± 0.004** |
| replay (20/class raw) | none | 0.495 |
| **contracted prototypes** | prototypes frozen; answers may drift | **0.510** (forced 0.552, cov 0.83) |
| NCM (SimpleCIL-style) | prototypes frozen; answers may drift | 0.552 |
| joint upper bound | none | 0.670 |

**The organism does not survive this regime in its cascade form.** With 100
fine-grained classes sharing a mediocre frozen feature space, per-task
territories overlap massively: false accepts hit 0.72–0.98 before tightening and
the subspace vetoes only halve them (residual 0.19–0.65) instead of erasing them
as on the MNIST ladder. Growth flips stay exactly 0 — the certificate holds; its
price explodes.

The instructive part is the **guarantee ladder**. Plain NCM wins here because in
a frozen shared feature space, per-class prototypes never interfere — but NCM's
*answers* still drift as classes arrive, which the cascade forbids. Wrapping
NCM's topology in our contract machinery ("contracted prototypes": per-class
anchors + PCA territories + solved margins, all order-independent) matches NCM's
forced accuracy exactly while adding abstention semantics. So the ordering reads
as a price list: bitwise answer stability costs ~16 points here (0.39 vs 0.55);
frozen-prototypes-with-drifting-answers costs ~12 (0.55 vs 0.67); no guarantee
is the ceiling. On the MNIST-family ladder the first price was ≈ 0 — class
density and feature quality set the exchange rate.

Placement against the literature, stated carefully: published pretrained-regime
methods (SimpleCIL/ADAM ≈ 0.76–0.87 on CIFAR-100) use far stronger backbones
(ViT-B/16, ImageNet-21k) — our joint ceiling on ResNet-18@224 features is 0.670,
so absolute numbers are not comparable; relative to ceiling, the prototype arms
reach 82% and the cascade 58%. The regime-appropriate topology for contracts
here is prototypes, not cascades — the contract machinery ports to it at zero
accuracy cost. The matched-backbone test is next.

## B-vit — matched backbone: ViT-B/16 ImageNet-21k (SimpleCIL's own)

Same protocol on frozen `vit_base_patch16_224.augreg_in21k` features (timm;
train subsampled 250/class; 2 seeds; [extract_cifar_vit.py](extract_cifar_vit.py)):

| method | guarantee | strict | forced |
|---|---|---|---|
| fine-tune / EWC (MLP head) | none | 0.010 / 0.013 | |
| replay 20/class (MLP head) | none | 0.252 | |
| **organism (cascade)** | answers never flip, bitwise | **0.660 ± 0.020** | 0.662 |
| **contracted prototypes** | prototypes frozen; drift logged | **0.833 ± 0.001** | 0.844 |
| NCM (SimpleCIL-style) | prototypes frozen; drift unlogged | — | 0.844 |
| joint (linear-probe head) | none | — | 0.813 |

**Contracted prototypes reach 0.833 strict / 0.844 forced at SimpleCIL's own
backbone** — level with published SimpleCIL last-step accuracy (≈ 0.81; their
average-over-steps metric is ≈ 0.876) and *above* our own linear-probe joint head
(0.813) — while adding abstention semantics (coverage 0.978, precision 0.852) and
the drift audit below. The contract machinery scales to the modern regime when
wrapped around the regime-appropriate topology.

Two more readings. **The cascade's guarantee got cheaper with feature quality**:
0.660 here vs 0.392 on ResNet-18 (78% vs 71% of the NCM reference; vetoes now cut
false accepts to 0.06–0.36). The bitwise-stability price is real everywhere but
shrinks as features separate classes better — the exchange rate is set by the
backbone. And **head-based baselines collapse in this regime** (fine-tune 0.01,
replay 0.25, joint head 0.81 < NCM 0.844): on strong frozen features, prototypes
beat trained heads, which is exactly why the literature moved here.

## B-drift — the intermediate guarantee, audited

The middle rung of the ladder ([exp_drift.py](exp_drift.py)): global
nearest-prototype routing where answers may change **only at logged
prototype-addition events**, each event audited twice — real flips on test, and a
**forecast made before the event from contract-sampled old territories alone**.

| backbone | final forced | flips right : wrong | corr(forecast, real on promised region) | real/forecast |
|---|---|---|---|---|
| ResNet-18 | 0.552 (= NCM) | 11487 : 2187 (5.3 : 1) | 0.86 / 0.98 | ≈ 1.5× |
| ViT-B/16 | 0.844 (= NCM) | 15875 : 948 (**16.7 : 1**) | 0.90 / 0.65 | ≈ 2.0× |

Final accuracy equals NCM by construction (prototype sets are order-independent);
what the contract adds is the certificate: drift is overwhelmingly benign, scoped
to events, and **forecastable in advance** on the promised region (the region old
contracts speak about) within a ~1.5–2× calibration factor. The early large "all"
flip rates (up to 0.63) live entirely on *unpromised* samples — classes no
contract had seen — which is precisely the honest scope of the guarantee.

## B-sampler — the better-generator lever, measured and redirected

The gaussian-subspace sampler was suspected as the binding constraint on
adolescence-at-scale. A direct fidelity shoot-out ([exp_sampler.py](exp_sampler.py):
real-vs-synthetic discriminability, and a "consolidation probe" — train a plain
classifier on pooled contract samples, test on real data) redirected the diagnosis:

| features | sampler | real-vs-synth AUROC | consolidation gap (real − synth) |
|---|---|---|---|
| ViT-B/16 | gaussian (pca) | **0.493** (indistinguishable) | +0.015 |
| ViT-B/16 | GMM (mix / mix48) | 0.494 / 0.503 | +0.007 / +0.009 |
| MNIST pixels | any | 1.000 | +0.08–0.10 |
| Fashion pixels | any | 1.000 | +0.05–0.07 |

**On good features the gaussian sampler is already faithful** — and the GMM buys
nothing on pixels either (the isotropic residual texture betrays every variant).
The mixture lever is dead; `pca` stays the default. The real bottlenecks were two
implementation truths found by dissection ([diagnosis on a cached pool]): the
**prototypical trunk collapses when trained on a 100-class pool** (raw accuracy
0.40 at feat 64, and 0.19 — *worse* — at feat 128/60 epochs: saturating
distance-squared logits), and the **gate was miscalibrated** (an absolute
per-class floor on the *minimum over 100 classes* implicitly hardens with class
count, an empty-acceptance artifact scored coverage holes as accuracy 0, and a
vacuous consolidation passes every safety check).

Three repairs, all in the project's grammar: consolidate into **closed-form
prototypes** at high class counts (no training — solve, never train); a
**usefulness floor** on holdout coverage (the vacuity detector, promoted into the
gate); and a **relative gate** — the consolidated organ must match the organism
it replaces per class on the same synthetic holdout, rather than clear an
absolute bar. Outcome ([exp_cifar.py](exp_cifar.py) `--adolescence --adol-mode
protos`, 2 seeds each):

| CIFAR-100 organism | before (strict) | after adolescence | audit |
|---|---|---|---|
| ViT-B/16 | 0.640 / 0.680 | **0.765 / 0.785** | flips 6 : 1 to-right; C answers 0.74–0.78 |
| ResNet-18 | 0.396 / 0.388 | — | **rejected**: C's per-class mean 0.64 vs the cascade's own 0.71 |

**Adolescence now works at 100 classes where the features are good** — a
data-free metamorphosis worth +10.5–12.5 points that turns the cascade (grown
under the full bitwise guarantee) into most of the prototype regime's accuracy:
a path between the guarantee ladder's rungs. And the rejection on ResNet-18 is
no longer a blunt refusal but a quantitative comparison on identical data. The
remaining gap to real-data contracted prototypes (0.775 vs 0.833) is the price
of deriving anchors from contract samples instead of data — measured, not
hidden.

## B-variety — longer streams, uneven tasks, real audio, mid-life metamorphosis

Four benchmarks added for variety and complexity (2 seeds each), each answering
something the earlier suite didn't.

**Tabular class-IL, including the longest stream yet.** Letter (26 confusable
classes, **13 tasks × 2**), satimage (36-d, 3 tasks), wall-robot (24-d, 2 tasks):

| | organism (strict) | replay | EWC | joint monolith |
|---|---|---|---|---|
| letter, 13 tasks | **0.773 ± 0.009** | 0.290 | 0.062 | 0.750 |
| satimage | **0.821** | 0.709 | 0.284 | 0.813 |
| robot | **0.799** | 0.618 | 0.631 | 0.818 |

The headline: on letter the organism **beats the jointly-trained monolith**
(0.773 vs 0.750) while learning classes two at a time — on hard multimodal
tabular data, per-task specialists are collectively a *better* representation
than one net, not merely a retention-safe one. Growth flips stayed 0 through 12
tightening events per organ.

**The literature's uneven protocol (B50: 50-class base + 5 × 10, CIFAR-100/ViT).**
The cascade improves to 0.689/0.695 strict (from 0.660/0.680 at 10×10 — a large
first organ means fewer frozen boundaries), and **contracted prototypes are
protocol-invariant to the digit: 0.833/0.844, identical to 10×10** — exactly what
order-independence predicts, now confirmed empirically.

**Real audio on the sheet (FSDD spoken digits, 5 tasks × 2).** The spatial
organism reaches 0.800/0.727 strict vs fine-tune 0.20, EWC 0.22 (replay 0.93,
joint 0.95 — small data, high speaker variability). The veto machinery works on
real audio (false accepts 0.45–0.79 → 0.01–0.06), listening is worth up to +7.7
points, and the anatomy remains real (within-organ weight 0.48–0.54 vs 0.20
null). Honest placement: mid-pack — 3.5× the regularization methods, 13–20
points behind replay.

**Mid-life adolescence composes safely — and, after one structural fix,
profitably.** Metamorphosing after task 5 of 10 (CIFAR-100/ViT) is accepted,
helps locally (tasks 1–5: 0.75 → 0.84), and **the invariant survives it**
(phase-2 growth flips exactly 0). The first version regressed overall
(0.588/0.622 vs 0.640/0.680 without adolescence): the consolidated organ's
opaque union territory could not take vetoes, so it front-ran and wrongly
claimed **19% of later-class samples**. The fix is structural — a union of PCA
territories *is* a PCA territory with more classes, so territories are now
**merged into a first-class, veto-capable prox**
([merge_pca_family](cgrow/proximity.py)). Rerun: the frozen-error channel
collapses to **4–5%** and mid-life adolescence turns net-positive:
**0.697/0.713**, above the no-adolescence baseline. Late inhibition over a
consolidated part works.

## B-pfc — the full lifecycle: growth → adolescence → prefrontal arbiter → ultimate consolidation

Biology's ordering (sensory cortex consolidates first, prefrontal control
matures last, operating plastically over frozen parts before consolidating
itself — Gogtay et al. 2004, posterior-to-anterior maturation, myelination into
the mid-20s) mapped onto the organism and run end-to-end on CIFAR-100/ViT
([exp_pfc.py](exp_pfc.py), 2 seeds). The **arbiter** is a small net that reads
every organ's evidence (distance, margin, territory score, accept bit) and
learns *who to trust* — it answers nothing itself, it routes; it is trained
purely on contract-sampled data; and it faces the same verify-then-freeze gate
as adolescence, its answer changes audited as one logged event.

| lifecycle stage | strict (seeds 0/1) | guarantee at this stage |
|---|---|---|
| growth (cascade) | 0.640 / 0.680 | answers never flip, bitwise |
| + adolescence | 0.765 / 0.785 | one verified, audited metamorphosis |
| **+ PFC arbiter, consolidated** | **0.827 / 0.831** | one more verified, audited event; then frozen |
| contracted prototypes (reference) | 0.833 | drift logged throughout |
| NCM (reference) | 0.844 | none |

**The full lifecycle closes the guarantee gap to 0.2–1.3 points.** The
routing-error hypothesis is confirmed: most of the cascade-vs-prototype gap was
misrouting, and an executive module trained on the contracts alone recovers it
(gate: holdout per-class mean 0.97 vs the pre-arbiter organism's 0.94; last-task
real val *improves* 0.74 → 0.80; logged flips ~4:1 to-right). The organism ends
fully frozen, with every change in its life either forbidden (growth), gated
(two consolidations), or logged (tightening, vetoes, arbitration) — at
essentially the accuracy of the plastic regime.

## B-lifecycle — the full lifecycle on all twelve benchmarks

The complete lifecycle (growth → adolescence, gate may refuse → prefrontal
arbiter, gate may refuse → ultimate consolidation) run everywhere
([exp_lifecycle.py](exp_lifecycle.py), 2 seeds, strict accuracy; *refused*
means the gate left the organism unchanged):

| benchmark | growth | + adolescence | + arbiter |
|---|---|---|---|
| split-digits | 0.848 | 0.893 | **0.959** (joint 0.935) |
| split-MNIST | 0.897 | refused | **0.918** |
| split-KMNIST | 0.809 | refused | **0.826** |
| split-Fashion | 0.798 | refused | refused (0.798) |
| letter (13 tasks) | 0.773 | refused | **0.779** (joint 0.750) |
| satimage | 0.821 | refused | **0.836** |
| wall-robot † | 0.799 | refused | 0.686 |
| CIFAR-100 (frozen R18) | 0.392 | refused | **0.489** (replay 0.495) |
| CIFAR-100 (ViT-B/16) | 0.660 | 0.775 | **0.831** |
| sheet: sweeps | 0.957 | 0.981 | refused (0.981) |
| sheet: shapes | 0.978 | refused | **1.000** |
| sheet: spoken digits | 0.764 | refused | **0.867** |

**The arbiter is accepted on 10/12 and improves every benchmark it joins**
(+1 to +11 points): digits ends *above* its joint bound, shapes reaches 1.000,
real audio gains +10, and even weak-feature CIFAR recovers +10 (to replay
level) — confirming that misrouting, not representation, was the cascade's
main deficit everywhere. Adolescence is accepted exactly where the contract
sampler is faithful (3/12), and the arbiter's own gate correctly refuses where
nothing remains to improve (Fashion, sweeps).

† wall-robot is the study's one measured gate false-accept: the arbiter passes
both gate signals (synthetic +0.14, real last-task val +0.04) yet regresses on
test. Its fidelity score (AUROC 0.81) is elevated but below our pre-registered
flag threshold; we report it as the honest boundary of data-free gating rather
than re-tune the flag to catch the one failure we know about.

## Final scoreboard vs published results

*Development-seed scoreboard, kept as written. The evaluation-seed version is
"Final comparative table vs published results" near the end of this file, and
the current tables are in [BENCHMARKS.md](BENCHMARKS.md).*

| setting | best published | **this work** |
|---|---|---|
| split-MNIST, exemplar-free | generative classifier 0.938; DGR 0.913 | 0.893 certified (0.915 uncertified), + certificates |
| split-MNIST, with stored data | iCaRL ≈0.946 (2000 images); GDumb ≈0.97 | 0.918, zero stored data |
| CIFAR-100, ViT-B/16-in21k | SimpleCIL 0.813 last; RanPAC 0.922 | 0.844 forced / 0.831 lifecycle |
| CIFAR-100, from scratch | DER/FOSTER ≈0.65–0.70 (trained backbone + buffers) | not comparable (frozen R18 ceiling 0.67) |
| letter/satimage/robot/FSDD/sheet class-IL | none published | 0.69–1.00 vs same-data joint bounds |

A paper draft assembling all of this is at [paper/main.pdf](paper/main.pdf)
("Forgetting Is a Certificate: A Verified Developmental Lifecycle for
Continual Learning"), with every number regenerable via `claims.py`.

## B-daring — sleep, apoptosis, and the market (evaluation seeds throughout)

Three speculative mechanisms, each run under the frozen protocol: one works,
one resolves into a positive claim about the anatomy after five escalating
certificates, and one fails, is diagnosed, and is then **repaired** — the
repaired version ties the central arbiter on the flagship benchmark.

**Sleep cycles work — if the organism dreams.** Nightly micro-consolidation
(rebuild the prototype organ C from all contracts after every task) plus a
nightly **dream** (retrain the arbiter on contract samples), both gated per
night ([exp_sleep.py](exp_sleep.py)). Sleep-with-dreams matches the one-shot
end-of-life lifecycle — digits **0.950** vs 0.948, cifar-ViT **0.813** vs
0.835 (one seed's dreams kept being refused) — while adding the **anytime
property**: the organism is consolidated at every point in its life, and the
mid-life failure mode cannot occur. Sleep *without* dreams costs 4–6 points
(nightly C front-runs the cascade and starves new organs; the dream's routing
rehearsal compensates) — the biological reading writes itself. Cost: churn
(~8,700 logged flips over a ViT life vs ~1,100 for one metamorphosis); every
night is gated (kmnist's nights are correctly refused almost entirely).

**Apoptosis, v1–v4: the mechanism works, and four certificates leaked.** Gated
organ death ([exp_apoptosis.py](exp_apoptosis.py)): a redundancy certificate
sampled from the dying organ's own contract, death audited, and the contract
archived to a **genome** that provably still powers future consolidations
(rebuilt-and-gated C from live + dead specs: works). Controls behave (0 deaths
on unconsolidated organisms, 7/9 runs); compression is dramatic where C
shadows (**2.2M → 0.3–0.5M params, 4–8×, 7–9 organs retired**). But accuracy
leaks ~8–10 points through **four escalating certificates**: local synthetic
checks (−8/−10, the ratchet), globally anchored synthetic (−10/−8 — not a
ratchet after all), a 200-sample real escrow (−8/−7), and a 1,000-sample real
escrow (identical — the global check never binds).

**Apoptosis, resolved (v4): the leak was per-class, and closing it kills
nothing.** The fifth certificate adds **per-class floors** against the original
organism (preregistered `CLASS_SLACK` = 0.05, escrow of *N* real samples per
class used only by the gates, never for training). Result on cifar-ViT dev
seeds, at two escrow sizes (15/class and 60/class): **0 deaths, 0 leak** —
strict accuracy exactly unchanged (0.765 and 0.786), every candidate refused.
The refusal diagnostic pins the mechanism directly rather than inferring it.
For a typical candidate organ: territory coverage without it **0.97–0.99**,
global accuracy without it equal to or *better* than the anchor (e.g. 0.843 vs
0.831), validation strict usually *improved* (0.749 vs 0.741) — **every
mean-based gate waves it through** — while a single class drops **6.7–13.3
points**. Of 20 refusals, **19 cite the per-class gate and 15 cite it alone**;
averaged over all refusals the global-accuracy gate moves by **+0.001** and the
validation gate by **−0.007**, both far inside their slacks.
A control rules out measurement granularity: at 60 samples/class the per-class
resolution (0.017) is far below the slack (0.05), and the drops persist
(0.033–0.117). Two conclusions, one negative and one positive. Negative: the
~8–10 point leak of v1–v3 was **real damage, invisible to means** — nine deaths
each quietly costing one class ~13 points compound into it, which is why a
1,000-sample escrow measured no worse than a 200-sample one. Positive: **growth
produced no per-class-redundant organ.** The anatomy is minimal in a strong
sense — each organ is the sole competent authority for at least one class
pocket — so a *sound* death certificate correctly retires nobody, and the
dramatic 4–8× compression reported by the earlier certificates was compression
paid for in unmeasured per-class accuracy. The standing conclusion holds with a
sharper edge: **death is the most expensive certificate in the lifecycle**, and
its price is per-class held-out data, not more of it.

**The bidding market fails, and the failure is the finding.** Routing by
auction ([exp_market.py](exp_market.py)): each organ publishes a price curve —
a logistic from its own evidence to P(correct), fitted on its own stored birth
statistics; highest bid answers. Fully decentralized, no shared data, no
central trainer. Result: **anti-calibrated** (pool Brier 0.23–0.80; digits
0.19–0.44 strict vs the plain cascade's 0.88, cifar-ViT 0.42 vs 0.67), losing
to both the dumb cascade and the trained arbiter (0.72–0.97) everywhere. The
mechanism echoes the whole project: an organ's self-model, fitted only on its
own birth distribution, has never seen negatives — its prices are confident
nonsense outside its territory. **Routing quality lives in the joint
distribution of organ correctness, which no organ can know alone: prices need
a market-maker.**

**The market-maker: decentralized routing matches central arbitration after
all.** The middle design is now measured ([exp_market.py](exp_market.py)
`--pricing pool`, [summarize_market_maker.py](summarize_market_maker.py),
eval seeds 2–4). *Identical* auction, *identical* per-organ price model — one
tiny logistic on the same three evidence features — with a single change: each
curve is fitted on the **shared contract pool** rather than on the organ's own
birth statistics, so every organ sees the others' territories as negatives.
Decentralized model, joint data. No central network, no real data, no joint
training.

| eval seeds 2–4 | cascade | self-priced | **market-maker** | arbiter |
|---|---|---|---|---|
| digits, raw | 0.877 ± 0.028 | 0.279 ± 0.116 | 0.843 ± 0.002 | 0.958 ± 0.003 |
| kmnist, raw | 0.805 ± 0.004 | 0.723 ± 0.032 | 0.611 ± 0.014 | 0.807 ± 0.016 |
| cifar-ViT, raw | 0.667 ± 0.010 | 0.421 ± 0.125 | **0.681 ± 0.003** | 0.729 ± 0.005 |
| digits, +C | — | 0.207 | 0.940 | 0.969 |
| **cifar-ViT, +C** | — | 0.437 ± 0.137 | **0.836 ± 0.002** | 0.835 ± 0.003 |

On the flagship benchmark, consolidated, the market-maker **ties the central
arbiter** (0.836 ± 0.002 vs 0.835 ± 0.003) and beats first-accept even on the
raw cascade (0.681 vs 0.667). The earlier "decentralized routing fails" verdict
was an artifact of *how the prices were fitted*, not of decentralization.

**Two axes, dissociable.** Calibration is cured everywhere — pool Brier
improves **200× on digits** (0.707 → 0.004), 11× on cifar-ViT (0.519 → 0.047),
2× on kmnist (0.308 → 0.178) — but accuracy does not follow: +0.56 digits,
+0.26 cifar-ViT, **−0.11 kmnist**. Honest prices can route *worse* than
dishonest ones, so calibration quality and routing quality must be reported
separately.

**Why: the generalist is a participant, not a price index** ([exp_market.py]
(exp_market.py) `--force-c`, [summarize_indexed.py](summarize_indexed.py), eval
seeds 2–4). The first explanation offered for the kmnist regression — that
specialists need a consolidated generalist to *price against* — was tested
directly by indexing the reserve to C's bid (a specialist must outbid λ·bid_C,
λ swept on the contract pool) and by exposing a gate-**refused** C as a price
reference only (`consolidate_protos(force=True)`; never trusted as an
answerer). It is wrong in its mechanism and right in its ingredient.

| eval seeds 2–4, mean strict | maker, no C | maker, +C bidder | gain | C-indexed | index gain | C alone | C forced |
|---|---|---|---|---|---|---|---|
| kmnist | 0.611 | **0.703** | +0.092 | 0.698 | −0.005 | 0.187 | 3/3 |
| digits | 0.843 | **0.933** | +0.091 | 0.942 | +0.009 | 0.561 | 2/3 |
| cifar-ViT | 0.681 | **0.836** | +0.155 | 0.832 | −0.004 | 0.658 | 0/3 |

Adding C to the auction as an *ordinary bidder* is worth **+0.09 to +0.16**
everywhere. Indexing the reserve to its bid then adds **nothing** (−0.005,
+0.009, −0.004), and the λ sweep confirms it from the other side: it selects
the grid floor in **7 of 9 runs**, pushing the indexed rule back toward the
flat auction — what an optimizer does with an uninformative knob. The
mechanism is not "specialists price themselves against C" but "**C absorbs the
generic inputs specialists were mis-winning**". A striking corollary: on kmnist
the forced C scores **0.187 alone** — the consolidation gate was entirely right
to refuse it as an answerer — and it still delivers +0.092 as a bidder.
**Answering and pricing are different jobs, and a module can be too weak for
the first while remaining valuable at the second** (measured for a refused C on
5 of 9 runs). The `indexed_abstain` control, which lets nobody answer when no
specialist outbids C, is uniformly worse at ~0.67–0.82 coverage, so none of the
gain is the indexed *rule* — it is C answering and bidding.

**And why kmnist resists at all: it has no routing headroom.** The gap the
arbiter opens over first-accept bounds what *any* router can win, and it is
benchmark-specific:

| eval seeds 2–4 | cascade | arbiter | **headroom** | maker +C | vs cascade | vs arbiter |
|---|---|---|---|---|---|---|
| kmnist | 0.805 | 0.807 | **+0.001** | 0.703 | −0.102 | −0.103 |
| digits | 0.877 | 0.958 | +0.082 | 0.933 | +0.056 | −0.029 |
| cifar-ViT | 0.667 | 0.729 | +0.062 | **0.836** | +0.169 | **+0.002** |

On kmnist the cascade already *is* the optimal router (the arbiter beats it by
0.001), so re-routing by price can only destroy the birth-order prior — the
market's −0.10 there is a property of the benchmark, not a defect of the
market. Where headroom exists, the market-maker takes most of it: +0.056 over
first-accept on digits and **+0.169 on cifar-ViT, where it edges the central
arbiter (+0.002)**.

**The law, final form.** (1) Prices need joint *data* — a market-maker
suffices and the contract pool is enough; no real data, no central trainer.
(2) The auction needs a generalist **participant**, not a generalist price
index; even a gate-refused one works. (3) Markets can only win where the
cascade leaves routing headroom, which is measurable in advance as
arbiter − cascade. The residual gap to the arbiter where it remains (digits
−0.029) is a model-class limit: independent per-organ logistics cannot express
the cross-organ interactions ("A and B both bid high, but B is right whenever
A's margin is small") that the arbiter's multi-label net over the joint
evidence vector can. **On the flagship benchmark, after consolidation, the
market-maker is a decentralized equal of the central arbiter** with no central
trained state; the arbiter remains the mechanism of record everywhere else.

## B-ensemble — redundant growth: what a duplicate organ is worth

Three linked measurements, all on twin organisms: the same class split grown
independently at seeds *s*, *s*+100, … and concatenated into one cascade, so
every class is covered *k* times. No knob was tuned for any of this (constants
inherited from the apoptosis and PFC work; the pass/fail rule below was fixed
before running), so all five seeds are reported together.

**1. The death certificate can fire — the control session-15 was missing.**
Session-15 measured 0 deaths under per-class certificates and concluded the
grown anatomy is *minimal*. That conclusion is only sound if the certificate is
capable of firing at all, so [exp_redundant.py](exp_redundant.py) builds an
anatomy that is redundant **by construction** and asks it to. Preregistered:
PASS if ≥25% of organs retire within the 0.02 accuracy slack, FAIL if <10%
retire (which would mean the gate is simply too strict and session-15's
positive claim collapses into a statement about the gate).

| twins, 5 seeds | verdicts | organs retired | accuracy cost | params |
|---|---|---|---|---|
| digits | 3 PASS, 2 PARTIAL, **0 FAIL** | 3.4 ± 1.4 of 10 (34%) | 0.010 ± 0.009 | 70k → 35k |
| kmnist | **5 PASS** | **5.0 ± 0.0 of 10 (50%)** | 0.003 ± 0.001 | 3.14M → 1.05M |

On kmnist the certificate retires **exactly one twin from each of the five
pairs, on every seed, at a cost of 0.003** — the mechanism doing precisely what
it was designed to do. So *the gate fires when redundancy is real*, and
session-15's claim stands as stated: no organ in the **grown** anatomy is
dispensable at per-class resolution. Two honest caveats: digits seed 3 spent
0.024 for 3 deaths, just outside the preregistered slack (hence PARTIAL), and
kmnist's deaths cause 99 ± 39 logged answer flips at ~zero accuracy change —
flips and damage are not the same quantity, and only the latter is gated.

**2. The merged-twin gain is not an ensemble — it is a coverage extension.**
Concatenating twins looked like a free ensemble (0.909 from parents of 0.829
and 0.873 on one seed). It cannot be one: in a first-accept cascade the
**prefix wins every contest**, so organs appended after organism A answer only
where A *abstained*. [exp_ensemble.py](exp_ensemble.py) confirms the theorem
bitwise and decomposes the channel (digits, 5 seeds):

| k organisms | strict | coverage | **precision on covered** | newly answered | acc there | flips vs prefix |
|---|---|---|---|---|---|---|
| 1 | 0.867 ± 0.029 | 0.934 ± 0.039 | 0.929 ± 0.018 | — | — | **0** |
| 2 | 0.898 ± 0.012 | 0.970 ± 0.012 | 0.926 ± 0.019 | 16.2 | 0.854 | **0** |
| 3 | 0.903 ± 0.009 | 0.979 ± 0.014 | 0.923 ± 0.022 | 4.0 | 0.717 | **0** |
| 4 | 0.905 ± 0.010 | 0.981 ± 0.014 | 0.922 ± 0.022 | 1.2 | 0.417 | **0** |

Precision on covered inputs does not rise — it drifts very slightly *down*
(0.929 → 0.922) — while coverage climbs 0.934 → 0.981. The entire +0.038 is
abstention-filling at the precision the organism already had, with **zero
flips at every k**: a twin never overrules its sibling, by construction and in
fact. Returns diminish sharply and the marginal organ is poor: the samples a
new twin newly answers are handled at 0.854, then 0.717, then 0.417 — far below
the organism's own 0.92 — so a second twin is worth most of the gain and the
fourth is worth 0.002. On kmnist (3 seeds) the effect nearly vanishes
(0.808 → 0.814 ± 0.002) precisely because coverage starts at 0.984: **no abstentions,
no gain**, which is the coverage story's sharpest prediction.

**Width is not a substitute.** One organism widened four-fold scores
0.872 ± 0.025 at 166k parameters against the four twins' 0.905 at 105k — worse,
with 59% more parameters, at unchanged coverage (0.945). Whatever redundant
independent growth buys, it is not capacity.

**Voting buys a little, genuinely.** Over the same 20-organ pool, majority vote
among the organs that *accept* scores 0.915 ± 0.006 against first-accept's
0.905 ± 0.010 — a small but consistent **+0.010**, the one real ensembling
effect here. Voting over *forced* predictions does not help (0.903 ± 0.012),
and the pool-priced market-maker is well below the cascade (0.861 ± 0.008),
exactly as the session-16 law predicts: the merged twin pool contains no
consolidated generalist to price against.

**3. Where the arbiter's advantage comes from.** The arbiter over the merged
pool reaches 0.963 ± 0.004, well above first-accept. Since
[exp_pfc.py](exp_pfc.py)'s `evidence()` hands it every organ's **forced**
prediction and `arbiter_predict()` answers with the argmax organ *whether or
not that organ accepted*, it has three powers the cascade lacks, and
[exp_arbiter_source.py](exp_arbiter_source.py) separates them exactly (with
*c*, *a* ∈ {0,1} for cascade/arbiter correctness and abstention counted as
wrong, each partition contributes Σ(aᵢ−cᵢ)/n and the parts sum to Δ):

| digits, 5 seeds | k = 1 | k = 4 |
|---|---|---|
| cascade → arbiter | 0.867 → 0.959 (**+0.092**) | 0.905 → 0.963 (**+0.058**) |
| re-route (answer differently where the cascade answered) | +0.045 ± 0.013 | **+0.058 ± 0.019** |
| fill (answer where the cascade abstained) | **+0.052 ± 0.034** | +0.011 ± 0.010 |
| drop (θ-abstain where the cascade answered) | −0.005 ± 0.002 | −0.011 ± 0.005 |
| answers from a **non-accepting** organ | 8.7% (71% of the gain) | 8.6% (**45% of the gain**) |

At k=1 the arbiter mostly *fills*; at k=4, where the cascade already covers
0.981, filling has nothing left to do and essentially all of the gain is
**re-routing — 134 corrections against 4 regressions across the five seeds
(33:1)**. This is genuine ensembling: the arbiter picks a better organ on
inputs the cascade had already answered.

**The guarantee-relevant caveat.** Roughly 8.6% of the arbiter's answers come
from an organ that did **not** accept the input, and they carry 45% of its
advantage at k=4 (71% at k=1); about 31% of its re-routes also go to a
non-accepting organ. Half the arbiter's benefit is therefore re-routing *inside*
certified territory, and half is bought by answering *outside* it — where no
contract was willing to speak. The arbiter is not purely a router over
certified answers, and the report should not imply that it is.

**4. The caveat is fixable: a territory-respecting arbiter.** Restrict the
arbiter's choice to organs that **accept** the input (abstain when none do) and
every answer is contract-covered by construction —
[exp_arbiter_source.py](exp_arbiter_source.py) verifies the outside-territory
rate is exactly **0.0000** on all ten runs. The question is what that costs:

| digits, 5 seeds | cascade | free arbiter | **territory-respecting** | gain kept |
|---|---|---|---|---|
| k = 1 | 0.867 (cov 0.934) | 0.959 (cov 0.976) | 0.892 ± 0.031 (cov 0.906) | **28%** |
| k = 4 | 0.905 (cov 0.981) | 0.963 (cov 0.973) | **0.946 ± 0.007 (cov 0.956)** | **72%** |

At k=1 gating is expensive (28% of the gain kept) because the free arbiter's
advantage there is mostly *filling*, which gating forbids by definition. At
k=4 it is cheap — **72% kept, +0.041 over first-accept** — because redundant
coverage has already eliminated the fill channel, leaving re-routing, which
gating largely permits (**122 corrections vs 4 regressions, 30:1**, against the
free arbiter's 134:4). The residual cost is coverage: the gated arbiter
abstains a little more than the cascade (0.956 vs 0.981).

On digits, then, redundancy and *certified* arbitration are complements even
though redundancy and the free arbiter are substitutes: redundant growth removes
precisely the channel certification forbids, so **twins + territory-respecting
arbiter reach 0.946 with every answer inside some organ's certified
territory**, against 0.963 uncertified and 0.905 for first-accept — 1.7 points
to recover the 45% that was uncovered.

**5. On the flagship, this reverses — and the flagship is what the headline
claim rests on.** The lifecycle's CIFAR-ViT number is a *free*-arbiter number,
so [exp_arbiter_cifar.py](exp_arbiter_cifar.py) reruns exp_pfc's exact phase-2
and phase-3 protocol on the cached organisms (same 3,500-sample contract pool,
same rng, same θ grid) and adds the gated variant (eval seeds 2–4):

| CIFAR-ViT, eval seeds 2–4 | strict | coverage | contract-covered? |
|---|---|---|---|
| growth (cascade) | 0.667 ± 0.010 | 0.974 | yes |
| adolescence (`[C]`+cascade) | 0.776 ± 0.003 | 0.984 | yes |
| **free arbiter** (the headline) | **0.836 ± 0.004** | **1.000** | **94% of its gain is not** |
| territory-respecting arbiter | 0.795 ± 0.003 | 0.955 | yes (0.0000 outside, 3/3) |
| contracted prototypes | **0.835 / 0.844 forced** | 0.978 | **yes** |

The gated arbiter keeps only **32%** of the gain here (+0.019 of +0.060) against
72% on digits, and the caveat is *worse*: outside-territory answers are 15.6% of
all answers but carry **94%** of the free arbiter's advantage. The mechanism is
visible in one number — of the free arbiter's 3,209 re-routes (1,959 wrong→right
vs 423 right→wrong, 4.6:1), only **18.3% land on an organ that accepted the
input**, where on digits it was 69%. At 100 fine-grained classes the territories
are a poor guide to who should answer, and the arbiter's accuracy is bought
precisely by disregarding them. Two corollaries. **θ never binds** for the free
arbiter (coverage 1.000): it answers everything, which is to say it has stopped
behaving like a contracted system. And **exp_pfc's phase-4 acceptance gate
accepts both variants 3/3** — the lifecycle's own governance cannot see whether
its arbiter respects territories, which is a real blind spot.

**But the conclusion is not that certification is unaffordable here — it is that
the arbiter was the wrong way to buy it.** Contracted prototypes reach
**0.835 strict / 0.844 forced, fully certified**, statistically indistinguishable
from the free arbiter's 0.836 and 4 points above the gated arbiter. On CIFAR-ViT
the entire apparent contribution of the uncertified arbiter is recoverable by
changing *topology* rather than relaxing the contract. So the honest
recommendation inverts by regime: where organs' territories are well calibrated
(digits) arbitration is worth certifying and gating costs 1.7 points; where they
are not (100-class CIFAR) skip the cascade-plus-arbiter route and use the
prototype organ, which is certified by construction and just as accurate.

**Redundancy and the *free* arbiter are substitutes, not complements.** Redundancy
moves the cascade 0.867 → 0.905 (+0.038); the arbiter alone already reaches
0.959 at k=1; stacking them adds **+0.004 ± 0.006**, straddling zero. Both
mechanisms climb toward the same ≈0.96 ceiling by different routes — the twins
by buying coverage, the arbiter by fixing routing — and the arbiter gets there
on its own. This is the session-16 headroom picture again: the arbiter defines
the ceiling, and every cheaper mechanism is a partial climb toward it.

## B-negotiate — the market's real job: agreeing the contracts

Every market variant so far priced organs at **inference** time (who answers
this query?) and the verdict ran from bad to mixed. But the measured defect was
never in routing, it was in the **boundaries**: on CIFAR-ViT only 18.3% of the
arbiter's beneficial re-routes land on an organ that accepted the input
(B-ensemble), which is exactly why 94% of its gain is not contract-covered. So
put the market where the contracts are written ([exp_negotiate.py]
(exp_negotiate.py)). Each organ publishes a pool-calibrated price curve — the
one thing the market demonstrably does well (Brier 0.71 → 0.004) — the organs
bid over contested regions of the shared contract pool, and every region an
organ **loses** is removed from its territory by a **veto**. Vetoes are
monotone, so the renegotiated contract is still a finite checkable spec and
first-accept over the new boundaries stays **100% contract-covered**. Two arms,
both data-free (pool labels come from the spec itself): `market` cedes where a
rival outbids the incumbent by a margin; `oracle` cedes what the incumbent gets
*wrong* and someone else gets *right* — the upper bound on what any boundary
agreement could buy.

**The preregistered self-loss budget is a null, for a structural reason.** At
the inherited veto budget of 0.02 **not one veto survives** on any seed
(+0.0000, 5/5). Ceded points are the organ's own-class-*shaped* points, so
*"I lose a bidding contest here"* is **not a geometric property** expressible in
that organ's whitened coordinates: a veto can exclude what looks foreign, never
what looks native but is handled better elsewhere. The budget was re-fixed to
0.10 on **dev seeds 0–1** (where it saturates — 0.10, 0.30 and 1.0 are
identical) and then applied unchanged to evaluation seeds.

**On digits it works, and the market matches the oracle.**

| digits, eval seeds 2–4 | strict | Δ vs cascade | certified? |
|---|---|---|---|
| cascade | 0.877 | — | yes |
| **market-negotiated contracts** | **0.892 ± 0.021** | **+0.0148** | yes |
| oracle-negotiated (upper bound) | 0.892 ± 0.018 | +0.0148 | yes |
| territory-respecting arbiter | 0.907 | +0.030 | yes |
| free arbiter | 0.960 | +0.083 | no (45% outside territory) |

Market and oracle agree to four decimals: **pool-calibrated bids identify the
same contested regions ground truth would**, so the market is a sound
*agreement* mechanism even though it was a poor *routing* one. Fifteen vetoes
move 70 test answers, 36 right against 16 wrong; the gate accepts every arm on
every seed; coverage slips 0.955 → 0.938. Negotiation buys about **half** what
the territory-respecting arbiter buys, both fully certified — but it needs **no
inference-time trained component at all**: what ships is a redrawn spec, not a
network.

**On CIFAR-ViT it fails, and the failure names a law.** Excluding the
consolidated generalist, the oracle finds only **23 contested points** in a
2,800-sample pool: after adolescence `C` answers ~76% of inputs, so the
specialists are barely routed anything to argue over. Letting `C` bid and cede
(`--include-c`) unlocks the contested set — and makes things *worse*:

| CIFAR-ViT, `C` participating (eval seeds 2–4) | Δstrict | ceded | vetoes | moved (right / wrong) |
|---|---|---|---|---|
| market | −0.0001 ± 0.0002 | 1,466 | 28 | 49 (13 / 17) |
| **oracle** (knows the answers) | **−0.0045 ± 0.0014** | 5,111 | 181 | 505 (**73 / 209**) |

A ground-truth oracle, ceding exactly the points `C` gets wrong where a
specialist is right, **loses accuracy**. The mechanism is exact: **a veto
removes a claim but cannot direct it.** Boundary renegotiation is *subtractive*,
so first-accept hands each ceded point to the next accepter in birth order, not
to the organ that won the bid. With five organs on digits the fallback is
usually right; with eleven organs over 100 fine classes it usually is not.

**The fallback ratio predicts this in advance.** Points moved right : moved
wrong tracks the outcome monotonically across every configuration run:

| configuration | fallback ratio | Δstrict |
|---|---|---|
| digits dev, oracle | 3.00 | +0.011 |
| digits dev, market | 2.33 | +0.009 |
| digits eval, market & oracle | 2.25 | **+0.015** |
| CIFAR, market (`C` in) | 0.76 | −0.000 |
| CIFAR, market (`C` out) | 0.50 | −0.000 |
| CIFAR, oracle (`C` in) | 0.35 | **−0.0045** |

Above 1 renegotiation pays, below 1 it costs, and the ratio is computable **on
the contract pool before committing to any change** — so an organism can decide
whether to renegotiate at all without touching real data. The deeper statement:
**the arbiter's power is selection, and no monotone boundary edit can replicate
selection.** A contract change can only delete a claim and hope the cascade's
next accepter is the right one; that is the price of keeping every answer inside
a territory. Finally, the gate accepted the −0.0045 oracle arm on all three seeds —
the same phase-4 blind spot already reported for the arbiter, now seen a second
time: **our gates certify per-class quality on contract samples, and neither
catches a change that is quietly negative on real test data.** Both are repaired
in B-gatefix — a data-free territory check and a McNemar escrow on held-out
validation refuse the arbiter and this oracle arm respectively, while still
accepting the two changes that deserve it.

## B-gatefix — repairing the acceptance gate

Two changes walked past the lifecycle's own gate and were reported above as a
blind spot: the **free arbiter**, which buys 94% of its CIFAR-ViT gain by
answering outside every certified territory (accepted 3/3), and the
**negotiation oracle arm**, quietly −0.0045 on real test (accepted 3/3). Both
slipped through because the phase-4 gate scores per-class strict accuracy on
**contract samples** plus the last task's real validation, and neither signal
can see either failure. [cgrow/gates.py](cgrow/gates.py) adds the two missing
checks and keeps the original one, so every verdict names what fired
([exp_gatefix.py](exp_gatefix.py)).

**A. Territory — data-free and exact.** Every answer must come from an organ
that *accepted* the input. This needs no data, no labels and no threshold, and
it also repairs a **representational** gap: the verdict now reports `certified`
separately from `accepted`, so **"beneficial but uncertified"** — the honest
description of the free arbiter — becomes sayable at all. The old gate had no
vocabulary for it.

**B. Escrow — a paired test on held-out real data.** Two designs failed first,
and both failures are worth recording because each is a trap the earlier
sections walked into.

* *Multiple comparisons.* A fixed 0.05 per-class floor over 100 classes at ~60
  samples each **false-refused `arbiter_gated`** — a territory-clean change
  worth +0.024 on test and +0.031 on escrow — because one class in 100 lost 4
  samples of 60. The minimum of 100 noisy per-class estimates falls below any
  small fixed slack almost surely.
* *Escrow provenance.* Drawing the escrow from **train** flattered every
  candidate: it scored the genuinely −0.0048 oracle arm at **+0.0003**, because
  the organs fit train better than test.

The repaired check draws its escrow from the **held-out validation split**,
judges overall quality by a one-sided **McNemar test on discordant pairs**
(z ≥ −1.645; only discordant pairs carry information, so the concordant
majority stops adding noise), flags a class only when its own paired test
survives a Bonferroni correction (z_crit 4.68 at 100 classes), and keeps an
unconditional **hard floor at 0.25** so the apoptosis failure mode — a single
class collapsing — stays catchable regardless of significance.

**All twelve verdicts are now correct** — four candidates × three
evaluation seeds (CIFAR-ViT; a 60/class escrow request is capped by the
split, so the gate consumes the **entire held-out validation set** —
2,500 samples, 15–38 per class):

| candidate | test Δ | old gate | **repaired gate** | escrow z | why |
|---|---|---|---|---|---|
| free arbiter | +0.0606 ± 0.0020 | ACCEPT 3/3 | **REFUSE 3/3** | +10.10 | territory: 14.2% ± 1.3% uncertified |
| gated arbiter | +0.0223 ± 0.0020 | ACCEPT 3/3 | **ACCEPT 3/3** | +5.77 | territory clean, escrow positive |
| negotiate, market | −0.0001 ± 0.0002 | ACCEPT 3/3 | **ACCEPT 3/3** | −0.67 | a true null, correctly kept |
| negotiate, oracle | −0.0045 ± 0.0014 | ACCEPT 3/3 | **REFUSE 3/3** | −2.44 | real regression, caught every seed |

The free arbiter is refused on territory *while its escrow test is strongly
positive* (z = +10.07) — which is exactly the intended reading: a real accuracy
gain that the certificate does not cover. The gated arbiter, previously
false-refused, now passes. And the diffuse −0.0048 regression that no
contract-sample gate could see is caught.

**Two honest limits.** The oracle catch fires on all three seeds at
z = −1.77, −2.83 and −2.71 (mean −2.44 against a −1.645 threshold), so the
margin is real rather than marginal — but the spread matters: the weakest seed
clears the bar by only 0.13, and it is the seed whose true regression was
smallest (−0.0048 against −0.0061). A regression appreciably smaller than half a
point would not reliably clear it. And the resolution
depends on how *discordant* a candidate is: `detectable_drop()` assumes a 5%
discordant rate and predicts 0.0074 for this escrow, but the oracle arm moved
only 1.3% of answers, so fewer, quieter pairs gave finer resolution than the
generic estimate. A change that is both diffuse **and** highly discordant would
still need a larger escrow; the formula is in the module so the requirement can
be computed rather than guessed.

**What this does and does not close.** The territory half is closed *exactly*
and for free — it is a structural property, not a statistical one. The
regression half is closed *statistically*, on this benchmark, at this escrow
size, and it degrades gracefully rather than silently: the gate now reports its
own resolution. The general lesson matches the apoptosis arc — **gates that
score contract samples certify contract fidelity, not real behaviour** — and the
remedy in both cases was the same: a small quantity of real held-out data,
used only by gates, never for training.

### The audit: re-running every lifecycle gate decision

A repaired gate is only worth having if it *discriminates*. So the two new
checks were run over every MLP lifecycle arbiter decision — seven benchmarks ×
three evaluation seeds, 21 runs, including the one measured false-accept the
paper reports ([exp_gate_audit.py](exp_gate_audit.py)).

**The territory check refuses all 21.** Uncertified fractions run 1.3%–9.8%,
never zero, because the free arbiter answers with its argmax organ *whether or
not that organ accepted* — so it always answers somewhere no contract speaks.
It refuses the 4 harmful arbiters (wall-robot −0.104/−0.128/−0.111, kmnist
−0.022) and equally the **11 helpful ones** (digits +0.123, +0.058; mnist
+0.023, +0.022, +0.024; satimage +0.030 ×3; kmnist +0.014).

**That makes it a policy, not a detector** — a correction to how this section
first described it. The territory check does not *detect* the wall-robot
regression; it prohibits the entire category of routing that wall-robot belongs
to, and would have prevented that false-accept only by also giving up every
beneficial arbiter in the suite. The paper's original claim therefore stands:
detecting *that regression as such* does need held-out data. What is new is that
a **structural** check would have prevented it for free — at a price paid
elsewhere, in accuracy, not in data.

**The escrow check is the discriminating one, and it is imperfect.**

| escrow (McNemar on the whole validation split) | n | refuses |
|---|---|---|
| harmful arbiters (Δ < −0.01) | 4 | **4** |
| helpful arbiters (Δ > +0.01) | 11 | **1** |
| near-neutral (\|Δ\| ≤ 0.01) | 6 | 4 |

With the escrow enlarged to the whole validation split (B-certified), it now
catches **all four** harmful arbiters — including the wall-robot seed that
previously slipped through at z = −1.37, now z = −6.48 — at the cost of one
false refusal among the eleven helpful ones (kmnist seed 2, Δ = +0.014, flagged
on a per-class test rather than the overall one) and four flags among six
near-neutral runs. **More power bought sensitivity and cost a little
specificity**, exactly the trade to expect. It does not change any decision
here, because territory already refuses every free arbiter; it matters only for
the *gated* arbiter, where the certified lifecycle records zero false refusals.
**The escrow is a useful screen, not a proof.**

**Net effect on the historical record: 11 of 21 decisions change.** Three are
wall-robot, correctly refused. The other eight are beneficial arbiters now
refused on territory — accuracy given up in exchange for contract coverage.
That is exactly the trade the CIFAR-ViT analysis priced (a territory-respecting
arbiter keeps 72% of the gain on digits, 32% on the flagship), and it is the
honest cost of insisting every answer sit inside a contract: **the free arbiter
is not a gateable mechanism, it is an uncertifiable one.** The recommendation
that follows is the same one B-ensemble reached from the other direction — use
the territory-respecting arbiter, or the prototype topology, and do not try to
gate the free arbiter into acceptability.

## B-certified — the lifecycle rerun with the territory-respecting arbiter

*Terminology: "certified" in this section and the comparative table means
**contract-covered** — every answer comes from an organ whose contract accepted
the input. It is a routing property, checked exactly; it is not the per-answer
robust-radius certificate, which accepted answers carry separately.*

The audit concluded that the free arbiter is not a gateable mechanism but an
uncertifiable one, so it was replaced everywhere and the whole lifecycle rerun:
twelve benchmarks × three evaluation seeds, arbiter restricted to organs that
accept, acceptance by the repaired gate ([exp_lifecycle.py](exp_lifecycle.py)
`--arbiter gated --gate repaired`). The historical free-arbiter run is kept for
comparison rather than overwritten.

| benchmark | growth | adolescence | **certified final** | free (uncertified) | Δ vs adol | arbiter accepted |
|---|---|---|---|---|---|---|
| digits | 0.877 | 0.888 | **0.912** | 0.964 | +0.024 | 3/3 |
| mnist | 0.892 | 0.892 | **0.893** | 0.915 | +0.001 | 2/3 |
| kmnist | 0.805 | 0.805 | **0.805** | 0.805 | 0.000 | 0/3 |
| fashion | 0.799 | 0.799 | **0.799** | 0.797 | 0.000 | 0/3 |
| letter | 0.782 | 0.782 | **0.782** | 0.788 | 0.000 | 0/3 |
| satimage | 0.832 | 0.832 | **0.833** | 0.861 | +0.001 | 1/3 |
| wall-robot | 0.800 | 0.800 | **0.800** | 0.685 | 0.000 | 0/3 |
| CIFAR-100 R18 | 0.395 | 0.395 | **0.470** | 0.483 | +0.075 | 3/3 |
| CIFAR-100 ViT | 0.667 | 0.776 | **0.794** | 0.835 | +0.018 | 3/3 |
| sheet sweeps | 0.979 | 0.989 | **0.989** | 1.000 | 0.000 | 0/3 |
| sheet shapes | 0.985 | 0.985 | **0.985** | 1.000 | 0.000 | 0/3 |
| spoken digits † | 0.772 | 0.772 | **0.790** | 0.872 | +0.018 | 3/3 |

**Every answer is inside a contract.** († spoken digits: shared-speaker
split; the speaker-disjoint numbers are in B-speakers and are roughly half.)
The uncertified fraction is **0.0000
across all 36 runs** — by construction, since the gated arbiter can only choose
among organs that accepted. That is the property the whole project is about, and
it now holds for the final configuration rather than for an intermediate one.

**The price is 1–8 points on eight of twelve benchmarks.** Against the
historical free-arbiter lifecycle: spoken digits −0.082, CIFAR-ViT −0.041,
digits −0.036, mnist −0.022, shapes −0.015, cifar-R18 −0.013, satimage −0.009,
kmnist −0.007; fashion, letter and sweeps unchanged. **Wall-robot improves by
+0.114**, because the uncertified run had accepted a catastrophic arbiter there.
Certifying the lifecycle costs real accuracy on most benchmarks and buys it back
only where the uncertified version was badly wrong.

**The headline moves.** On the flagship the certified lifecycle reaches
**0.794** (growth 0.667 → adolescence 0.776 → certified arbiter 0.794), against
0.835 for the free arbiter. Contracted prototypes remain the best fully-covered
configuration at **0.835 strict / 0.844 forced**, so the honest summary of
CIFAR-ViT is unchanged from B-ensemble: *use the prototype topology*; the
cascade-plus-arbiter route is 4 points behind it once every answer must sit
inside a territory.

**The acceptance pattern is the headroom law, enforced.** B-daring measured
that routing headroom (arbiter − cascade) bounds what any router can win;
B-ensemble measured that part of that headroom lives outside every territory.
The gate's verdicts across the twelve benchmarks reproduce both measurements
without being told them: it accepts 3/3 exactly where contract-covered headroom
exists (digits +0.024 captured of +0.076 free; cifar-R18 +0.075 of +0.088;
cifar-ViT +0.018 of +0.059; fsdd +0.018 of +0.100), refuses where there is no
headroom at all (kmnist, fashion, robot — free headroom ≤ 0), and refuses where
headroom exists but is **uncertified** — satimage's +0.029 and the sheets'
+0.011/+0.015 live almost entirely outside territory, so the gated arbiter can
capture ≤5% of them and the gate correctly declines to buy routing churn for
nothing. What looked like three separate findings is one mechanism seen three
ways.

**The gate now accepts no regression at all — after one more fix.** The first
run of this battery left four accepted changes that still regressed, the worst
being wall-robot seed 3 at −0.136 (escrow z = −1.57 against a −1.645
threshold). That looked like a power limit, and it was not: the regression is
**fully present in validation** (z = −6.86 on the complete split), but
`escrow_sample` was capping the escrow at 60 samples per class, which on a
four-class dataset kept 214 of 613 available samples and destroyed the signal.
Validation is held out by construction, so spending less of it than exists buys
nothing; the escrow now defaults to the **whole validation split**.

Rerun end-to-end, the result is clean:

| | 60/class escrow | **whole split** |
|---|---|---|
| arbiters accepted | 19/36 | **15/36** |
| accepted changes that regress | **4** (worst −0.136) | **0** |
| clearly-helpful arbiters refused | 0 | **0** |
| uncertified answers | 0.0000 | **0.0000** |

Exactly the four regressive changes were dropped and no beneficial one was lost
with them — wall-robot is now refused on all three seeds (z = −5.94, −6.86,
−6.67) and falls back to adolescence at 0.800. **Every accepted change in the
certified lifecycle is non-regressive on real test data, and every answer sits
inside a contract.**

**Disclosure: the gate was iterated on the evaluation seeds.** The organism,
its thresholds and the lifecycle were frozen before seeds 2–4 were first run —
but the *gate* was not: both escrow redesigns (the McNemar rewrite, then the
full-split escrow) were made in direct response to eval-seed failures. The
"0 regressions in 36 runs" above is therefore in-sample **for the gate design**,
and by this project's own seed discipline it requires out-of-sample
confirmation. The gate specification is now frozen (territory check; whole-split
validation escrow; one-sided McNemar at z ≥ −1.645; Bonferroni per-class flags;
0.25 hard floor) and a confirmation battery on **seeds 5–7, never used for any
decision**, is reported below. A regression small enough to hide inside
validation noise would still pass — `detectable_drop()` reports that threshold
for a given escrow — and CIFAR confirmation seeds await the GPU box, since they
require growing fresh 100-class organisms.

**Escrow independence, tested.** The disclosure above noted that the escrow
reuses the validation split that the lifecycle's own gates (adolescence's
last-task check, the original rule) also see. The ablation splits validation
into class-stratified halves — half A for the lifecycle's gates, half B for the
escrow — and reruns the ten CPU benchmarks on evaluation seeds
(`--escrow-split`). **Two of thirty verdicts change, and both change toward
*more* acceptance**: letter seed 4 (refused → accepted; a −0.015 regression now
admitted) and satimage seed 3 (refused → accepted; a +0.005 gain). No verdict
flips from accept to refuse, which is what reuse-inflated acceptance would have
produced. The reuse is therefore demonstrably benign, and the price of
independence is the expected one — half the escrow is ~√2 less power, and one
sub-resolution regression slips through that the full-split gate had caught.
Final accuracies are identical on eight benchmarks, −0.005 on letter and
+0.002 on satimage. The frozen gate keeps the whole split.

**Confirmation on untouched seeds 5–7 (gate frozen first).** Ten CPU
benchmarks × three fresh seeds, 30 runs, nothing adjusted afterwards:

| confirmation, seeds 5–7 | result |
|---|---|
| uncertified answers | **0.0000** in all 30 runs |
| arbiters accepted | 9/30 |
| clearly-helpful candidates refused | **0** |
| wall-robot refused | **3/3** (z = −3.35, −6.34, −7.72) |
| accepted changes that regress | **1**: satimage seed 6, **−0.0068** at escrow z = +0.38 |

The one slip is **half the gate's own stated resolution** for that escrow
(`detectable_drop` = 0.0137 at satimage's n = 723): a validation-vs-test gap
small enough that no test on this data could have seen it. So the out-of-sample
form of the guarantee is the precise one, not the absolute one: **the frozen
gate admits no regression above its stated resolution, refuses every harmful
change it can resolve, and refuses no clearly-helpful one** — and the in-sample
"0 regressions" from seeds 2–4 should be read with that qualifier. The headroom
pattern also reproduces on fresh seeds (digits and fsdd chased 3/3 at free
headroom +0.068/+0.079; kmnist, fashion, letter, robot and both sheets skipped),
so the gate's acceptance logic, not just its refusals, transfers. CIFAR
confirmation seeds still await the GPU box.

## B-meta — second-order contracts: an SLA on the organism's own drift

The intermediate guarantee forecasts drift; a second-order contract *promises*
it ([exp_meta.py](exp_meta.py)): before each prototype-entry event the system
publishes a bound — its contract-sampled forecast times a factor calibrated
**online, conformally, from its own past events** (α = 0.25, prior 2.0 while
history is short; all preregistered). Measured on real test flips over 54
events (two backbones × three eval seeds × nine events): **5 violations
(9.3%) against the 25% budget, at median tightness bound/real ≈ 1.1–1.45** —
promises that hold and are not vacuous. The online calibration beats the fixed
prior decisively (5 vs 14 violations; the fixed factor breaks on ViT where
drift ratios run higher). The organism can sell an SLA on its own change —
governance extended one level up, audited like everything beneath it.

## B-fractal — an organ that is an organism

The contract algebra composes upward ([exp_fractal.py](exp_fractal.py)): a
**SuperOrgan** wraps a whole sub-organism behind one merged territory (a union
of PCA territories *is* a PCA territory), presents the standard organ
interface, and — because the merged territory is first-class — accepts
top-level vetoes and tightening exactly like a leaf. CIFAR-100/ViT, ten tasks
arriving in three super-epochs ([4, 3, 3]), each epoch grown as a normal
sub-organism, consolidated, wrapped, and cascaded at the top (3 eval seeds):

strict **0.718 ± 0.003** (above the flat cascade's 0.667; below the arbitered
flat lifecycle's 0.835 — three consolidations, no global arbiter), coverage
0.98, **top-level growth flips exactly 0** (the invariant survives a level of
nesting), and the acid test: **an entire ~30-class super-organism rebuilt from
its merged contract alone agrees at 0.75–0.78** on its region — the two-level
version of replacement, weaker than the leaf version (0.92–0.96) exactly as
scale predicts. Two levels, one algebra; heterogeneous primitives inside
super-organs are now a plumbing exercise rather than a design question.

## B-isolate — the feature ledger, and the final configuration

Every remaining mechanism isolated through the FULL pipeline (growth →
gated adolescence → gated arbiter) on both flagships, eval seeds 2–4
([exp_isolate.py](exp_isolate.py); removal rule preregistered):

| feature (removed in isolation) | split-MNIST final | CIFAR-ViT final | verdict |
|---|---|---|---|
| **full configuration** | **0.915** | **0.835** | reference |
| − vetoes | 0.726 (−0.19) | 0.339 (−0.50) | **most load-bearing**: the arbiter does *not* subsume inhibition; without vetoes growth itself drowns (0.10–0.13) |
| − spectral cap | 0.892 (−0.02) | 0.761 (−0.07) | load-bearing; uncapped evidence scales also break the arbiter gate (0/6 accepted) |
| − adolescence | 0.915 (±0) | 0.743 (−0.09) | load-bearing where its gate accepts: consolidation fixes representation, the arbiter fixes routing — not redundant |

Previously measured and already removed: scalar tightening (−0.08 when added
back), trunk-mode consolidation (collapses at 100 classes), mid-life
consolidation timing (−0.05 vs end-of-epoch), territory renegotiation from
generated data (vacuous), **self**-priced decentralized routing (−0.2 to
−0.65 — superseded: shared-pool pricing recovers it and ties the arbiter after
consolidation, see B-daring).
**Nothing new is removed: the final configuration is the current pipeline.**

## B-order — class-order robustness of the certified lifecycle

Every class-incremental result above uses the default task order (ascending
class index). The certified lifecycle was rerun with two random permutations of
the class order (`--class-order-seed 11` and `22`) on the six MLP benchmarks,
evaluation seeds 2–4.

| certified final, eval seeds 2–4 | default | order 11 | order 22 | spread |
|---|---|---|---|---|
| digits | 0.912 | 0.918 | 0.927 | 0.014 |
| mnist-scale kmnist | 0.805 | 0.813 | 0.809 | 0.008 |
| fashion | 0.799 | 0.757 | 0.774 | **0.042** |
| letter (13 tasks) | 0.782 | 0.783 | 0.786 | 0.004 |
| satimage | 0.833 | 0.809 | 0.836 | 0.027 |
| wall-robot | 0.800 | 0.788 | 0.752 | **0.048** |

**Order moves the certified number by 0.4–4.8 points**, most on wall-robot and
fashion (few tasks, so one bad early organ sets the cascade's ceiling) and
least on letter (thirteen tasks average it out). The default order is not
systematically favourable — it is best on fashion and robot, worst on digits and
letter — so the headline numbers are not order-cherry-picked, but any single
figure should be read with a ±2–4 point order band on the small-task benchmarks.
The certification properties are unaffected: uncertified fraction 0.0000 in
all 36 permuted runs.

**The gate's record, updated with these 36 runs.** Ten arbiters accepted; two
accepted regressions. satimage seed 4 (−0.004) sits below its escrow's stated
resolution (0.0137) and is the kind of slip the guarantee already allows.
**letter seed 4 under order 11 (−0.023 at z = −1.10) is above letter's stated
resolution and is therefore a genuine miss** — the validation split did not
show a regression that test does. Across every run of the frozen gate now on
record (36 evaluation + 30 confirmation + 36 permuted = 102), that is one miss
above resolution and two sub-resolution slips, against 4/4 known-harmful
changes refused and 0 clearly-helpful ones refused. The guarantee statement
stands as written — "admits no regression above its stated resolution" —
with a measured exception rate of 1 in 102 that the paper now quotes.

## B-ledger — a lifetime budget on accepted drift

The gate's honest residual is that regressions below its stated resolution
pass. Over a long life of many gated events such slips can *compound*, each
individually invisible — and the sleeping organism (B-daring), with a gated
consolidation and a gated dream every night, is the setting where that matters.
`cgrow/gates.py` now carries a **DriftLedger**: every accepted change records
its signed escrow delta (c−b)/n, and a candidate that passes its own McNemar
test is still refused if accepting it would push the *cumulative* delta below
−budget (preregistered 0.02). This is the second-order SLA of B-meta applied to
the escrow. Run on the seven MLP benchmarks × evaluation seeds 2–4, with the
dream arbiter territory-respecting, in two arms: the frozen escrow check alone,
and escrow plus ledger ([exp_sleep.py](exp_sleep.py) `--gated --escrow` /
`--budget 0.02`).

| sleep, eval seeds 2–4 | nights per life | nights refused by escrow | refused by **ledger** | cumulative accepted drift | final (both arms) |
|---|---|---|---|---|---|
| digits | 5 | 0 | 0 | +0.018 | 0.922 |
| mnist | 5 | 2 | 0 | +0.000 | 0.899 |
| kmnist | 5 | 7 | 0 | +0.000 | 0.805 |
| fashion | 5 | 4 | 0 | +0.000 | 0.799 |
| satimage | 3 | 0 | 0 | −0.002 | 0.836 |
| wall-robot | 2 | 3 | 0 | 0.000 | 0.800 |
| **letter** | **13** | 9 | **1** | **−0.012** (min −0.013) | 0.782 |

Two findings. **The escrow check does real work in the sleep setting**: 25
nightly candidates that the original synthetic-plus-validation gate would have
judged on contract samples alone were refused on real held-out data — kmnist 7,
letter 9, fashion 4, robot 3, mnist 2. **And the ledger binds exactly where the
mechanism predicts and nowhere else**: on letter, the only long life (13 nights),
accepted sub-resolution drifts accumulated to −0.013, and one dream that passed
its own per-night test was refused because accepting it would have breached the
budget. On every five-night life cumulative drift stayed ≈0 or positive and the
budget never came close. The one refusal changed no final accuracy (0.782 in
both arms), so on these lives the ledger is free insurance; its value is on long
deployments, where the accumulation it guards against is the only way a
correctly-gated system can still drift.

## B-vote — certified voting is a null on grown organisms

B-ensemble found one genuinely contract-covered ensembling gain: majority vote
among the organs that *accept* an input beat first-accept by +0.010 on the
20-organ twin pool. Put to work on the real certified organisms
([exp_vote.py](exp_vote.py); ten CPU benchmarks × evaluation seeds 2–4, both
routers contract-covered by construction), it changes **nothing**:

| certified organism, eval seeds 2–4 | organs | contest rate (≥2 accept) | answers changed by voting |
|---|---|---|---|
| digits / mnist / kmnist / fashion | 5 | 0.42 / 0.58 / 0.65 / 0.70 | **0** |
| letter (13 tasks) | 13 | 0.78 | **0** |
| satimage / wall-robot | 3 / 2 | 0.52 / 0.53 | **0** |
| sheet sweeps / shapes / spoken digits | 5 / 4 / 5 | 0.84 / 0.00 / 0.60 | **0** |

Zero changed answers in 30 runs — not a small effect, none. The mechanism is
sharper than tie-breaking (with two accepters a disagreement is a tie and falls
back to first-accept): even on letter, with thirteen organs and contests on 78%
of inputs, **the organs that co-accept an input agree on its label**. Vetoes
trim the false accepts, so the organs left accepting are the consolidated
generalist and the true specialist, and those concur. The twin pool's +0.010
came from *duplicated* competence — two independently grown organs disagreeing
on the same territory — which the grown anatomy never contains (B-ensemble's
death-certificate result said the same thing from the other side: no organ in a
grown anatomy is redundant). Voting is a free, certified router upgrade with
nothing to route. It stays in the ledger as a measured null, and the twin-pool
figure is now qualified everywhere it appears.

## B-speakers — the spoken-digits result does not survive a speaker-disjoint split

Every FSDD number in this report so far comes from a **random** split of the
3,000 recordings, which places the same six speakers in train, validation and
test. That is the standard way the dataset is used, and it is also a leak: a
model can score by recognising the speaker rather than the digit. The honest
split holds out one speaker for test and a second for validation, training on
the remaining four (`exp_sheet.py` stream `fsdd_spk`; evaluation seeds 2–4
rotate which speakers are held out).

| spoken digits (sheet), eval seeds 2–4 | shared-speaker split | **speaker-disjoint** |
|---|---|---|
| organism, growth | 0.772 | **0.349** (0.244 / 0.350 / 0.452) |
| organism, certified lifecycle | 0.790 | **0.328** |
| unconstrained arbiter | 0.872 | 0.391 |
| replay (strengthened) | 0.93 | **0.451** |
| joint (upper bound) | 0.95 | **0.518** |
| fine-tune / EWC | 0.20 / — | 0.16 / 0.17 |

**Everything collapses, so the collapse is the benchmark's, not the
organism's.** Even the joint model that sees all data at once loses 43 points
on a new speaker; replay loses 48; the organism loses 42. Relative standing is
preserved — ours trails replay by 0.10 here against 0.16 under the shared
split — so the earlier comparison was not *distorted* by the leak, but its
absolute level was inflated by roughly a factor of two, and the paper's
spoken-digits row now carries this caveat. At this feature scale (an 8×16
log-spectrogram, four training speakers) the dataset does not support
cross-speaker generalisation for any method tested.

**A second, sharper lesson about the gate.** On seed 3 the certified arbiter
was *accepted* at 0.288 against an adolescence cascade of 0.350 — a −0.062
regression on test — with a strongly *positive* escrow verdict (z = +4.91). The
escrow was not wrong about what it measured: validation is one held-out
speaker, test is another, and the arbiter genuinely improved on the validation
speaker. **The gate's guarantee assumes validation and test are exchangeable**,
and a new-speaker deployment breaks that assumption by construction. This is
the correct reading of every acceptance-gate result in this report: the gate
certifies non-regression *on the distribution the escrow samples*; it cannot
certify generalisation to a distribution it has never seen, and no held-out
escrow can. Where deployment data will differ from validation in a known way
(new speakers, new sensors, new sites), the escrow must be drawn from that
axis — held out by speaker, not by row — or the gate's verdict is about the
wrong population.

## B-backbones — the RanPAC question, answered on the GPU box (session 31)

*Four feature caches were extracted on the RTX 3070 (`gpu/setup_and_extract.ps1`,
`gpu/extract_gpu.py`) and copied into `results/`: ViT-B/16-in21k on the FULL
train set (500/class; the reference cache `cifar100_vitb.npz` has 250/class),
DINOv2-B (768-d), ViT-L/16-in21k (1024-d), and CIFAR-10 on ViT-B. Everything
below is CPU work on those caches: evaluation seeds 2–4, the frozen repaired
gate, the territory-respecting arbiter, no changes to any hyper-parameter.
Families `B_cifar100_{vitb_full,dinov2b,vitl}`, `B_cifar_adol_*_adol_protos`,
`B_lifecycle_backbones`. Runbook: `gpu/README.md`.*

**The question.** The certified CIFAR-ViT number (contracted prototypes
0.835 strict / 0.844 forced) sits 8.7 points under RanPAC's 0.922 on the same
ViT-B/16-in21k backbone. Is that gap features (a frozen backbone the contract
machinery cannot exploit) or method (RanPAC's trained random-projection head
buying what nearest-class-mean topologies cannot)?

| backbone (train/class) | growth | adolescence | **certified lifecycle** (arbiter accepted) | free arbiter | **contracted prototypes** strict / forced | NCM same features | same-data replay | joint |
|---|---|---|---|---|---|---|---|---|
| ViT-B/16-in21k (250) — reference | 0.667 | 0.776 | **0.794** (3/3) | 0.835 | **0.835 / 0.843** | 0.843 | 0.862 | 0.807 |
| ViT-B/16-in21k (500) | 0.704 | 0.825 | **0.825** (0/3) | 0.836 | **0.832 / 0.842** | 0.842 | 0.859 | 0.849 |
| DINOv2-B (500) | **0.816** | **0.865** | **0.865** (0/3) | 0.873 | **0.865 / 0.873** | 0.873 | 0.889 | 0.891 |
| ViT-L/16-in21k (500) | 0.727 | 0.860 | **0.861** (1/3) | 0.869 | **0.867 / 0.877** | 0.877 | 0.892 | 0.877 |

Growth flips: 0 on every run. Uncertified answer fraction: 0.0000 on all 9 new
lifecycle runs. All 9 adolescence consolidations accepted (C answers 96–97%).

**Answer: the gap is method.** On every backbone the contracted prototypes'
*forced* accuracy equals nearest-class-mean on the same features to the third
decimal (0.843/0.842/0.873/0.877), and strict trails forced by exactly the
0.8–1.0 point of abstention the contract buys. The certificate stack rides
feature quality one-for-one — DINOv2 lifts the certified number by +3.0 —
but it never rises *above* the prototype topology it wraps. RanPAC's extra
8 points on ViT-B come from a trained random-projection + ridge head on top of
the same features, i.e. from a classifier family the current contract topology
does not contain. The lever is therefore a contracted head, not a better
backbone.

**Three further readings.**

1. *Data was not the limit for the prototypes but it was for the lifecycle.*
   Doubling the ViT-B train set leaves contracted prototypes unchanged
   (0.835 → 0.832) while growth gains +3.7 and adolescence +4.9; the certified
   lifecycle moves 0.794 → 0.825 without the arbiter, because the consolidated
   organ absorbs the extra data and the arbiter then has nothing left to route.
2. *On strong features the cascade route catches the prototype route.* At
   ViT-B the certified lifecycle was 4 points under contracted prototypes
   (0.794 vs 0.835), which is why the paper's CIFAR-ViT recommendation was the
   prototype topology. On DINOv2-B the two coincide (0.865 vs 0.865) and on
   ViT-L they are 0.6 apart (0.861 vs 0.867). The recommendation is
   substrate-dependent, and the routing-headroom law says why: free arbiter
   minus cascade shrinks from +0.059 (ViT-B) to +0.008 (DINOv2-B).
3. *The arbiter is refused 8 times in 9, every time by the same clause, and
   every refusal is a foregone sub-resolution gain, not a saved harm.* Territory
   and escrow pass on all 9 runs (McNemar z from −1.04 to +3.91). What fails is
   the original contract rule's `val_new ≥ val_ref` requirement: the arbiter
   scores 0.4–2.6 points lower than the cascade on the last task's *real*
   validation split, while beating it on the synthetic per-class mean by
   1.6–2.4 points. The refused candidates would have added +0.1 to +0.7 on test.
   The one acceptance (ViT-L seed 3, the only run where last-task validation
   did not drop) added +0.001. The gate's record therefore stays at 0 accepted
   regressions, now over 45 evaluation runs, and the routing-headroom
   prediction holds: where the free arbiter's edge is under a point, the gated
   arbiter is not worth deploying.

**What does not change.** Same-data strengthened replay still leads the
certified organism on strong features (0.889 vs 0.865 on DINOv2-B, 0.892 vs
0.861 on ViT-L), by the same 2.5–3 points as on ViT-B. Joint training on the
frozen features sits at replay's level (0.891 DINOv2-B), so on these
substrates replay is effectively the ceiling and the certified organism is
2.5–3 points under it with nothing stored.

**Not compared.** No verified published class-incremental CIFAR-100 number
exists in our citation audit for DINOv2-B or ViT-L/16-in21k backbones, so those
rows are anchored to same-feature NCM, same-data replay and joint only. RanPAC's
0.922 is a ViT-B/16-in21k number and is compared on that row alone.

## B-ptm — the pretrained-model class-IL suite (GPU box, session 32)

*Six datasets on which SimpleCIL / APER (Zhou et al. 2023) and RanPAC
(McDonnell et al. 2023) publish ViT-B/16-in21k numbers in the B0 protocols we
use: ImageNet-R, ImageNet-A, CUB-200, ObjectNet-200 (inc 20), OmniBenchmark
(inc 30), VTAB (inc 10). Features extracted on the RTX 3070
(`gpu/extract_suite.ps1`; every cache matches the distributed split's class
and image counts), pipeline run unchanged on the GPU box's CPU by a separate
Claude session following `docs/HANDOFF_GPU.md`, evaluation seeds 2–4, frozen gate.
Anchors from Zhou v1 supplementary Tables 14/22/30/38/45/50 (IN21K block,
last-step) and RanPAC v3 Table 1, read from the PDFs by the GPU-box session and
spot-checked here on Tables 6, 7, 22, 30 and RanPAC Table 1 (`ANCHORS_PTM.md`).
Per-seed detail and the GPU-box write-up: `docs/PTM_RESULTS.md`; families
`B_ptm[<name>_vitb]`.*

| dataset (classes × inc) | growth | adolescence (acc./3; after) | **certified lifecycle** (arbiter acc./3) | free arbiter | **contracted protos** strict / forced | NCM | replay | joint | SimpleCIL / APER-adapter / RanPAC last |
|---|---|---|---|---|---|---|---|---|---|
| ImageNet-R (200 × 20) | 0.280 | 3/3; 0.431 | **0.431** (0/3) | 0.492 | **0.524 / 0.551** | 0.551 | 0.575 | 0.435 | 0.546 / 0.680 / 0.779 |
| ImageNet-A (200 × 20) | 0.109 | 3/3; 0.450 | **0.450** (0/3) | 0.450 | **0.454 / 0.511** | 0.511 | 0.365 | 0.105 | 0.489 / 0.494 / 0.624 |
| CUB-200 (200 × 20) | 0.513 | 3/3; 0.822 | **0.822** (0/3) | 0.850 | **0.858 / 0.866** | 0.866 | 0.861 | 0.717 | 0.867 / 0.871 / 0.903 |
| ObjectNet-200 (200 × 20) | 0.255 | 3/3; 0.400 | **0.400** (0/3) | 0.484 | **0.506 / 0.543** | 0.543 | 0.517 | 0.388 | 0.536 / 0.580 / — |
| OmniBenchmark (300 × 30) | 0.615 | 2/3; 0.690 | **0.669** (0/3) | 0.690 | **0.724 / 0.733** | 0.733 | 0.757 | 0.674 | 0.732 / 0.744 / 0.799 |
| VTAB (50 × 10) | 0.523 | 3/3; 0.886 | **0.886** (0/3) | 0.886 | **0.888 / 0.893** | 0.893 | 0.918 | 0.609 | 0.844 / 0.844 / 0.922 |

Growth flips: 0 on 18/18 seeds. Uncertified answers: 0.0000 on 18/18 lifecycle
runs. Adolescence accepted 17/18 (OmniBenchmark seed 4 refused: consolidated
mean regressed). Arbiter refused 18/18; on 11 the escrow McNemar test failed
(z −1.0 to −3.5: measurably worse than the organism on real held-out data), on
the rest it never cleared the +0.01 synthetic-mean requirement. Zero accepted
regressions.

**Three readings.**

1. *The CIFAR finding holds on every dataset.* Contracted prototypes' forced
   accuracy equals nearest-class-mean on the same features to the third decimal
   on all six datasets and all 18 seeds, and lands within about one point of
   published SimpleCIL-last on five of them (ImageNet-R 0.551 vs 0.546,
   ImageNet-A 0.511 vs 0.489, CUB 0.866 vs 0.867, ObjectNet 0.543 vs 0.536,
   OmniBenchmark 0.733 vs 0.732); on VTAB ours is 0.893 against Zhou's 0.844
   but equal to RanPAC's own "NCM only" ablation (0.884). RanPAC's
   random-projection head then buys a further 4 to 23 points everywhere it
   reports. The strict number trails forced by the abstention the contract
   buys: 0.5 points on VTAB, 0.8 CUB, 0.9 OmniBenchmark, 2.7 ImageNet-R,
   3.7 ObjectNet, 5.7 ImageNet-A.
2. *The certified lifecycle is only as good as the consolidated organ's reach.*
   With the arbiter refused everywhere, certified = post-adolescence. That sits
   0.2–0.4 points under strict prototypes where C answers 91–100 % of the test
   set (VTAB, ImageNet-A, CUB 3.6), and 5.5–10.6 points under where C answers
   only 43–75 % (OmniBenchmark, ImageNet-R, ObjectNet) and the cascade of weak
   grown organs (growth strict 0.11–0.62) answers the rest. This is the CIFAR-ViT
   lesson again, now with its mechanism exposed: the prototype topology is the
   certified recommendation, and the cascade route only catches it when the
   consolidated organ's territory covers the input.
3. *The gate refused 18 arbiter proposals and was right to.* On 11 seeds the
   escrow test found the candidate worse than the organism on real validation
   data; on the remaining 7 the candidate matched the organism (z ≈ 0, ImageNet-A
   and VTAB, where C already answers everything, so there is nothing to route).
   The one seed where the arbiter cleared the original rule (OmniBenchmark
   seed 4, z +7.13 overall) was stopped by the escrow's per-class flags. Routing
   headroom (free − cascade) was 0.0–0.08 and predicted every verdict.

**What did not survive contact with the suite: three latent bugs, fixed.** The
pipeline had only ever seen organs whose class territories all had the full 24
PCA components. On these datasets an organ grown on residual samples routinely
has classes with fewer than 25 samples and therefore narrower territories, and
`Organ._proj_nearest` stacked per-territory coordinates of unequal width and
crashed (every chain, same line). Two smaller issues surfaced alongside on
ImageNet-A (classes with 2 training images): a NaN anchor for a class no
territory covers, and NaN-poisoned gate aggregates that would have *silently
accepted* a consolidation and *refused an arbiter by arithmetic*. The GPU-box
session fixed all three with the user's authorisation (`docs/GPU_BOX_FIXES.md`); its
patch (`gpu/patches/gpu_box_fixes_2026-09-07.patch`, which shipped with
pristine copies of the three files) is applied verbatim here. Before it arrived the fixes were re-implemented from
the description and found equivalent: CIFAR predictions are bitwise identical
before and after under both versions, and the VTAB adolescence and lifecycle
steps reproduce the GPU-box files to 16 digits. No CIFAR or MLP result is affected (their
territories were all full-width, and every class had pool samples).

**RanPAC, decomposed.** `docs/RANPAC_VS_US.md` sets the two methods side by side.
RanPAC's own ablation rows place our forced prototypes at its "NCM only" level
on every shared-backbone dataset; its random-projection ridge head then buys
+3.6 to +10.6 on frozen features and first-session PETL a further +0 to +6.1.
Its NCM-only row also reveals the checkpoint behind each column: on ImageNet-R,
ImageNet-A and CUB it matches Zhou's **IN1K** SimpleCIL block (61.28 / 49.44 /
85.16), not the IN21K block we use, so those three RanPAC anchors are on the
ImageNet-1K-fine-tuned backbone (worth about +6.7 on ImageNet-R at the NCM
level); CIFAR, OmniBenchmark and VTAB are consistent with IN21K.

**Anchor correction.** The CIFAR-100 APER-adapter value used since 2026-08-31
(0.852 last / 0.907 avg) was Zhou's Table 1 column, which is B0 **Inc5**. The
B0 Inc10 value (v1 Table 7, IN21K block, verified on the PDF here) is 0.875 /
0.922, so our contracted-prototype comparison on CIFAR moves from −1.7 to −4.0
against APER-adapter. SimpleCIL's last-step value (0.813) is split-invariant
and unchanged, so the +2.2 headline stands.

## Preregistration 2026-09-14 — B-horizon, B-ood, B-cost (CPU) and battery 4 (GPU box)

*Written before any of these runs was read. Configurations, thresholds and the
reading rules below are frozen; anything changed afterwards is recorded as a
change, with the reason, in the results section that replaces this stub.*

**B-horizon (`--inc` on the existing caches; CPU here).** CIFAR-100 ViT-B at
inc 5 / 2 / 1 (20 / 50 / 100 steps) and ImageNet-R at inc 10 / 4 / 2 (20 / 50 /
100 steps), evaluation seeds 2–4, same class order per seed as the inc-10
runs. Configurations: contracted head (`exp_head.py --inc`, M = 2,000 and
10,000), grown cascade + gated adolescence + certified lifecycle
(`exp_cifar.py --adolescence --inc`, `exp_lifecycle.py --inc`; organisms cached
as `adol_org_<tag>_<seed>_inc<k>.pkl`), contracted prototypes (verified
horizon-invariant numerically, `results/horizon_protos_check.json`). One
protocol change, made for memory and applied identically at every horizon:
the synthetic pools are bounded by a TOTAL budget equal to the inc-10 values
(consolidation 40,000 = 10 × 4,000; arbiter 25,000 = 10 × 2,500), so every
≤ 10-organ run is bit-for-bit unchanged and a 100-organ organism samples 400
per organ. Preregistered readings: (i) growth flips must be 0 at every step at
every horizon — a single non-zero is a broken invariant, reported as such;
(ii) the head's `answer_flips_under_growth` is expected to rise with the
number of re-solves — reported per 100 test points against the horizon;
(iii) certified-lifecycle strict at each horizon against the inc-10 value
(0.794), the gate's accept/refuse counts, uncertified fraction (must be 0.0);
(iv) no published method reports beyond Inc5; SimpleCIL / APER Inc5 anchors
(CIFAR: 81.26 / 85.15 last, Zhou v1 Table 1, already verified 2026-08-31) are
the only comparable published cells.

**B-ood (`exp_ood_ptm.py`; CPU here).** ID = CIFAR-100 ViT-B test split; OOD =
CIFAR-10 test (`cifar10_vitb.npz`, near) now, SVHN / DTD (far) when the box
returns `ood_*_vitb.npz`. Score = the certificate slack s(x) = max_i
q_i(x)/θ_i over the accept conjunction (margin as τ/margin), so s ≤ 1 ⇔ accept
up to vetoes; cascade = min over non-vetoing organs. Configurations: grown
cascade (`adol_org_vitb_<seed>.pkl`), cascade + consolidated C (the escrow
gate re-run; if refused, the arm equals the cascade and says so), contracted
prototypes, contracted head M = 2,000. Baselines on the same features and
split: NCM-cosine, kNN-1 (normalised), Mahalanobis (shared covariance), MSP of
a joint logistic probe. Metrics: AUROC (OOD positive), FPR@95 TPR, the
contract's own operating point (ID coverage vs OOD accept rate); selective
prediction on ID with s as selector: AURC, risk at the contract's coverage,
risk at 80 % / 90 %. Nothing is tuned on OOD data. Hypothesis: slack ≥
NCM-cosine on AUROC; kNN / Mahalanobis may be stronger and are reported either
way. Published OOD anchors are fine-tuned-encoder numbers and will be named as
such, not compared numerically, unless a frozen-encoder anchor is found in a
source PDF.

**B-cost (`exp_cost.py`; CPU here).** Per configuration on CIFAR-100 ViT-B,
DINOv2-B and VTAB: state floats (to answer), learning-state floats (to keep
learning: the head's Gram), inference MACs per sample after the encoder, and
wall-clock to absorb one task (protos, head; the cascade timed by replaying
growth on seed 2 only). RanPAC's published M = 10,000 head by arithmetic, not
run; a 20/class replay buffer as raw pixels and as features. The encoder is
stated once (shared). Timing is a scale indication, not a tuned quantity.

**Battery 4 (GPU box, `docs/HANDOFF_GPU3.md`).** (a) First-session-adapter caches
for confirmation seeds 5–7 on all seven datasets, then the standard chain, so
the headline table gets its own confirmation block; reading rule as for the
MLP confirmation: accepted regressions and helpful refusals counted, nothing
re-tuned. (b) Far-OOD caches (SVHN, DTD; CIFAR-10 on DINOv2-B). (c) Text
substrate: Banking77 (77 classes, inc 7) and CLINC150 (150 in-scope intents,
inc 15; the out-of-scope queries become an OOD cache) on a frozen
all-mpnet-base-v2 sentence encoder, mean-pooled, never tuned; the standard
chain unchanged. (d) Tabular substrate: an intrusion-detection stream
(NSL-KDD if downloadable, else CIC-IDS2017 or UNSW-NB15), fixed preprocessing
in `gpu/extract_tabular.py` (one-hot categoricals, log1p numerics, classes
with ≥ 300 train / ≥ 30 test rows, 5,000-row cap, inc 2), the standard chain
unchanged with the 768-d CIFAR configuration — a first pass, not a tuned one;
if the organ configuration is visibly wrong for ~100-d inputs that is reported,
not fixed on the evaluation seeds. (e) Optional: the ImageNet-1K checkpoint
with adapters on all six RanPAC datasets. For (c) and (d) no published anchor
is quoted until read from a source PDF; the comparison is against the
same-data baselines the chain produces (NCM, replay, joint).

## B-horizon — the same stream cut into 20, 50 and 100 updates (session 41)

*Preregistered above (2026-09-14). Every published exemplar-free result on this
protocol uses 5 or 10 steps; a deployed system takes an update whenever a
class appears. `--inc` on the existing caches, same class order per seed, so
each horizon is the same sequence of classes arriving in smaller batches.
CIFAR-100 ViT-B at 10 / 20 / 50 / 100 steps and ImageNet-R at 10 / 20 / 50 /
100 steps, evaluation seeds 2–4. Family `B_horizon`.*

| CIFAR-100, ViT-B/16 | inc10 (10 steps) | inc5 (20 steps) | inc2 (50 steps) | inc1 (100 steps) |
|---|---|---|---|---|
| head 2000: answers changed per 100 test points over the run | **13.0** | **18.7** | **31.0** | **40.6** |
| head 2000: final strict | 0.869 | 0.869 | 0.869 | 0.869 |
| head 10000: answers changed per 100 test points over the run | **14.5** | **20.0** | **28.1** | **36.9** |
| head 10000: final strict | 0.875 | 0.875 | 0.875 | 0.875 |
| contracted prototypes: strict (seeds 2/3/4) | 0.8333/0.8366/0.8358 | 0.8333/0.8366/0.8358 | 0.8333/0.8366/0.8358 | 0.8333/0.8366/0.8358 |
| contracted prototypes: answers changed | **0** | **0** | **0** | **0** |
| cascade growth strict | — | 0.630 | 0.592 | 0.564 |
| consolidation accepted (of 3) → strict | — | 2 → 0.717 | 3 → 0.747 | 3 → 0.729 |
| arbiter accepted (of 3) | — | 0 | 0 | 0 |
| **certified lifecycle strict** | — | **0.717** | **0.747** | **0.729** |
| free arbiter (uncertified, for reference) | — | 0.776 | 0.819 | 0.829 |
| growth flips / uncertified fraction | — | 0 / 0.0000 | 0 / 0.0000 | 0 / 0.0000 |

| ImageNet-R, ViT-B/16 | inc20 (10 steps) | inc10 (20 steps) | inc4 (50 steps) | inc2 (100 steps) |
|---|---|---|---|---|
| head 2000: answers changed per 100 test points over the run | **40.3** | **59.9** | **95.6** | **140.9** |
| head 2000: final strict | 0.633 | 0.633 | 0.633 | 0.633 |
| head 10000: answers changed per 100 test points over the run | **66.4** | **97.8** | **148.3** | **201.5** |
| head 10000: final strict | 0.652 | 0.652 | 0.652 | 0.652 |
| contracted prototypes: strict (seeds 2/3/4) | 0.5177/0.5338/0.5208 | 0.5177/0.5338/0.5208 | 0.5177/0.5338/0.5208 | 0.5177/0.5338/0.5208 |
| contracted prototypes: answers changed | **0** | **0** | **0** | **0** |

**The re-solved head: same destination, more churn.** Its final accuracy does
not depend on the horizon at all (CIFAR strict 0.869 at
every inc for M = 2,000, 0.875 for M = 10,000;
ImageNet-R 0.633 / 0.652):
the last ridge solve sees the same Gram and the same class projections
whatever the split, so the last answers are the same answers. What changes is
everything in between. Cumulative answer changes on the test set rise
monotonically with the number of re-solves — CIFAR 13 → 19 → 31 → 41
per 100 test points from 10 to 100 steps (M = 2,000; 15 → 37 at M = 10,000),
ImageNet-R 40 → 60 → 96 → 141 (66 → 202) —
while the churn *per update* falls (1.30 → 0.41 changed answers per
100 test points per step on CIFAR), because each later update moves a smaller
share of the total. A deployment that updates a hundred times has rewritten
41 % of its test-set answers along the way, on a benchmark where it
ends at the same accuracy as one that updated ten times. On ImageNet-R the
count exceeds the test-set size: on average every answer has changed more
than once.

**The prototypes: the same numbers to twelve decimals.** Strict, forced and
coverage are identical across every horizon on every seed of both datasets
(`horizon_protos_check.json`; `protos_horizon_invariant_all` = True). This is
by construction — each class's anchor and territory are fitted from that
class's rows alone and the margin is solved once on the whole validation
split — but it is verified rather than asserted, and it is the property the
overview (`overview/`) leads with: the stream's granularity is not a
hyper-parameter of the stable configuration.

**The cascade: the invariant holds at 100 organs; the accuracy pays for the
organs.** Growth flips are 0 over 501 checked
steps of 9 organisms, and the uncertified fraction is 0.0000 in every run. One
accounting note, stated because it matters: the `--adolescence` path that
grew these organisms does not save `run_ours`'s live flip counter, so the
number above is the same invariant measured post hoc on every prefix of the
cached anatomy — organs[:k] against organs[:k+1] on the test set, bitwise
(`horizon_growth_flips_check.json`). It is the identical quantity; it is not
the identical measurement. The grown cascade's strict accuracy falls as the
organs multiply: 0.667 at 10 → 0.630 at 20 → 0.592 at 50 →
0.564 at 100 organs. More organs means more inhibition (vetoes rise from
250 to 355 / 303 / 343) and more first-accept boundaries for an input
to fall through, and a one-class organ has no within-organ margin at all.

**Consolidation recovers most of it and the gate does its job three ways.**
The escrow-gated consolidation was accepted on 8 of 9 runs
(refused once, at 20 steps on seed 3, by the original synthetic rule: "worst
class regressed"), lifting strict to 0.717 / 0.747 / 0.729 at 20 / 50 /
100 organs against 0.776 at 10. The territory-respecting arbiter was
refused on all 9 of 9 runs — and the *reasons* are the point:
at 100 organs its candidate was -13.7 to -11.5 points *worse* than the
incumbent on test and the escrow refused it at z = -20.6 to -19.0 on all
three seeds — a genuinely harmful change, caught on held-out data, at the scale
where the arbiter's contract-sample training has 100 organs' worth of
synthetic evidence to be misled by. At 50 organs the candidates were within a
point of the incumbent (one refused by the original rule, two by the escrow at
z ≈ −1.9). At 20 organs one refusal was **helpful-refused**: seed 3's
candidate was +3.8 test points better with an overall escrow z of +4.93,
and the **hard floor** fired on class 47 (a drop of more than a quarter with at
least five lost samples). That is the per-class guarantee doing exactly what
it was built to do, priced at 3.8 points on that seed; the record now stores
which check fired (`escrow_detail`), because the saved record could not
explain this refusal until it did.

**Where this leaves the certified lifecycle.** 0.795 at 10 steps, 0.717 at
20, 0.747 at 50, 0.729 at 100 — against a free (uncertified) arbiter at
0.836 / 0.776 / 0.819 / 0.829, which answers everything and whose gain
is bought outside every territory, and against contracted prototypes that sit
at 0.835 whatever the horizon. The 20-step value is the lowest because of the
one refused consolidation; without that seed it would be 0.756. Reading
for the overview: at a hundred updates the stable configuration has
changed zero answers and lost nothing; the certified lifecycle has changed
zero answers and lost 6.6 points to the 10-step value; the re-solved
head has kept its accuracy and rewritten 41 answers in every hundred. Two
things did not run: the ImageNet-R lifecycles at the longer horizons (the
CIFAR ones took 7.4 h per seed at 100 organs, quadratic in the number of
organs through tightening), and the same-data baselines at the longer
horizons. One protocol change was made and applied at every horizon: the
synthetic pools are bounded by a total budget equal to the inc-10 values
(40,000 / 25,000), so every ≤ 10-organ run is bit-for-bit unchanged. One
bookkeeping bug: the three single-seed `--adolescence` processes per horizon
overwrote a shared `*_adol_protos_inc<k>.json`, so those files hold one seed
each; the per-seed lifecycle files carry the escrow-gated decision for every
seed and are what the table reads. `exp_cifar.py` now merges by seed.

## B-ood — declining, measured (session 41; extended 2026-09-18 with far-OOD sets, a second vision substrate and a language stream)

*Preregistered 2026-09-14. The certificate says every accepted answer comes
from inside a satisfied specification; this asks how well "outside" is
detected. Score = the certificate slack s(x) = max_i q_i(x)/θ_i over the
accept conjunction (margin as τ/margin), so s ≤ 1 ⇔ accept up to vetoes;
cascade = min over non-vetoing organs. Baselines fitted on the seed's train
split, scored on the same points; the joint logistic probe is an upper-bound
comparator (it sees all classes at once and is retrained). Nothing is tuned
on OOD data. Far-OOD caches came from the GPU box (battery 4, block B).
Family `B_ood`.*

| CIFAR-100, ViT-B/16 (frozen, in21k), seeds [2, 3, 4] | ID cov | ID strict | AURC ↓ | DTD (far, textures): AUROC / FPR@95 / **accepted at the contract's threshold** | SVHN (far, digits): AUROC / FPR@95 / **accepted at the contract's threshold** | CIFAR-10 (near): AUROC / FPR@95 / **accepted at the contract's threshold** |
|---|---|---|---|---|---|---|
| **cascade** slack | 0.974 | 0.667 | 0.3376 | 0.985 / 0.024 / **0.036** | 0.502 / 0.825 / **0.858** | 0.683 / 0.731 / **0.818** |
| **lifecycle** slack | 0.984 | 0.776 | 0.1505 | 0.994 / 0.018 / **0.037** | 0.570 / 0.851 / **0.928** | 0.761 / 0.658 / **0.837** |
| **protos** slack | 0.983 | 0.835 | 0.0867 | 0.995 / 0.016 / **0.035** | 0.662 / 0.939 / **0.986** | 0.841 / 0.588 / **0.793** |
| **head** slack | 0.983 | 0.869 | 0.0681 | 0.996 / 0.014 / **0.032** | 0.696 / 0.892 / **0.954** | 0.843 / 0.554 / **0.759** |
| NCM cosine | 1.000 | 0.843 | 0.1254 | 0.993 / 0.029 / — | 0.641 / 0.919 / — | 0.753 / 0.736 / — |
| kNN-1 | 1.000 | 0.843 | 0.1249 | 0.987 / 0.058 / — | 0.650 / 0.942 / — | 0.774 / 0.774 / — |
| Mahalanobis | 1.000 | 0.843 | 0.1485 | 0.998 / 0.008 / — | 0.665 / 0.820 / — | 0.744 / 0.801 / — |
| joint probe, MSP | 1.000 | 0.887 | 0.0176 | 0.970 / 0.135 / — | 0.971 / 0.160 / — | 0.903 / 0.471 / — |

| CIFAR-100, DINOv2-B, seeds [2, 3, 4] | ID cov | ID strict | AURC ↓ | SVHN (far, digits): AUROC / FPR@95 / **accepted at the contract's threshold** | CIFAR-10 (near): AUROC / FPR@95 / **accepted at the contract's threshold** |
|---|---|---|---|---|---|
| **cascade** slack | 0.971 | 0.816 | 0.1704 | 0.612 / 0.659 / **0.721** | 0.891 / 0.335 / **0.422** |
| **lifecycle** slack | 0.991 | 0.865 | 0.0562 | 0.712 / 0.739 / **0.922** | 0.959 / 0.200 / **0.448** |
| **protos** slack | 0.978 | 0.865 | 0.0432 | 0.746 / 0.776 / **0.892** | 0.960 / 0.201 / **0.349** |
| **head** slack | 0.979 | 0.890 | 0.0348 | 0.786 / 0.713 / **0.852** | 0.958 / 0.225 / **0.408** |
| NCM cosine | 1.000 | 0.873 | 0.0562 | 0.693 / 0.823 / — | 0.961 / 0.212 / — |
| kNN-1 | 1.000 | 0.873 | 0.0664 | 0.749 / 0.822 / — | 0.948 / 0.250 / — |
| Mahalanobis | 1.000 | 0.873 | 0.0713 | 0.664 / 0.741 / — | 0.951 / 0.258 / — |
| joint probe, MSP | 1.000 | 0.914 | 0.0113 | 0.951 / 0.309 / — | 0.941 / 0.306 / — |

| CLINC150 intents, all-mpnet-base-v2, seeds [2, 3, 4] | ID cov | ID strict | AURC ↓ | CLINC out-of-scope queries: AUROC / FPR@95 / **accepted at the contract's threshold** |
|---|---|---|---|---|
| **cascade** slack | 0.953 | 0.782 | 0.2079 | 0.840 / 0.457 / **0.447** |
| **lifecycle** slack | 0.959 | 0.813 | 0.1022 | 0.881 / 0.399 / **0.447** |
| **protos** slack | 0.976 | 0.922 | 0.0174 | 0.955 / 0.232 / **0.416** |
| **head** slack | 0.979 | 0.941 | 0.0131 | 0.961 / 0.196 / **0.362** |
| NCM cosine | 1.000 | 0.938 | 0.0209 | 0.978 / 0.100 / — |
| kNN-1 | 1.000 | 0.938 | 0.0350 | 0.976 / 0.115 / — |
| Mahalanobis | 1.000 | 0.938 | 0.0257 | 0.970 / 0.139 / — |
| joint probe, MSP | 1.000 | 0.966 | 0.0039 | 0.969 / 0.160 / — |

**What the certificate declines depends on what "outside" looks like, and
the pattern is consistent across substrates.** Textures (DTD) are almost
entirely refused — at the contract's own operating point the prototypes
accept 3 % and the head 3 % of DTD, AUROC
0.995 / 0.996 — because texture patches land far from every
class territory of a frozen ViT. Street-number digits (SVHN) are accepted
almost entirely (99 % / 95 %, AUROC 0.662 / 0.696):
a low-variance image of a digit projects into the interior of the feature
space, inside territories calibrated to admit 99 % of their own class. CIFAR-10
sits between (79 % / 76 % accepted, AUROC 0.841 / 0.843).
DINOv2-B features separate CIFAR-10 far better (prototypes AUROC 0.960,
35 % accepted) but SVHN no better (0.746, 89 % accepted): the
failure is not the certificate's arithmetic but the geometry of the frozen
encoder, which places digit images inside object territories.

**On language the certificate is a good detector.** CLINC150's out-of-scope
queries against the 150 in-scope intents: prototypes AUROC 0.955, head
0.961, and at the contract's own threshold only 42 % / 36 % of
out-of-scope queries are answered — against kNN-1 0.976, NCM-cosine
0.978, Mahalanobis 0.970 and the joint probe's softmax 0.969. The
sentence-encoder space is one where intent territories are tight and
out-of-scope text falls between them; this is the one setting on record
where "the system declines what it has not seen" is measured to be
approximately true at the deployed threshold.

**Against the baselines.** The preregistered hypothesis (slack ≥ NCM-cosine
on AUROC) holds on every set of every substrate. kNN-1 and Mahalanobis are
below the prototypes' slack on CIFAR-10 and on SVHN; on DTD every
score saturates. The joint probe's softmax is the strongest ranking score
on CIFAR-10 (0.903) and the strongest ID selector everywhere (AURC
0.0176 vs the head's 0.0681 on CIFAR-ViT); on SVHN it is
strong (0.971), so the SVHN blind spot is the encoder's, not the
certificate's. The probe is an upper bound the certified configurations
refuse to pay for (joint retraining).

**Reading for the overview.** "The system may decline" is true and
measured, and the certificate is a *provenance* statement: an accepted
answer came from inside a checked region. What it declines depends on the
substrate: nearly all texture-like input and most out-of-scope text, but not
low-variance images that a frozen vision encoder maps inside object
territories. Do not claim OOD detection in general. Named lever:
contrastive territories, fitted to separate rather than to cover. The
near-only results of 2026-09-14 (CIFAR-10 alone) are archived in
`results/archive/` and are reproduced with zero difference by the extended
run on the two seeds they share. Two process notes. The original seed-4 runs
died when the machine restarted, so seed 4 was recomputed on 2026-09-18 and
merged; its CIFAR-10 numbers sit inside the spread of seeds 2-3. And the
baseline lane was changed between seed 3 and seed 4 to score each OOD set
once and reuse it across the four baselines instead of re-scoring it per
baseline: the scorer is a deterministic function of its fitted state, checked
by scoring two sets twice and comparing bit for bit, so no number moves. It
also did not make the runs faster (162 and 154 minutes against 159 and 143
for seed 2) -- the prediction that the baselines dominated the wall clock was
wrong, and the cost is elsewhere.

## B-cost — what each configuration costs after the encoder (session 41)

*Preregistered above. State to answer, state to keep learning, MACs per
sample, and seconds to absorb one task, per configuration, on three caches;
RanPAC's published head by arithmetic. Family `B_cost`.*

| after the encoder | state to answer (M floats) | state to keep learning (M floats) | MAC / sample (M) | seconds to absorb one task | answers change under growth |
|---|---|---|---|---|---|
| **CIFAR-100 ViT-B** (100 classes, d = 768, 22,500 train rows; BLAS threads 1 (OMP_NUM_THREADS=1 at launch)) | | | | | |
| contracted prototypes | 1.92 | 0.0 | 0.11 | 2.0 | **no** |
| grown cascade (organs + vetoes) | 4.91 | 0.0 | 5.94 | 290.0 | **no** |
| contracted head, M = 2,000 | 3.66 | 4.2 | 1.77 | 1.9 | yes |
| contracted head, M = 10,000 | 10.60 | 101.0 | 8.72 | 108.3 | yes |
| RanPAC head M = 10,000 (published config, arithmetic) | 8.68 | 101.0 | 8.68 | — | yes |
| replay buffer, 20 / class | 301 MB raw pixels / 1.54 M floats as features | — | — | — | (retrains) |
| **CIFAR-100 DINOv2-B** (100 classes, d = 768, 45,000 train rows; BLAS threads 1 (OMP_NUM_THREADS=1 at launch)) | | | | | |
| contracted prototypes | 1.92 | 0.0 | 0.11 | 3.3 | **no** |
| grown cascade (organs + vetoes) | 4.82 | 0.0 | 5.94 | — | **no** |
| contracted head, M = 2,000 | 3.66 | 4.2 | 1.77 | 3.4 | yes |
| contracted head, M = 10,000 | 10.60 | 101.0 | 8.72 | 136.6 | yes |
| RanPAC head M = 10,000 (published config, arithmetic) | 8.68 | 101.0 | 8.68 | — | yes |
| replay buffer, 20 / class | 301 MB raw pixels / 1.54 M floats as features | — | — | — | (retrains) |
| **VTAB ViT-B** (50 classes, d = 768, 1,617 train rows; BLAS threads 1 (OMP_NUM_THREADS=1 at launch)) | | | | | |
| contracted prototypes | 0.96 | 0.0 | 0.08 | 0.5 | **no** |
| grown cascade (organs + vetoes) | 1.82 | 0.0 | 2.52 | 86.1 | **no** |
| contracted head, M = 2,000 | 2.60 | 4.1 | 1.67 | 2.2 | yes |
| contracted head, M = 10,000 | 9.14 | 100.5 | 8.22 | 82.8 | yes |
| RanPAC head M = 10,000 (published config, arithmetic) | 8.18 | 100.5 | 8.18 | — | yes |
| replay buffer, 20 / class | 151 MB raw pixels / 0.77 M floats as features | — | — | — | (retrains) |
| shared encoder (vitb) | 86 M params | — | 17600 M | — | frozen |

**Reading.** Everything here is downstream of a shared frozen encoder that
costs 17.6 GMAC per image and 86 M parameters, so the whole comparison is
about the last few percent of compute — and about state. Contracted
prototypes are the cheapest by every measure (1.9 M floats, 0.11 M MAC,
2 s per task, nothing kept for learning). The grown cascade is
4.9 M floats and 5.9 M MAC on CIFAR — every organ's territories are
scored, which is what buys the certificate — and organ birth is the slowest
update in the project at 290 s per task on one thread, most of it the
inhibition solved against every earlier organ. The contracted head at
M = 2,000 costs about what the prototypes cost to answer (3.7 M) but keeps a
4.2 M-float Gram to go on learning; at M = 10,000, the width RanPAC
publishes, that learning state is 101 M floats — more than the encoder
itself — and one re-solve takes 108 s. RanPAC's own head is the same
arithmetic without the territories (8.7 M + 101 M). A 20/class replay
buffer is 301 MB of pixels, or 1.5 M floats if features may be
kept, plus a retraining run per update. The two configurations that never
change an answer are also the two that carry no learning state: the cascade
and the prototypes learn by *adding*, and there is nothing to re-solve.
Timing is single-threaded wall-clock on the analysis machine and is a scale
indication, not a tuned quantity.

## B-ptm-confirm — the headline table on its own untouched seeds (session 42)

*Battery 4, block A, on the GPU box (`docs/BATTERY4_RESULTS.md`; every chain's
`.err` empty). The first-session-adapter recipe of battery 3 (rank-16 parallel
adapters on task 0, 20 epochs, then frozen), run on gate-confirmation seeds
5–7 for all seven datasets — 21 new seed-specific caches — followed by the
unchanged chain: contracted prototypes, contracted head (M = 10,000), grown
cascade, gated consolidation, gated arbiter. Reading rule as for the MLP
confirmation block: accepted regressions and helpful refusals counted, nothing
re-tuned. Family `B_phase1_confirm`; the eval-seed reference is `B_phase1`.*

| dataset | protos strict 2–4 → 5–7 | head strict 2–4 → 5–7 | certified 2–4 → 5–7 | consolidation acc. (5–7) | arbiter acc. (5–7) | growth flips | uncertified max |
|---|---|---|---|---|---|---|---|
| CIFAR-100 | 0.867 → 0.859 | 0.906 → **0.903** | 0.858 → 0.847 | 3/3 | 0/3 | 0 | 0.0000 |
| ImageNet-R | 0.695 → 0.697 | 0.763 → **0.769** | 0.571 → 0.576 | 3/3 | 1/3 | 0 | 0.0000 |
| ImageNet-A | 0.516 → 0.535 | 0.559 → **0.573** | 0.510 → 0.543 | 3/3 | 0/3 | 0 | 0.0000 |
| CUB-200 | 0.854 → 0.858 | 0.891 → **0.888** | 0.740 → 0.812 | 3/3 | 1/3 | 0 | 0.0000 |
| ObjectNet-200 | 0.536 → 0.525 | 0.639 → **0.642** | 0.429 → 0.408 | 1/3 | 0/3 | 0 | 0.0000 |
| OmniBenchmark | 0.711 → 0.712 | 0.766 → **0.766** | 0.656 → 0.661 | 0/3 | 1/3 | 0 | 0.0000 |
| VTAB | 0.882 → 0.892 | 0.939 → **0.936** | 0.882 → 0.887 | 3/3 | 0/3 | 0 | 0.0000 |

**The headline table replicates.** The contracted head with the birth step
moves by at most 1.4 points on any dataset between seed blocks (prototypes by at
most 1.8), and it stays above the best published APER variant on 7 of 7
datasets on the untouched seeds as it did on the evaluation seeds. Growth flips
are 0 on 21/21 seeds; the uncertified fraction is 0.0 on 21/21; forced
prototypes equal nearest-class-mean on 21/21. The overview's sentence
about "two independent blocks of unseen seeds" now covers the headline table
itself, not only the MLP benchmarks.

**The gate record on seeds 5–7.** Consolidation accepted 16/21; the five
refusals are ObjectNet (2) and OmniBenchmark (3), all "mean regressed" on the
synthetic holdout — with battery 3, adapter-tuned OmniBenchmark consolidations
are now refused on 6 of 6 seeds, the birth step damaging that substrate every
time and the gate catching it every time. Arbiter accepted 3/21. **One accepted
regression**: CUB seed 7, candidate -0.4 points below the incumbent on
test, escrow z -0.83 — inside the −1.645 tolerance, below the escrow's
resolution on 2,358 test points; it joins the satimage seed-6 case (−0.7) as
the second sub-resolution slip on record, against zero clearly-helpful
refusals (0 candidates better than the incumbent by more than a point were
refused). **The CIFAR arbiter does not replicate**: accepted 3/3 on seeds 2–4
with candidates +0.6 / +0.8 / +0.4 above the incumbent, refused 3/3 on seeds
5–7 with candidates +0.7 / +0.6 / +0.2 and escrow z +2.9 / +3.5 / +3.5 — all
by the original rule's +0.01 synthetic-holdout requirement. The configuration
sits on the rule's threshold; 3/6 overall, and the certified CIFAR value with
adapters is 0.858 on the evaluation seeds and 0.847 (the
consolidated organism, arbiter refused) on the confirmation seeds. Both are
reported; neither is the headline, which is the head.

**A reproducibility note the box surfaced.** On the evaluation seeds the box's
own re-run of the battery-3 caches agrees with this machine to ±0.003 on
prototypes and heads (closed-form, bitwise up to BLAS summation order) but not
always on the *gate decisions*: CUB certified 0.787 there against 0.740 here,
OmniBenchmark 0.651 against 0.656, one arbiter decision each. Likewise the
three battery-2 `ptm_*_vitb1k.json` files the box re-ran differ from the
copies on this machine by up to 0.8 points in *growth* strict and 0.3 in
replay (torch training under different thread counts), while prototypes and
NCM agree to the last digit. The files on record here are the battery-2
originals; the discrepancy is noted, not resolved: closed-form configurations
reproduce across machines, trained ones reproduce to within a point, and a
gate decision within a point of its threshold can go either way.

## B-text — the certificate on a language substrate (session 42)

*Battery 4, block C. Two intent-classification streams on a frozen sentence
encoder (`sentence-transformers/all-mpnet-base-v2`, mean-pooled, fixed in
advance, never tuned): Banking77 (77 classes, 10,003 / 3,080, inc 7) and
CLINC150 (the 150 in-scope intents, 18,000 / 4,500 with HF's validation split
folded into train, inc 15; the 1,350 out-of-scope queries become an OOD
cache). Evaluation seeds 2–4, the standard chain unchanged — the caches are
768-d and carry the `ptm_` prefix, and that is all the pipeline needs. No
published anchor is quoted: none has been read from a source PDF. Families
`B_ptm` / `B_head` under `banking77_mpnet`, `clinc150_mpnet`.*

| stream (frozen all-mpnet-base-v2, 768-d) | growth strict (flips) | contracted protos strict / forced | head 10k strict / forced / cov / changes per 100 | consolidation | certified lifecycle (arbiter) | NCM | replay 20/cl | joint MLP | EWC | fine-tune |
|---|---|---|---|---|---|---|---|---|---|---|
| **Banking77 (77 classes, 11 × 7)** | 0.755 (0) | 0.853 / 0.871 | **0.921 / 0.939** / 0.979 / 7 | 3/3 → 0.790 | 0.814 (2/3) | 0.871 | 0.918 | 0.881 | 0.347 | 0.204 |
| **CLINC150 (150 in-scope, 10 × 15)** | 0.782 (0) | 0.922 / 0.938 | **0.950 / 0.965** / 0.981 / 4 | 3/3 → 0.813 | 0.830 (2/3) | 0.938 | 0.956 | 0.928 | 0.639 | 0.253 |

**The certificate transfers unchanged.** Growth flips 0 and uncertified
fraction 0.0 on 6/6 seeds; forced prototypes equal nearest-class-mean on 6/6.
Nothing in `cgrow/` or the experiment scripts was touched; the only code change
on the box was the dataset loader (`datasets` 5 refuses script-based Hub repos;
the CSVs the original `banking77` script downloads are read directly and mapped
with its own class names, so texts, splits and label ids are identical).

**Where it lands.** The contracted head is the strongest configuration on both
streams — above same-data replay on forced accuracy (Banking77 0.939 vs
0.918; CLINC150 0.965 vs 0.956) and above a jointly trained MLP
(0.881, 0.928) — while covering 98–98 % of the test set. The
prototypes equal NCM (0.871, 0.938) and beat the joint MLP on CLINC150. The
certified lifecycle (0.814, 0.830) trails the prototypes by 6–11 points because
the consolidated organ answers only 51–59 % (Banking77) and 39–40 % (CLINC150)
of the inputs and the grown cascade answers the rest — the same reach limit
seen on the vision suite.

**The arbiter's first acceptances outside CIFAR.** 4 of 6, each lifting the
certified number by 3–4 points with escrow z between +1.6 and +5.7 and no
accepted regression. The two refusals (seed 2 on both streams) came from the
original rule's last-task real-validation clause while the synthetic holdout
and the escrow both favoured the candidate — and in both cases the refused
candidate scored higher on test (+2.9 and +4.0 points). Two helpful refusals,
both by the same clause: it is conservative on these streams and is now the
gate component with the clearest cost on record.

## B-tabular — new attack categories on raw flow records (session 42)

*Battery 4, block D. NSL-KDD, no encoder: 122 features after one-hot
protocol / service / flag and log1p on non-negative numerics; classes kept iff
≥ 300 train and ≥ 30 test rows (8 of 40 attack types and normal: back,
ipsweep, neptune, nmap, normal, portsweep, satan, smurf; the 32 dropped, with
their counts, are printed verbatim in `docs/BATTERY4_RESULTS.md`), 5,000-row cap
per class, inc 2 → 4 tasks; 25,258 / 16,498 rows. The 768-d CIFAR organ
configuration ran unmodified on 122-d rows. Seeds 2–4. Joint NCM 0.763 on the
raw cache. Family `B_ptm` / `B_head` under `nslkdd_raw`.*

| NSL-KDD, seeds 2–4 | strict | forced / coverage | changes under growth |
|---|---|---|---|
| grown cascade | 0.781 (per seed 0.68 / 0.89 / 0.78) | 0.782 / 0.81–0.99 | **0** |
| contracted prototypes | 0.428 | 0.445 | **0** |
| contracted head, M = 10k | **0.935** | 0.951 / 0.978 | 0–5 per 100 |
| consolidation | refused 3/3 (regressed on the last task's real validation) | | |
| certified lifecycle | 0.781 (= cascade; arbiter refused 3/3, escrow z −8 to −15) | | **0** |
| NCM / replay / joint MLP / EWC / fine-tune | 0.445 / 0.951 / 0.952 / 0.802 / 0.624 | | |

**Reading.** The certificate transfers to raw tabular rows — flips 0,
uncertified 0.0, forced prototypes = NCM on 3/3 — but nearest-class-mean is a
poor readout on this space (0.445: one-hot / log1p flow records are not
separable by distance to a class mean), so the prototype configuration is the
weak one here and the head is the strong one, at the level of replay and a
joint MLP (0.951 vs 0.951 / 0.952) with 98 % coverage. The grown
cascade sits between them and varies a lot by seed (0.68–0.89): the 768-d organ
configuration is not visibly wrong on 122-d inputs, and it was not tuned. The
gate refused everything, correctly: the consolidated prototype organ inherits
NCM's weakness and regressed on real validation on every seed, and the arbiter
candidates were 1–25 points below the organism with the strongest escrow
refusals of any battery (z −8.4 / −14.6 / −9.4). One dataset was run because it
was neither trivially easy nor broken, per the handoff rule; UNSW-NB15 and
CIC-IDS2017 remain available. This is the overview's use case in
miniature, and its honest reading is: the guarantees hold, the accuracy
configuration is the head, and the prototype route needs a representation
before it is useful on this kind of data.

## B-in1k-phase1 — the RanPAC comparison on its own checkpoint family (session 42)

*Battery 4, block E (optional; run last). The ImageNet-1K-fine-tuned ViT-B/16
(`augreg_in21k_ft_in1k`), frozen, on the three datasets battery 2 had not
covered (ObjectNet, OmniBenchmark, VTAB), and with the first-session-adapter
birth step on all six RanPAC datasets, seeds 2–4; prototypes and head only
(steps 1, 2, 5). RanPAC's Table 1 does not state its checkpoint per column;
its NCM-only row matches the IN1K SimpleCIL block on ImageNet-R / -A / CUB and
the IN21K values elsewhere (`docs/RANPAC_VS_US.md`), so "same checkpoint family" is
established for those three. Its metric A_T (mean of per-task accuracies)
differs from our overall last-step accuracy by a few tenths where test sets
are unbalanced. Families `B_in1k` (frozen), `B_in1k_phase1` (+ adapters).*

| dataset | in21k frozen protos / head | in1k frozen protos / head | in1k + adapters protos / head (strict / forced) | RanPAC full (A_T) |
|---|---|---|---|---|
| ImageNet-R | 0.524 / 0.652 | 0.622 / 0.693 | 0.700 / **0.770** (forced 0.777) | 0.779 |
| ImageNet-A | 0.454 / 0.451 | 0.558 / 0.560 | 0.529 / **0.602** (forced 0.607) | 0.624 |
| CUB-200 | 0.858 / 0.867 | 0.849 / 0.875 | 0.848 / **0.886** (forced 0.895) | 0.903 |
| ObjectNet-200 | 0.506 / 0.531 | 0.504 / 0.551 | 0.521 / **0.631** (forced 0.637) | n.a. |
| OmniBenchmark | 0.724 / 0.756 | 0.722 / 0.762 | 0.707 / **0.764** (forced 0.772) | 0.799 † |
| VTAB | 0.888 / 0.929 | 0.885 / 0.932 | 0.871 / **0.939** (forced 0.944) | 0.922 † |

† RanPAC's OmniBenchmark and VTAB cells are consistent with the in21k
checkpoint; the like-for-like rows for those are the in21k + adapters rows of
`B_phase1` (head 0.766 / 0.774 and 0.939 / 0.943), which the in1k rows equal
within 0.002.

**With the checkpoint asterisk removed the gap to RanPAC is a few tenths to a
point and a half.** Contracted head + adapters on the 1k checkpoint: forced
0.777 vs 0.779 on ImageNet-R (-0.2), 0.607 vs 0.624 on ImageNet-A
(-1.7), 0.895 vs 0.903 on CUB (-0.8) — strict is 0.7–0.9 lower still,
and the A_T-vs-last metric difference is unresolved and worth a few tenths on
ImageNet-A's unbalanced test set. The comparison table's RanPAC column can
now be read on the same footing for those three datasets; the remaining
difference is the metric definition, named, not a checkpoint.

**What the checkpoint and the birth step are each worth.** The 1k checkpoint
alone moves the frozen prototypes by +7.7 / +6.5 points on ImageNet-R /
ImageNet-A and by less than a point elsewhere, and the frozen head by +4.1 /
+7.7 / ≤ 1.3. Adapters on the 1k checkpoint move the head by +7.7 / +3.2 / +1.2 /
+6.3 / +0.3 / +0.7 but the *prototypes* by +7.9 / −1.8 / −0.5 / +1.1 / −1.4 /
−1.1: on four of six datasets first-session tuning makes the prototype space
slightly worse while the ridge head still gains — the OmniBenchmark divergence
of battery 3, seen again. Growth flips 0 on 36/36 seeds; forced prototypes =
NCM on 36/36.

## B-size and B-prune — how big the grown network is, and how much of it is needed (session 44, 2026-10-02)

*Two questions the project had never asked: how large is the thing it grows,
and is any of that size load-bearing. `exp_size.py` counts an organism built
through the ordinary path (`exp_lifecycle.build`, same config, splits and
seeds) and scores the same object, so size and accuracy cannot drift apart.
`exp_prune.py` then prunes at birth and sweeps the sparsity. Split-MNIST,
evaluation seeds 2–4. Records `size_mnist.json`, `prune_mnist.json`; families
`B_size_mnist`, `B_prune_mnist`.*

**The unpruned organism.** Five organs, one per task, each a 784→256→32 trunk:
**5,148 neurons and 1,117,013 connections** (784 shared input units,
1,440 trunk units, ~2,900 veto units; 1,044,480 trunk weights and ~72,500 veto
weights), plus 196,595 contract floats — anchors, territories and
thresholds, which are compared against rather than computed through, and which
are what replaces a replay buffer. Strict accuracy 0.8930, coverage
0.985, growth flips 0 over 12 additions. For scale, the single network the
baselines train on the same data is 784-256-10: 1,050 neurons and
203,264 connections, i.e. the organism is 5.5× it, which is
what one organ per task costs when nothing is shared.

**Pruning, and where it is allowed.** At birth, before the organ is frozen:
train the trunk, drop the smallest-|w| fraction of each layer, retrain the
survivors with the dropped weights pinned at zero, *then* solve the thresholds
and freeze. The contract is therefore solved against the network that ships.
Pruning a *frozen* organ is unsound — its anchors, radii and territory describe
a function that no longer exists — and `cgrow/organ.py` offers no way to do it.
Preregistered in `exp_prune.py` before the runs: flips stay 0 (structural, so a
non-zero would be a bug and not a finding), sparsity 0 reproduces the control
bit for bit, and accuracy holds to some sparsity and then falls — the
experiment measures where.

| trunk sparsity | organs | strict | coverage | trunk connections | veto connections | **total** | compression | growth flips |
|---|---|---|---|---|---|---|---|---|
| 0 | 5 | 0.8930 ± 0.0013 | 0.985 | 1,044,480 | 72,533 | **1,117,013** | 1.0× | **0** |
| 0.5 | 5 | 0.8959 ± 0.0013 | 0.985 | 522,240 | 70,400 | **592,640** | 1.9× | **0** |
| 0.8 | 5 | 0.8930 ± 0.0021 | 0.985 | 208,895 | 72,533 | **281,428** | 4.0× | **0** |
| 0.9 | 5 | 0.8941 ± 0.0032 | 0.985 | 104,445 | 71,467 | **175,912** | 6.3× | **0** |
| 0.95 | 5 | 0.8943 ± 0.0022 | 0.986 | 52,225 | 72,533 | **124,758** | 9.0× | **0** |
| 0.98 | 5 | 0.8937 ± 0.0026 | 0.984 | 20,890 | 70,400 | **91,290** | 12.2× | **0** |
| 0.99 | 5 | 0.8914 ± 0.0019 | 0.984 | 10,445 | 70,400 | **80,845** | 13.8× | **0** |
| 0.995 | 5 | 0.8843 ± 0.0040 | 0.984 | 5,225 | 69,333 | **74,558** | 15.0× | **0** |
| 0.998 | 5 | 0.8129 ± 0.0570 | 0.947 | 2,085 | 72,533 | **74,618** | 15.0× | **0** |
| 0.999 | 4 | 0.4951 ± 0.1532 | 0.745 | 836 | 48,000 | **48,836** | 22.9× | **0** |

**The trunks were almost entirely redundant.** Up to 99 % sparsity the strict
accuracy does not move outside its seed spread (0.8930 unpruned against
0.8914 at 99 %), while the trunks fall from 1,044,480 connections to
10,445. At 99.5 % the cost is still under a point (0.8843); the cliff is
between 99.5 % and 99.8 %, where the mean drops to 0.813 and the seed
spread explodes (per seed [0.8457, 0.7327, 0.8601]), and by 99.9 % the organism has
collapsed (0.495, per seed [0.2858, 0.6482, 0.5512]) — and has only four
organs, because the fifth could no longer meet its own precision target and the
growth rule refused it. **Growth flips are 0 at every sparsity, including the
collapsed ones.** That is the useful shape of the result: the guarantee does
not degrade with size, only the accuracy does, so the search for a smaller
organism is an optimisation under a hard constraint rather than a trade-off.

**Where the size actually is, after pruning.** The compression of the *system*
saturates at about 15× while the trunks shrink 200×, because the vetoes are not
pruned: at 99 % they are 70,400 of the 80,845 remaining connections, 87 % of
the total. The contract's 196,595 floats are larger still — more than double the
connection count. Anyone continuing this should attack those two, not the
trunks, which are now close to exhausted. Against the small-network literature
(Han et al. 2015, `Learning both Weights and Connections`, Table 1: a
LeNet-300-100 pruned 12× to ~22K parameters at 98.4 %, **jointly trained**, a
different protocol from this one and not comparable on accuracy), the trunks at
99 % hold 10,445 weights, half that budget; the system around them does not.

**Two bugs, both found by the sweep.** (1) The log line formatted the sparsity
with two decimals, so the 0.995 rows printed as "0.99" and sat under the real
0.99 rows looking like seed noise; the stored records were always right, and
the table above is built from them. Fixed. (2) A latent bug in `run_ours`,
present since the beginning and never reachable before: the birth veto is
guarded by `t > 0`, which assumes that being past the first task means an
earlier organ exists. It does not, if growth was *refused* on every earlier
task — which first happened at sparsity 0.998, where the first organ was too
small to pass its own gate. The guard is now `len(org.organs) > 1`. It took a
deliberately crippled organ to expose an assumption that had been silently
true for every configuration ever run.

## B-head-blocks — a head that never re-solves: the trade does not dissolve (session 40)

*The contracted head (`B-head`) is the accuracy configuration and the cascade is
the retention configuration, and until now nothing had both: the head re-solves
its ridge after every task, so 8 to 155 already-given answers per 100 test
points change over a run, against exactly 0 for the cascade. This asks whether a
head can be frozen the way an organ is. `exp_head_blocks.py`: each task gets its
own ridge block, solved from that task's rows alone with its own λ, margin τ and
territory, then frozen; answers come from first-accept over blocks in birth
order. Evaluation seeds 2–4, M = 2,000. Family `B_head_blocks`.*

| substrate | **frozen blocks** strict / forced (cov) | re-solved head | contracted protos | certified lifecycle | growth flips | tightening repairs |
|---|---|---|---|---|---|---|
| CIFAR-100, ViT-B/16 | **0.482** / 0.486 (0.99) | 0.869 | 0.835 | 0.794 | **0** | 21717 → abstain, 17238 → other (5082R:435W) |
| CIFAR-100, DINOv2-B | **0.685** / 0.699 (0.97) | 0.890 | 0.865 | 0.865 | **0** | 21328 → abstain, 9676 → other (3809R:522W) |
| VTAB, ViT-B/16 | **0.381** / 0.383 (1.00) | 0.923 | 0.888 | 0.886 | **0** | 2304 → abstain, 448 → other (148R:15W) |

**The invariant holds, exactly.** Appending a block changed **zero** answers on
every seed of every substrate, checked bitwise, including in the ablation. That
was the point of the design and it is delivered: a frozen ridge block behaves
like a frozen organ.

**The accuracy does not survive.** Against the re-solved head the frozen-block
version loses 38.7 points on CIFAR-ViT (0.482 vs 0.869), 20.5 on DINOv2-B
(0.685 vs 0.890) and 54.2 on VTAB (0.381 vs 0.923). It is also below the
contracted prototypes on all three, so it is not merely worse than the head: it
is the weakest certified configuration in the project.

**Why, and why this is not an implementation gap.** A ridge readout's advantage
over class means is that it is solved over all classes *jointly*, which
decorrelates them; RanPAC's own ablation prices that at +3.6 to +10.6 points. A
block frozen at birth is solved over one task's classes with no negatives from
any other, so it cannot decorrelate against classes that do not exist yet.
Retention demands that an answer be fixed when it is first given; joint
decorrelation demands knowledge of every class. The two requirements are in
direct conflict, and no amount of engineering inside this design removes it.
The gap widens exactly where the head's joint solve helps most: VTAB, where the
re-solved head beats the prototypes by 4.1 points, loses the most from freezing.

**What the tightening is worth, and what it costs.** Without the monotone
tightening the first block answers everything: strict 0.148 at
coverage 1.00 on CIFAR-ViT, against 0.482 with it, so the
inhibition is worth about 32 points and is what makes the design function at
all. It is not free of consequence, and the accounting is the flat protocol's:
tightening a block is monotone for that block — τ only rises, ρ_in only falls,
so its own answer can become an abstention but never a different answer — yet in
a cascade the abstention falls through and a later block may answer differently.
Those are logged repairs, 21717 to abstention and 17238 to a
different block on CIFAR-ViT, net positive at 5082 right against
435 wrong, the same signature the organs show. Growth flips and
tightening repairs are counted separately, and an earlier draft of this
experiment conflated them and briefly reported 13,607 "flips"; that number was a
measurement error, not a broken invariant.

**Where this leaves the question.** Accuracy and bitwise retention remain
separate configurations, and the reason is now named rather than assumed. The
honest recommendation is unchanged: use the contracted prototypes or the
certified lifecycle when answers must not move, and the re-solved contracted
head when accuracy matters and coverage semantics suffice. A design that could
close the gap would need negatives for classes it has not seen — which is what
contract sampling provides for the *organs*, so the untried variant is a block
solved against samples drawn from earlier blocks' territories. That buys
decorrelation against the past, not the future, so it should be expected to
recover part of the gap and not all of it.

## B-adol-escrow — the adolescence gate gets the full real-data escrow (session 39)

*Until 2026-09-09 a consolidation was judged on the synthetic contract holdout,
per class against the grown organism, plus the LAST task's real validation
split. Every other class's real-data evidence was absent, and the
synthetic-versus-real fidelity gap is documented (wall-robot). The gate now
also faces the escrow the arbiter gate uses — one-sided McNemar over the whole
validation split, Bonferroni per-class flags, hard floor for single-class
collapse — comparing the consolidated organism against the grown one over every
class seen so far. `cgrow/adolescence.py`; A/B harness `exp_adol_escrow.py`;
family `B_adol_escrow`. The old rule stays reachable
(`cfg["adol_escrow"]=False`) so the change is measured, not asserted.*

**What the escrow says about the decisions already on record.** Every cached
grown organism (67 of them, across 14 substrates) was replayed through
three gate arms. Of the 57 decisions that reached the escrow, **not one had
negative overall evidence**: McNemar z ran from +3.78 to +22.22 with a median
of +13.73. The adolescence gate's synthetic evidence was therefore not
misleading — every consolidation it had accepted is decisively supported by real
held-out data over all classes. That is the strongest independent confirmation
the frozen gate has received, and it was worth running before trusting the
change.

| gate arm | consolidations accepted, of 67 |
|---|---|
| old: synthetic holdout + last task's validation | 57 |
| escrow transplanted verbatim from the arbiter gate | 54 |
| **escrow with the corrected hard floor (final)** | **56** |

**The transplant exposed a flaw in the shared escrow.** The 0.25 per-class hard
floor was a FRACTION with no condition on how many samples the breach
represented. On ImageNet-R — about 120 escrow classes at a median of 14
validation samples — one class going from 7/10 to 4/10, three samples, refused a
consolidation that improved overall validation accuracy from 0.289 to 0.456
(z = +18.1) and improved 94 of 119 classes. Transplanted verbatim the floor
produced 3 such refusals, costing 16.5, 12.2, 7.3 test points.
The floor now also requires the loss to amount to at least five samples, which
only excludes swings of one to four — swings that cannot survive the
multiplicity correction the flagged test already applies (with ~120 classes that
threshold is z ≈ 4.7, needing ~22 net losses) — while the collapse the floor
exists for, a class falling from ~0.9 to 0 on 20+ samples, loses 18 or more and
is still caught. Unit-tested both ways. The floor had never fired in the 41
recorded arbiter escrow decisions, so the condition cannot disturb any verdict
already on record.

**One refusal stands, and it is priced.** imr_vitb seed 3: a class falls from
24 of 26 validation samples to 17 of 26, a 26.9% drop on real evidence, and the
floor catches it. Refusing costs 12.2 test points while the consolidation
would have improved 84 of 117 classes and overall validation from 0.288 to
0.405. This is the doctrine working as written: no single class may collapse,
whatever the aggregate gain, and the cost of that promise is reported rather
than hidden. ImageNet-R's certified lifecycle moves from 0.431 to 0.391 as a
result.

**Scope of the re-judgement.** The escrow is an additional check reached only
after the earlier ones pass, so it can turn an acceptance into a refusal but
never the reverse: the 29 prior refusals in the flagship family are untouched by
construction. Of its seven acceptances, CIFAR-ViT's three were covered by the
replay and stand; digits seed 4 and all three sweeps seeds were re-run and are
unchanged (escrow z +0.0, +4.80, +1.41, +3.87). The flagship family's certified
numbers are therefore identical, and the only number in the project that moves
is ImageNet-R's frozen-backbone lifecycle.

**Gate record after the change.** Zero accepted regressions still stands across
the flagship family. The adolescence gate has now been audited on real
held-out data over 57 decisions, of which it refuses one for protecting a
single class.

## B-phase1 - first-session adapters, and the checkpoint confound (session 38)

*Batteries 2 and 3 of `docs/HANDOFF_GPU2.md`, extracted on the RTX 3070 (24 caches)
and analysed here on evaluation seeds 2-4 under the frozen gate. Phase 1 =
parallel bottleneck adapters (rank 16, 0.31 M parameters) on every block's MLP,
trained 20 epochs on the first task's classes of the seed being evaluated, then
frozen; the caches are seed-specific and each was run with its own seed alone.
Nothing downstream changed, so the difference from the frozen-backbone rows is
Phase 1 alone. Full tables: `docs/PHASE1_RESULTS.md`; families `B_phase1`, `B_in1k`.*

| dataset | protos (frozen -> +adapters) | head M=10k (frozen -> +adapters) | certified lifecycle (frozen -> +adapters) | APER-adapter | RanPAC full | verdict after Phase 1 |
|---|---|---|---|---|---|---|
| CIFAR-100 | 0.832 -> **0.867** | 0.879 -> **0.906** | 0.825 -> **0.858** | 0.875 | 0.922 | above APER-adapter; 0.016 under RanPAC-full |
| ImageNet-R | 0.524 -> **0.695** | 0.652 -> **0.763** | 0.431 -> **0.571** | 0.680 | 0.779 | above APER-adapter; 0.016 under RanPAC-full |
| ImageNet-A | 0.454 -> **0.516** | 0.451 -> **0.559** | 0.450 -> **0.510** | 0.494 | 0.624 | above APER-adapter; 0.065 under RanPAC-full |
| CUB-200 | 0.858 -> **0.854** | 0.867 -> **0.891** | 0.822 -> **0.740** | 0.871 | 0.903 | above APER-adapter; 0.012 under RanPAC-full |
| ObjectNet-200 | 0.506 -> **0.536** | 0.531 -> **0.639** | 0.400 -> **0.429** | 0.580 | - | above APER-adapter |
| OmniBenchmark | 0.724 -> **0.711** | 0.756 -> **0.766** | 0.669 -> **0.656** | 0.744 | 0.799 | above APER-adapter; 0.033 under RanPAC-full |
| VTAB | 0.888 -> **0.882** | 0.929 -> **0.939** | 0.886 -> **0.882** | 0.844 | 0.922 | above APER-adapter; above RanPAC-full |

**Five readings.**

1. *A birth step is worth points, and the doctrine allows it.* Tuning the
   substrate once on organ 0's data and freezing it for life leaves every later
   answer unaffected, so the retention invariant is untouched. Feature quality
   (joint nearest-class-mean over all classes at once) rises by +16.3 on
   ImageNet-R, +13.1 on ImageNet-A, +8.9 on CIFAR-100, +5.8 on ObjectNet and
   +4.8 on VTAB, barely moves on CUB (+0.6, already 0.862 frozen), and *falls*
   on OmniBenchmark (-1.2).
2. *OmniBenchmark refuses the birth step, as the literature says it should.*
   Zhou et al. report their fine-tuned variant below SimpleCIL on that
   benchmark; its 300 classes span domains and the first task shows only 30, so
   tuning on them narrows the space. It is the clearest case in the project of a
   substrate change a gate would have to refuse, and it argues for gating
   Phase 1 itself.
3. *The gate blocked what the birth step broke.* On OmniBenchmark, the one
   dataset where Phase 1 damages the feature space, `consolidate_protos` was
   refused on all three seeds ("mean regressed": 0.764 against the grown
   organism's 0.831) and the anatomy stayed as grown. The machinery caught the
   bad substrate change downstream without being told about it.
4. *CUB seed 4 is a weaker substrate, not a gate failure.* Its certified
   endpoint is 0.639 against 0.813 frozen and its consolidated organ answers
   only 12.8% of the real test set, but replaying the decision through the full
   escrow passes it (McNemar z = +3.78 over every class, real validation 0.655
   to 0.677, test 0.616 to 0.639): the consolidation improved the organism it
   was applied to. The endpoint difference is the substrate that seed drew, and
   a gate compares a change against the organism before it, never against a
   different run. An earlier draft called this a gate miss; withdrawn.
5. *The checkpoint confound in the RanPAC comparison is closed.* The in1k caches
   reproduce RanPAC's "NCM only" row on ImageNet-R (0.611 vs 0.612) and CUB
   (0.855 vs 0.851); ImageNet-A comes out 2 points high (0.513 vs 0.493),
   which the metric difference allows, since RanPAC's `A_T` is a mean over
   per-task accuracies and that dataset's per-class test counts are unbalanced.

## B-head — the contracted head: RanPAC's lever, under a contract (session 35)

*RanPAC's ablations placed our contracted prototypes at its "NCM only" level and
showed that its head alone (random projection to M dimensions, ReLU, ridge
classifier solved in closed form from accumulated second-order statistics)
buys +3.6 to +10.6 points on frozen features. `exp_head.py` puts that head
under the contract machinery: per-class PCA territory in the feature space
(fitted once at the class's arrival), a cosine-style margin on the head's
top-two scores solved on validation, accept iff inside the predicted class's
territory and the margin clears τ. The head is order-invariant and
exemplar-free and never sees a gradient; λ is chosen on the validation split
(grid, at every task for M = 2,000 and at the last task for M = 10,000).
Evaluation seeds 2–4, random class order (the anchors' protocol, so these rows
ARE comparable to the published numbers). Family `B_head[cache][M]`.*

| dataset (seeds 2–4) | contracted protos strict / forced | **head M=2000** strict / forced | **head M=10000** strict / forced | coverage (M=10k) | answer-change events under growth, per 100 test points (M=10k) | RanPAC: NCM only / head only / full |
|---|---|---|---|---|---|---|
| CIFAR-100, ViT-B/16-in21k | 0.835 / 0.843 | 0.869 / 0.878 | 0.875 / 0.884 | 0.98 | 15 | 0.834 / 0.890 / 0.922 |
| CIFAR-100, ViT-B full data | 0.832 / 0.842 | 0.870 / 0.879 | 0.879 / 0.888 | 0.98 | 11 | — |
| CIFAR-100, DINOv2-B | 0.865 / 0.873 | 0.890 / 0.898 | 0.901 / 0.909 | 0.98 | 8 | — |
| CIFAR-100, ViT-L/16 | 0.867 / 0.877 | 0.897 / 0.909 | 0.904 / 0.915 | 0.98 | 8 | — |
| ImageNet-R | 0.524 / 0.551 | 0.633 / 0.641 | 0.652 / 0.659 | 0.99 | 66 | 0.612 / 0.718 / 0.779 † |
| ImageNet-A | 0.454 / 0.511 | 0.423 / 0.480 | 0.451 / 0.498 | 0.74 | 155 | 0.493 / 0.582 / 0.624 † |
| CUB-200 | 0.858 / 0.866 | 0.857 / 0.867 | 0.867 / 0.876 | 0.99 | 14 | 0.851 / 0.887 / 0.903 † |
| ObjectNet-200 | 0.506 / 0.543 | 0.516 / 0.547 | 0.531 / 0.561 | 0.84 | 74 | — |
| OmniBenchmark | 0.724 / 0.733 | 0.734 / 0.745 | 0.756 / 0.766 | 0.97 | 19 | 0.731 / 0.782 / 0.799 |
| VTAB | 0.888 / 0.893 | 0.923 / 0.927 | 0.929 / 0.934 | 0.99 | 12 | 0.884 / 0.922 / 0.922 |

† RanPAC's ImageNet-R / ImageNet-A / CUB columns are on the ImageNet-1K-fine-tuned
checkpoint (see `docs/RANPAC_VS_US.md`); ours are in21k. "Head only" = RanPAC's
"No Phase 1" row.

**Readings.**

1. *The lever works, and it certifies.* Against the contracted prototypes, the
   contracted head changes strict accuracy by CIFAR-100 +0.040, CIFAR-100 +0.047, CIFAR-100 +0.036, CIFAR-100 +0.037, ImageNet-R +0.128, ImageNet-A -0.002, CUB-200 +0.009, ObjectNet-200 +0.025, OmniBenchmark +0.032, VTAB +0.042. Coverage stays at
   0.68–0.99 except on ImageNet-A, where the
   contract correctly refuses much of what the head would get wrong. On CIFAR
   the head lands within a point of RanPAC's head-only row at one fifth of its
   width; on VTAB it matches RanPAC's full result. The margin part of the
   contract is nearly vacuous (margin-only strict ≈ forced); the territory does
   the certifying.
2. *ImageNet-A is the exception, and it is informative.* With 2–86 training
   images per class the ridge head is worse than nearest-class-mean
   (0.480 forced against 0.511), and the contract exposes it: coverage
   0.68, strict 0.423. A head is a second-order estimator and needs
   the samples; the prototype topology degrades more gracefully.
3. *What the head gives up is the cascade's invariant.* Summed over the run,
   answers on already-seen test points changed 8 to 155 times per 100 test
   points (a point can change more than once), because the ridge weights are
   re-solved after every task.
   The cascade's count is exactly 0. The head's final weights are
   order-invariant, so its *end state* is reproducible, but no answer it gives
   today is guaranteed tomorrow. This is the trade the two topologies make:
   the head buys accuracy with the retention guarantee; the contract restores
   coverage semantics but not retention.

**Where this leaves the comparison.** Certified strict on CIFAR-ViT moves from
0.835 (prototypes) to 0.875 (head), against SimpleCIL 0.813, APER-adapter
0.875 and RanPAC 0.922; on DINOv2-B from 0.865 to 0.901. Every answer the
contracted head gives is inside a territory and above a margin; what it does
not have is bitwise retention. A head that never re-solves (per-task ridge
blocks frozen at birth, with the cascade's first-accept over them) would have
both, at whatever accuracy it costs; that is the next experiment.

## B-curriculum — scaffolded learning programs: order policies and containers-then-refinement (session 33)

*Two curricula, both a DIFFERENT PROTOCOL from random-order class-IL: their
numbers are compared only to the random-order lifecycle of the same seeds and
to the order-invariant contracted prototypes, never to the published anchors.
Evaluation seeds 2–4, frozen gate, no hyper-parameter changed. Code:
`cgrow/curriculum.py`, `exp_lifecycle.py --order-policy`, `exp_hier.py`;
families `B_order`, `B_hier`.*

**Item 1 — class order as a curriculum (flat protocol).** Difficulty of a class
is its nearest-class-mean error on the training split, computed before any
organ exists (an oracle curriculum: it uses every class's labels up front and
asks whether order matters, not whether a stream can be ordered online).
Policies: `easy` (least confusable first), `hard`, `grouped` (confusable
classes share a task: CIFAR-100 superclasses two per task; greedy confusion
pairs elsewhere), `spread` (confusable pairs kept in different tasks).

| benchmark | metric | random | easy | hard | grouped | spread |
|---|---|---|---|---|---|---|
| CIFAR-100 ViT-B (10×10) | growth strict | 0.667 | 0.664 | 0.643 | 0.698 | 0.670 |
| CIFAR-100 ViT-B (10×10) | certified lifecycle | 0.794 | 0.788 | 0.794 | 0.799 | 0.786 |
| CIFAR-100 ViT-B (10×10) | vetoes over life | 250 | 226 | 227 | 159 | 264 |
| split-digits | growth strict | 0.877 | 0.866 | 0.858 | 0.842 | 0.855 |
| split-digits | certified lifecycle | 0.912 | 0.909 | 0.912 | 0.915 | 0.918 |
| split-digits | vetoes over life | 6 | 10 | 8 | 9 | 12 |
| split-MNIST | growth strict | 0.892 | 0.892 | 0.884 | 0.882 | 0.888 |
| split-MNIST | certified lifecycle | 0.893 | 0.895 | 0.884 | 0.882 | 0.888 |
| split-MNIST | vetoes over life | 22 | 22 | 21 | 19 | 21 |
| split-KMNIST | growth strict | 0.805 | 0.816 | 0.802 | 0.827 | 0.814 |
| split-KMNIST | certified lifecycle | 0.805 | 0.816 | 0.802 | 0.827 | 0.814 |
| split-KMNIST | vetoes over life | 21 | 23 | 21 | 21 | 21 |
| split-Fashion | growth strict | 0.799 | 0.789 | 0.757 | 0.740 | 0.783 |
| split-Fashion | certified lifecycle | 0.799 | 0.789 | 0.757 | 0.740 | 0.783 |
| split-Fashion | vetoes over life | 19 | 17 | 20 | 16 | 20 |

Contracted prototypes are unchanged under every policy by construction
(0.835 / 0.843 on CIFAR-ViT; class means do not depend on order), and so are
NCM and RanPAC's head; a curriculum can only move the cascade route. Mean
change against the random order: easy: growth -0.003 (range -0.011…+0.011), certified -0.001 (range -0.010…+0.011); hard: growth -0.019 (range -0.041…-0.003), certified -0.011 (range -0.041…+0.001); grouped: growth -0.010 (range -0.059…+0.031), certified -0.008 (range -0.059…+0.022); spread: growth -0.006 (range -0.022…+0.008), certified -0.003 (range -0.016…+0.008).

**Item 2 — containers first, refinement inside them (coarse-to-fine).** CIFAR-100
supplies the hierarchy. Stage A grows 20 container organs (the superclasses, in
four tasks of five) by need with vetoes, exactly as the flat protocol grows
class organs. Stage B refines two superclasses per task: a child organ over the
five fine classes is trained only on the points the container organism routes
to that superclass, so its territory is nested in the parent's by routing;
containers are never touched and children are frozen at birth. A query gets a
fine answer when the child accepts, the coarse label alone when it abstains
(a *graded* answer), or an abstention when no container accepts.

| configuration (CIFAR-100, ViT-B/16, seeds 2–4) | coarse strict | fine strict | fine forced | vetoes | fine flips | graded: fine / coarse-only / abstain | params |
|---|---|---|---|---|---|---|---|
| flat cascade, random order (reference) | — | 0.667 | — | 250 | 0 | all fine | 2.14M |
| flat cascade, superclass-grouped order (item 1) | — | 0.698 | — | 159 | 0 | all fine | 2.14M |
| **containers (taxonomy) → refinement** | 0.774 (cov 0.99) | **0.626** ± 0.015 | 0.696 | **30** | **0** | 0.79 / 0.20 (coarse acc 0.57) / 0.01 | 5.13M |
| containers (feature-clustered) → refinement | 0.735 | 0.598 | 0.667 | 30 | 0 | 0.76 / 0.21 / 0.02 | 5.13M |
| certified lifecycle, random order (reference) | — | 0.794 | — | | | | |
| contracted prototypes, any order (reference) | — | 0.835 | 0.843 | 0 | 0 | | |

**Three readings.**

1. *Containers cost accuracy, and the ceiling says why.* On these features a
   20-way nearest-mean over superclasses reaches 0.849 while the 100 fine means
   mapped to superclasses reach 0.911: a container that summarises five classes
   as one prototype loses 6 points at the coarse level, and the loss is
   unrecoverable downstream (hierarchical NCM, coarse then fine within, tops out
   at 0.777 against 0.843 flat). Feature-clustered containers are worse still
   (coarse ceiling 0.822, measured 0.735), and containers that keep five
   unnamed sub-modes recover the coarse level (0.891) but name fine classes at
   only 0.760 after refinement. The developmental caveat was right: on this
   substrate the superordinate level is not where learning should start, and no
   coarse-first scheme reaches the flat prototypes.
2. *What containers buy is not accuracy.* The container stage needs
   30 vetoes where the flat cascade needs about 250, the
   children need none (refinement is a partition, not an exclusion), fine
   answers already given never change across ten refinements (0 flips, checked),
   and 20% of inputs receive a certified coarse answer when the fine
   level declines. That coarse-only answer is right 57% of the time
   against 77% overall, so a child's abstention also flags where the
   container itself is unsure. These are the properties a scaffold is for.
3. *Grouping helps the cascade where containers do not.* Putting confusable
   classes in the same task (the information order of a taxonomy, without the
   container organ) is the one order policy that moved CIFAR growth
   (0.667 → 0.698) and the certified lifecycle
   (0.794 → 0.799); the organ born for a task then holds five-class
   sub-structure at full resolution instead of collapsing it. The right unit of
   scaffolding here is the *task composition*, not a coarse label.


**Follow-ups (same session).**

*Children as prototype organs, and two-step refinement.*

| CIFAR-100 ViT-B, seeds 2–4 | coarse strict | fine strict / forced | container vetoes | within-container vetoes | fine flips | children params |
|---|---|---|---|---|---|---|
| trained-trunk children (one step) | 0.774 | 0.626 / 0.696 | 30 | 0 | 0 (+0 within) | 4.27M |
| **prototype children** (closed form, one step) | 0.774 | 0.665 / 0.714 | 30 | 0 | 0 (+0 within) | 0.08M |
| trained-trunk children, **two-step** refinement (3 + 2) | 0.774 | 0.611 / 0.635 | 30 | 138 | 0 (+0 within) | 8.54M |

Prototype children (anchor = mean of the routed points per fine class, PCA
territory, thresholds solved on the routed validation points, no trunk) change
fine strict by +0.040 against trained-trunk children and cut the
children's parameters from 4.27M to 0.08M.
Two-step refinement (three fine classes, then two, inside each container) puts
the flat protocol's machinery to work within a container: 138 within-container
vetoes over 20 containers, 0 within-container growth flips, and fine strict
-0.015 against one-step children. Container-level numbers are identical
across the three rows because the containers are cached per seed.

*A curriculum of consolidation (flat CIFAR-ViT, random order).*

| schedule | final strict / forced | consolidations accepted | share answered by last C | organs at end | growth flips outside consolidation | params |
|---|---|---|---|---|---|---|
| consolidate at the end (adolescence) | 0.776 / 0.777 | 3/3 | 0.76 | 11 | 0 | 2.22M |
| consolidate after task 5 (mid-life) | 0.700 / 0.702 | 3/3 | 0.39 | 11 | 0 | 2.18M |
| **progressive**: after tasks 2, 4, 6, 8, 10 | 0.771 / 0.772 | 13/15 | 0.70 | 14 | 0 | 2.32M |

Consolidating progressively (every two tasks, each proposal folding the previous
consolidated organ and the organs grown since into a new one, every proposal
gated) ends at 0.771 against 0.776 for a single consolidation at the
end (-0.005) and 0.700 for the mid-life schedule; growth flips
outside consolidation events stayed at 0 on every schedule. Read per seed, progressive is not one-directional (−2.4, 0.0, +1.1 against
end-only): its value is that the organism is consolidated at every point of its
life at no final cost, and that two of the fifteen proposals were refused by the
gate and correctly so. Consolidating once mid-life and then growing behind the
consolidated organ is the one schedule that hurts, as B-midlife already showed:
the consolidated organ's union territory cannot take vetoes, so later classes
leak into it.

**Not claimed.** Neither curriculum is comparable to published class-IL
numbers; both use label information (superclasses, or an oracle difficulty)
that the standard stream does not provide. The certified prototype
configuration is unaffected by any of it.

## Engineering hygiene (session 30)

Three small pieces of infrastructure, each motivated by a failure this project
actually had.

- **Interpreter guard** (`cgrow/__init__.py`). Twice a background job silently
  ran under the Anaconda *base* interpreter instead of the project venv
  (different torch/sklearn/numpy) and produced numbers that were not comparable
  with the frozen protocol. Every experiment imports `cgrow`, so the package now
  refuses the conda base interpreter with a clear message (override:
  `CGROW_ALLOW_ANY_PYTHON=1`); it does not pin one venv path, so the GPU box's
  CUDA venv still works. Verified: the base interpreter is refused.
- **Results manifest** (`manifest.py` → `results/MANIFEST.json`): sha256, size
  and mtime for every result artefact (83 files), after several near-clobbers
  and one battery found running twice into the same output file.
- **Claim provenance**: `claims.py` now embeds `_provenance` — a hash of every
  results JSON it read — so each number in the paper traces to the exact file
  that produced it.

## Final comparative table vs published results (evaluation seeds, final config)

*Every "this work" number below is the **certified** configuration: the
territory-respecting arbiter under the repaired gate, uncertified answer
fraction 0.0000 across all 36 runs. Where the uncertified free arbiter scores
higher it is shown in brackets and is **not** claimed as the result. The table
was built before the contracted head and the first-session adapters (B-head,
B-phase1 and later); [BENCHMARKS.md](BENCHMARKS.md) carries those rows.*

| setting | best published (what it uses) | **this work, certified** (stores nothing) | reading |
|---|---|---|---|
| split-MNIST, exemplar-free | generative classifier **0.938** (one VAE per class); DGR **0.913** (trained generator) | **0.893** *(0.915 uncertified)* | −4.5 to the best exemplar-free and −2.0 to DGR. **The earlier +0.2 edge over DGR does not survive certification** and is retracted |
| split-MNIST, stored data | iCaRL **0.946** (2,000 raw images); GDumb ≈0.97 | 0.893 | −5.3 to a 2,000-image buffer, with zero storage |
| split-MNIST vs same-data replay | strengthened replay 0.871 (100 raw images/class) | **0.893** | **+2.2 over replay with nothing stored** — the comparison that survives certification |
| CIFAR-100, ViT-B/16-in21k | SimpleCIL **0.813** last (0.871 avg); APER-adapter **0.875** last (B0 Inc10, v1 Tab. 7; the 0.852 quoted before 2026-09-07 was the Inc5 column); **RanPAC 0.922** (trained random-projection head) | **0.835 strict / 0.844 forced** (contracted prototypes, certified); 0.794 certified lifecycle *(0.835 uncertified arbiter)* | **+2.2 over SimpleCIL-last** on the certified prototype number; −8.7 to RanPAC. The cascade-plus-arbiter route reaches only 0.794 once every answer must sit in a territory, so **the certified claim rests on the prototype topology** |
| CIFAR-100, DINOv2-B (GPU box, session 31) | none verified for this backbone; same-feature NCM 0.873, same-data replay 0.889, joint 0.891 | **0.865 strict / 0.873 forced** (contracted prototypes); **0.865 certified lifecycle** *(0.873 uncertified arbiter)* | best certified CIFAR-100 number; the cascade route now equals the prototype route; −2.4 to same-data replay with nothing stored |
| CIFAR-100, ViT-L/16-in21k (GPU box) | none verified; NCM 0.877, replay 0.892, joint 0.877 | **0.867 / 0.877 forced**; 0.861 certified lifecycle | contracted prototypes = NCM forced on every backbone tested: **the gap to RanPAC is the classifier head, not the features** |
| six-dataset PTM suite, ViT-B/16-in21k (B-ptm) | SimpleCIL last 0.489–0.867; APER-adapter 0.494–0.871; RanPAC 0.624–0.922 | contracted protos strict 0.454–0.888 / forced 0.511–0.893 (= NCM); certified lifecycle 0.400–0.886 | forced within ~1 pt of SimpleCIL-last on 5/6 datasets and +4.9 on VTAB; RanPAC head +4 to +23 beyond; certified lifecycle 0.2–10.6 under the prototypes, tracking the consolidated organ’s reach |
| CIFAR-100, self-trained backbone | FeTrIL ≈0.66 (ResNet trained on half the classes) | not comparable (frozen-R18 proxy: 0.470 certified; ceiling 0.669) | different regime, stated rather than compared |
| KMNIST / Fashion / letter(13) / satimage / robot / spoken digits / spatial sheet | none published for class-IL | 0.78–0.99 certified | anchored to same-data strengthened replay and joint bounds |
| vs same-data replay, all 12 | — | **ours ≥ replay on 5 of 12** (mnist, kmnist, fashion, letter, robot) | down from 6 of 12: certified digits 0.912 no longer clears replay's 0.933 |
| **certificate axis** | none exists in published continual learning | verified zero forgetting (bitwise, every run); conformal coverage; **0.0000 uncertified answers in the final configuration**; a gate that accepts **0 regressions in 36 runs** and refuses all 4 known-harmful changes; drift SLA (5/54 violations vs 25% budget); replaceable parts to super-organ scale | **the novelty claim** |

**Novelty.** Accuracy-wise this work is competitive, not leading, and
certification widens the gap rather than narrowing it: **2–9 points** behind the
best specialized methods on the two settings where published numbers exist
(−4.5 to the generative classifier on split-MNIST, −8.7 to RanPAC on
CIFAR-ViT), having given up the +0.2 edge over DGR that an uncertified arbiter
had bought. What survives everywhere is the comparison against same-data
replay with nothing stored (+2.2 on split-MNIST) and the +2.2 over
SimpleCIL-last from the certified prototype topology. The novel contribution is the axis no published method
occupies: a learning system whose entire life — growth, inhibition,
consolidation, arbitration, even its own future drift — is forbidden, gated,
or logged, with the costs of each guarantee measured rather than assumed.

**Impact.** Change-control for deployed learning systems (regulated updates
with provable non-regression on the covered region); privacy-constrained
continual learning (replay-level accuracy, zero stored data); modular ML
engineering (parts with specs, replaceable at leaf and super-organ scale).

**Future work, ranked by evidence gained per effort.** (1) *Answered* (see
B-backbones): the certificate stack rides feature quality one-for-one
(DINOv2-B: 0.865 certified) but never above the nearest-class-mean topology it
wraps (forced = NCM to the third decimal on four backbones), so the RanPAC gap
is the classifier head. Open successor: a **contracted head** — a
random-projection or ridge readout whose outputs carry a ball contract — is the
named lever. (2) *Answered* (see B-daring): the
generalist is a participant, not a price index, and kmnist's residual loss is
explained by its near-zero routing headroom. Open successor: can a headroom
estimate computed from contracts alone — no arbiter, no real data — predict in
advance whether the market is worth deploying? (3) *Answered* (see B-ensemble): on a
twin anatomy the certificate retires exactly half the organs at a cost of
0.003, so the grown anatomy really is minimal. The successor is
answered in the same section: a territory-respecting arbiter keeps 72% of the
gain with zero uncertified answers, and composes with redundancy. *Answered*: it does not — on
CIFAR-ViT the gated arbiter keeps only 32% and 94% of the free arbiter's gain is
uncertified, but contracted prototypes deliver the same accuracy fully
certified. Open next: is territory quality measurable in advance (the
re-route-inside-territory rate, 0.69 digits vs 0.18 CIFAR, predicts which regime
you are in) so the topology choice can be made from contracts alone? (4) Contrastive territories — the named fix
for pixel-space sampler texture. (5) LoRA-organs on a frozen LLM — the same
architecture where change-control has customers. (6) Task-free streams.

## B1 — static tasks (3 seeds each)

| dataset | organism strict / forced | single organ strict / forced | plain MLP | certAcc (org / single / plain) |
|---|---|---|---|---|
| digits | 0.938 / 0.966 | 0.947 / 0.965 | 0.935 | 0.52 / 0.72 / **0** |
| letter10 | 0.759 / 0.825 | 0.791 / 0.851 | 0.785 | 0.58 / 0.62 / **0** |
| letter (C=26) | 0.840 / 0.849 | 0.868 / 0.880 | 0.755 | 0.10 / 0.07 / **0** |

Two findings. **The 26-class wall is cleared**: both contract methods reach
0.84–0.88 on letter where the parents' box contracts got 0.32, frozen balls 0.54,
and the plain probe 0.75. **On static tasks, growth is not an accuracy technique**:
the organism ties or slightly trails a capacity-matched single organ — exactly
ASDbrain's own conclusion. Growth's value is the developmental setting (B2,
B-sheet), the anytime property (B0), and replaceability (below).

## B-replace — the replaceable organ (3 seeds)

Lesion organ 1 of a five-organ digits organism; re-derive it from its **published
contract alone** (the PCA proximity spec doubles as a data generator; zero data
access), at three different architectures.

| replacement | agreement on the organ's region | agreement elsewhere |
|---|---|---|
| same width | 0.922 ± 0.043 | 0.991 ± 0.002 |
| wider | 0.922 ± 0.048 | — |
| deeper | 0.932 ± 0.022 | — |
| **swapped-spec control** | **0.000 ± 0.000** (inverts, as it must) | |

(The earlier centroid-sampler version scored 0.95 ± 0.02.) The organism's anatomy
is not just lesionable (ASDbrain) — it is **replaceable** (NNcontracts), and a
differently shaped implementation built against the same spec slots in at 92–93%
behavioral agreement. The ~1% drift elsewhere is acceptance-boundary flutter from
re-solved thresholds, reported rather than rounded away.

## What is genuinely new here

1. **A certified need field.** Growth is driven by abstention — the same object
   that carries the guarantee tells the system where to grow.
2. **Retention as a checked invariant.** Zero forgetting under growth by
   construction *and* verified bitwise per run; no regularizer, no replay buffer,
   no stored raw data.
3. **Growth reopened.** NNcontracts closed growth-by-need because errors were
   unstructured in the frozen representation. Growing a *fresh representation*
   behind a contract works — and control A shows the closed mechanism failing in
   the same harness.
4. **The contract as its own negative sampler.** §3.7.2's "a contract is a
   sufficient specification, therefore a generator" turns out to be the active
   ingredient in continual learning, not just module replacement — it is what
   closed the replay gap.
5. **Anatomy that is replaceable and causal.** The lesion question and the
   replacement question get the same answer from the same finite spec; on the
   spatial substrate the dependency structure is lower-triangular by theorem and
   forward transfer is priced in measured fragility.

## What is not claimed

- **No static accuracy win from growth** (B1: single organ ties it).
- The joint upper bound is still ~8 points ahead on the MLP ladder: frozen residual
  errors (false accepts 0.00–0.09 after vetoes) and veto self-loss are the measured
  price of immutability.
- The MLP-ladder organism carries ~5× the baselines' parameters (frozen trunks
  accumulate; on the sheet the organism is *smaller* than the monolith), plus one
  small veto net per organ, class territory, and tightening event.
- Vetoes require the PCA proximity geometry (the whitened-subspace construction is
  load-bearing); tightening re-routes on MNIST are slightly net-negative
  (54 wrong→right vs 118 right→wrong, ~1% of test) — a small, reported blemish.
- The sheet streams are easy (joint saturates); the sheet result is parity with
  replay plus anatomy, not superiority.
- **Certified robustness is inherited, not improved**: certAcc comes from the ball
  geometry (parent §6.5); growth does not increase it, and on letter it is near
  zero at the dimension-unfair ε, as the parent found. The recurrent sheet reports
  a Lipschitz bound but no certified-radius claims.
- **Scale**: MLPs and a ≤160-unit recurrent sheet, ≤784 dimensions, 10–26 classes,
  2–3 seeds on large runs. The baselines are our implementations at matched size;
  published class-IL numbers are quoted for placement only.

## Reproduce

```
python exp_toy.py
python exp_continual.py --dataset digits|mnist|kmnist|fashion --seeds 3
python exp_static.py --datasets digits letter letter10 --seeds 3
python exp_replace.py --dataset digits --seeds 3
python exp_ood.py --datasets mnist kmnist fashion --seeds 2   # geometry shoot-out
python exp_veto.py --datasets mnist kmnist fashion --seeds 2  # veto measurement
python exp_ablate_tighten.py --seeds 2                        # tightening ablation
python exp_sheet.py --streams sweeps shapes --seeds 3         # spatial substrate
python exp_adolescence.py --seeds 2                           # verified metamorphosis
python exp_renegotiate.py --seeds 2                           # fixed vs renegotiated contracts
python exp_sheet_audit.py --seeds 2                           # per-organ refinement audit
python extract_cifar_feats.py                                 # one-time (downloads)
python exp_cifar.py --seeds 2 && python exp_cifar.py --only-cncm --seeds 2
python extract_cifar_vit.py                                   # one-time (ViT-B/16 in21k)
python exp_cifar.py --feats results/cifar100_vitb.npz --tag vitb --seeds 2
python exp_cifar.py --only-cncm --feats results/cifar100_vitb.npz --tag vitb --seeds 2
python exp_drift.py --tag r18 --seeds 2                       # audited drift (both backbones)
python exp_drift.py --feats results/cifar100_vitb.npz --tag vitb --seeds 2
python exp_cifar.py --adolescence --seeds 2                   # gate decisions on CIFAR
python claims.py        # -> results/claims.json
```

(These are the development-era commands, which start at seed 0; the evaluation
runs use `--first-seed 2 --seeds 3`, and each later section names the script
and flags behind it. Environment: `requirements.txt`. The OpenML datasets are
downloaded once and cached; the sheet streams need ASDbrain's generators (see
README.md); the feature caches come from `gpu/`.)
