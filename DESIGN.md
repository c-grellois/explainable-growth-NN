# Contracted growth: development behind certified interfaces

*Unification of ASDbrain ("Never Go Dense": developmental growth protects generality
and builds anatomy) and NNcontracts ("Interval contracts at neural network interfaces":
finite, checkable specifications make network parts replaceable).*

*This is the founding design, written 2026-08-20, and it is kept as written
except where the built system diverged from it — those places are marked. The
contract grew a third component (the input-space territory) and a fourth
(vetoes) that this document did not anticipate; §5 is the ladder as originally
planned, not as it was run. For what the system actually is, read
[TECHNICAL.md](TECHNICAL.md); for what it scored, [REPORT.md](REPORT.md).*

---

## 1. The two parents, and each one's hole

**ASDbrain** established a mechanism: *training, not pruning, destroys what a network
was born with* (86–95% of retention loss precedes any edge removal), so trajectories
that touch fewer weights early conserve more generality. Its hybrid method never goes
dense and develops anatomy — spatial modules where "where is this function?" has a
lesionable answer. **Its hole:** everything is *measured*, nothing is *guaranteed*.
Retention is a battery score, anatomy is a modularity statistic, and a lesionable
module is still not a replaceable one — there is no specification to replace it
against.

**NNcontracts** established the complementary object: a contract — a finite, checkable
specification at an internal interface — sufficient to re-derive the module behind it
without data, with ball contracts (anchors + slack + Lipschitz constant) surviving 26
classes and beating IBP certified training on its own metric. **Its hole:** growth.
Growth-by-need was *closed on both substrates*, with the cause named: "growth needs
errors that remain **structured in the frozen representation**, and both training
objectives destroy exactly that structure." Structure fixed in advance trains cleanly;
structure left to the optimizer goes vacuous. The note ends with development off the
table.

Each hole is the other's theorem:

- ASDbrain's protective mechanism — *weights never touched are never overwritten* —
  is exactly what a contract needs to stay valid: freeze the parent and the certificate
  survives verbatim. Contracts turn ASDbrain's empirical retention (86–95%) into an
  **invariant** (exactly 100%, by construction).
- NNcontracts' closure of growth assumed growth means *adding anchors to one frozen
  trunk* — errors had to be structured in a representation trained for something else.
  ASDbrain's answer: don't grow inside the parent; grow a **new organ with its own
  representation**, trained only on what the parent cannot do, and never touch the
  parent. The residual needs to be learnable by a fresh trunk, not structured in the
  old one. That reopens growth.

And the two need signals are the same object seen twice. ASDbrain grows where the
*need field* (gradient norm) is high — a heuristic. A contract system knows *exactly*
where it fails: where it **abstains**. Abstention is a certified need field.

## 2. The unified system: a contracted organism

An **organism** is an ordered list of **organs** O₁ … O_k. Each organ is:

- a **trunk** φᵢ: a small MLP with spectrally-capped layers, so its Lipschitz constant
  Lᵢ is a known product of capped norms (a knob, not an estimate);
- a **ball contract**: anchors {a_c} in its feature space (one per class it serves),
  a radius cap ρᵢ and a margin slack τᵢ;
- an **acceptance rule**: on input x with distances d₁ ≤ d₂ to the two nearest
  anchors, accept iff `d₁ ≤ ρᵢ` (proximity: x is close to known territory in feature
  space) and `d₂ − d₁ > 2τᵢ` (margin: the answer is determined). Predict the nearest
  anchor's class. Certified input robustness comes for free: the prediction is
  invariant to any perturbation of norm < (d₂ − d₁)/(2Lᵢ).
  *Since built:* two further conjuncts. A **territory** in input space — per class, a
  mean, its principal axes and the spread along them, with the edge drawn at a
  quantile of the class's own examples — must also admit x (`d_in ≤ ρ_in`); this is
  the conjunct that defends against classes that do not exist yet, and it is
  deliberately never tuned against precision. And no **veto** (§3b) may fire. Both
  only ever shrink what an organ claims.

**Routing** is first-accept in birth order: the organism answers with the first organ
that accepts; if none accepts, it abstains. All parameters of an organ — trunk,
anchors, ρ, τ — are **frozen at birth**.

**Development loop** (growth by abstention):

1. **Birth of O₁**: train a small trunk prototypically (cross-entropy over
   −‖φ(x) − a_c‖²) on the initial data/task; anchors are the published spec.
2. **Need field**: collect the samples the organism currently declines (and, when
   labels are available, declines-or-gets-wrong on held-in data). This *is* the need
   field — certified, localized, and free.
3. **Growth**: train a fresh organ only on the need set (its own trunk, its own
   anchors). The parent organism is frozen throughout.
4. **Verify, then accept** (the contracts note's §7 prescription — solve, never
   penalize): the candidate's ρ and τ are *solved* against held-out data (largest
   acceptance region at a target precision), and the organ is appended only if the
   organism's verified score improves. Rejected organs are discarded.
5. Repeat until the abstention budget is met or growth stops being accepted.

## 3. The retention theorem

> **Theorem (frozen-cascade retention).** Under growth alone, for any input x that
> the organism F_k answers, F_{k+j}(x) = F_k(x) for every j ≥ 0: the routing prefix
> and every organ's accept decision and output depend only on parameters frozen at
> birth. Hence every certified property of F_k on its accepted region — answers,
> robust radii — is preserved **verbatim** under all future growth, and coverage is
> **monotone non-decreasing** in k.
>
> With tightening, the invariant is **no silent drift**: an organ's accepted set can
> only shrink (ρ, ρ_in down; τ up — monotone by assertion), so an answer can change
> only by falling through to a later organ, and only at an explicit, logged
> tightening event whose expected effect was verified against measured false
> accepts. Growth events still cause exactly zero answer changes — checked bitwise
> in every experiment, not trusted. Empirically (split-digits/MNIST) tightening
> re-routes are few and net-positive (wrong→right dominates right→wrong ~5–7:1).

ASDbrain measured that touching weights spends retention; the organism makes touching
impossible where it matters. Forgetting is not regularized toward zero (EWC), not
replayed away (rehearsal) — it is *exactly* zero on the covered region, checkable by
arithmetic, with no stored raw data (the spec per organ is anchors + two scalars).

**The price, named up front** (there is always an exchange rate — contracts §2.4):
immutability cuts both ways. An early organ's *errors* are as frozen as its answers; a
confident wrong answer can never be repaired by growth, only pre-empted by abstention.
The exchange rate is therefore carried by (ρ, τ): accept aggressively → fast coverage,
frozen mistakes; accept conservatively → more abstention, more organs, better eventual
accuracy. This is the same antitone tradeoff as the contracts note's assume/guarantee
Galois connection, moved from one interface to the whole developmental trajectory.

**The known hard case, planned for rather than discovered later:** in sequential
(class-incremental) settings, organ O₁ never saw the classes that arrive later. Its
false-accept rate on future classes is the quantity that decides everything, and the
proximity cap ρ is the defense (novel classes should land far from old anchors in a
prototypically trained feature space — measurable, not assumable). We measure
false-accept vs ρ explicitly before relying on it.

## 3b. Learned tightening: the contract as its own negative sampler

Scalar tightening (ρ, τ, ρ_in) is three knobs; the measured OOD frontier of every
*unsupervised* input geometry (centroids, PCA, autoencoder) saturates far above what
class-IL needs. The lever that breaks the frontier is supervision — and the organism
supplies its own: **a contract is a data generator** (contracts note §3.7.2), so both
directions of the false-accept problem close without storing raw data:

- **old → new** (an old organ accepting incoming classes): train a per-class-territory
  **veto** — a small capped net in the territory's *whitened PCA coordinates*, where
  contract-sampled positives are faithful *by construction* (z ~ N(0, I) is exactly
  the model the spec asserts) — against the real falsely-accepted incoming samples.
  Conjunctive gate: the accepted set can only shrink; retention invariant intact.
  Measured: 95–99% exclusion at 6–10% self-loss; the naive raw-pixel version is a
  vacuous certificate (rejects all real data) — the whitening is load-bearing.
- **new → old** (a newborn accepting old classes): sample negatives from the old
  organs' contracts. Each birth veto is **verified against the newborn's real val
  data** and discarded if it costs > 3% own coverage (solve/verify, never guess).

One second-order finding with teeth: once vetoes exist, scalar tightening is **net
harmful** — its self-loss releases own-class samples that downstream organs then
answer *wrongly* (correct → wrong via cascade capture), while a veto converts false
accepts to abstentions (wrong → abstain at worst). Tightening should be learned-only.

## 3c. Adolescence: the verified metamorphosis

Development's invariant (never touch a frozen part) is also its ceiling: frozen
errors and redundant machinery accumulate. Biology's answer is adolescence — a
phase of massive recombination of local connectivity after enough learning. The
organism can afford the same move because of what it already owns: **every organ's
contract is a data generator**, so the joint problem the cascade never solved can
be trained *data-free* — sample labeled data from all contracts, train one
consolidated organ on the pooled synthetic set, re-place its units by the wiring
economy, re-learn its local wiring from scratch under the hinge cost, prune
cost-weightedly. The consolidated organ's territory is the **union of the parts'
proximity contracts** (each calibrated on real data at birth — fitting proximity
on synthetic data instead is exactly the vacuity trap, measured: 0% real
acceptance on the sheet). Verification gates the swap (synthetic per-class
holdout + the last task's real val); the old cascade stays behind it as fallback,
so coverage cannot drop. Adolescence is the **single sanctioned exception** to the
no-drift invariant: answers may change, every change is logged and audited
(flips to-right vs to-wrong on real test), and freezing resumes afterwards.

## 4. What is genuinely new

1. **A certified need field.** Growth driven by abstention rather than gradient
   heuristics; the growth signal and the guarantee come from the same object.
2. **Retention as an invariant, not a score.** The first growth method whose
   forgetting is exactly zero by construction *and* whose zero is checkable.
3. **Growth reopened.** Verified organ addition where verified anchor addition
   accepted zero candidates — the test of the "fresh representation" explanation.
4. **Anatomy that is replaceable, not just lesionable.** Each organ sits behind a
   finite spec (anchors, ρ, τ, L); the replaceable-organ experiment (retrain an organ
   from its spec alone, swap it in, organism unchanged) unifies ASDbrain's
   lesion-defined zones with NNcontracts' module replacement.

## 5. Benchmark ladder *(the plan, 2026-08-20)*

*Kept as written. What was actually run went well past B3 — twelve benchmarks,
four feature substrates, seven pretrained-model datasets, streams cut into up
to a hundred updates — and is recorded battery by battery in
[REPORT.md](REPORT.md). Two items below did not survive contact: the "beats
joint" target of B2/B3 was withdrawn when the joint baselines were strengthened,
and the certified-robustness arm of B3 was never run.*

**B0 — toy (2-D, visual).** Gaussian-mixture rings/moons. Watch the organism grow;
verify monotone coverage and exact retention. Controls that must fail: (a) unverified
growth (accept every candidate) must degrade the parent — the analog of the note's
0.88 → 0.39 crash; (b) fine-tuning the parent on the need set must damage previously
covered samples (measured retention loss) where the organism's is exactly 0.

**B1 — static hard tasks.** digits → mnist → kmnist → fashion (difficulty ladder,
shape and C fixed), and letter (C = 26, the wall both parents hit). Question: does
growth-by-abstention beat a single prototypical organ at matched parameters, and how
does the gap to a plain MLP scale with difficulty? Reference points from the parents:
letter — boxes 0.32, frozen balls 0.54, prototypical 0.84, plain probe 0.75.

**B2 — sequential (class-incremental).** Split-digits and split-MNIST (5 tasks × 2
classes, single-head, no task oracle at test time — the brutal setting). One organ per
task. Compare: fine-tune (collapses), EWC (collapses in class-IL), replay (upper
band), ours. Report accuracy, *certified* forgetting (exactly 0 vs measured for
baselines), false-accept rate vs ρ, parameter growth.

**B3 — against state of the art.** (a) Class-IL split-MNIST against published
numbers: regularization methods ~20%, LwF ~23%, DGR ~91%, iCaRL ~94% (van de Ven &
Tolias 2019) — the target is replay-class accuracy with zero stored raw samples and a
certificate none of them have. (b) Certified robustness: certAcc@ε of the grown
organism vs the within-codebase IBP-trained baseline (which the prototypical parent
already dominated on mnist/fashion), with the honest caveat that published
convolutional IBP numbers are higher.

**Controls throughout** (the contracts note's discipline): every experiment carries a
control that must fail — swapped anchors must invert, unverified growth must crash,
and any "zero forgetting" claim is checked by bitwise comparison of stored
pre/post-growth predictions, not trusted from the theorem.

## 6. Methodological inheritances

- **Solve, don't penalize** (contracts §7): ρ, τ, and any gain are solved against the
  certificate in closed form or by direct search; no certificate term ever enters a
  loss.
- **Never go dense** (ASDbrain): no stage of the organism ever trains a monolith;
  peak "touched parameters" per growth event is one organ.
- **Per-seed reporting** (contracts §4.7): bistability hides in means; all runs
  report per-seed.
- **Machine-regenerated claims** (ASDbrain): a single `claims.py` regenerates every
  number in the report from per-seed records.
