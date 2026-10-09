# Growing a network that cannot forget

*The contracted-growth project (ASDbrain × NNcontracts), explained twice: first
in plain English, then with the details. Every number here is regenerated from
per-seed records by `claims.py`; the full evidence is in
[REPORT.md](REPORT.md), the comparison tables in [BENCHMARKS.md](BENCHMARKS.md),
and the paper draft in `paper/main.pdf` (v3.6).*

---

## Part 1 — In plain English

### The problem

Neural networks forget. Teach one to recognise cats and dogs, then teach it
birds, and it quietly loses the cats and dogs. The field calls this
catastrophic forgetting, and every existing cure is a *score*: methods promise
they forget less, and you find out how much they forgot by testing afterwards.
None of them can promise not to.

We wanted something stronger: a system where forgetting is not unlikely but
**impossible** — and where "impossible" is checked, not trusted.

### The idea

Instead of one big network that keeps being retrained, we grow a body of small
frozen parts, which we call **organs**.

- Each organ is trained once, on one slice of experience, and then **frozen
  forever**. Nothing that happens later can touch its weights.
- Each organ sits behind a **contract**: a short, checkable description of
  what inputs it claims (its *territory*) and how it behaves on them. The
  contract is small enough to write down, and it doubles as a generator — you
  can sample fake examples "in the organ's own words" from it.
- When new material arrives, the body doesn't retrain anyone. It notices where
  it currently **abstains** — inputs no organ claims — and grows a *new* organ
  exactly there.

Because old organs are never touched, an answered question keeps its answer.
That's not a hope; the code compares every answer bit-for-bit before and after
every growth event. Across every benchmark and every seed we ran — raw pixels,
frozen image and text encoders, raw tabular records, a recurrent spatial sheet
— the number of answers changed by growth is **exactly zero**.

### How an organ learns what it may answer

An organ answers an input only if it can say yes to three questions, and it
learns its own numbers for each of them at birth, from its own slice of
experience.

1. *Is this the kind of thing I was built from?* Every class the organ serves
   gets a description of that class's own examples: the average example, the
   couple of dozen directions along which those examples actually vary, and
   how far they spread. An input is scored by how far outside that cloud it
   falls. This is the organ's **territory**.
2. *Is it close enough to one of my classes?* Inside the organ's own feature
   space each class has an anchor, and the input must land within a **radius**
   of the nearest one.
3. *Am I sure which class?* The nearest anchor must be clearly nearer than the
   runner-up — a **margin**, not a photo finish.

Any no, and the organ abstains. (Later it may also acquire vetoes, which can
only subtract.)

The territory is fitted from each class's own examples alone, and its edge is
drawn where about 90% of them fall inside. The radius and the margin are then
solved on validation examples the body currently declines:

- the **radius** covers *my own correct answers and nothing more*: it is placed
  at the far quantile of the distances of the validation points the organ got
  right, which buys a finite-sample promise — a fresh correct example of its
  own kind is accepted with probability about 99%, as long as new data
  resembles the calibration data;
- the **margin** is swept over fourteen candidate values and set to the widest
  reach that still keeps precision above the target (90% at birth). Not the
  safest value: the most generous one compatible with the promise.

That asymmetry is deliberate. Precision measured today cannot see classes that
do not exist yet, so the two radii are never tuned against it — they are the
defence against future strangers, and are set from own data only. The margin
is the knob tuned for accuracy.

Then the organ is frozen, and three things are kept: the exact Lipschitz
constant of its small network — so every accepted answer can carry a certified
radius, meaning how far the input could move before the answer changes — and,
for each birth-time validation point, its distance, its margin, its territory
score and whether the organ got it right. That record is the organ's licence
for later: when it is asked to narrow itself, the cost is priced against its
own birth statistics, without consulting data again.

One side effect matters downstream. A territory is an average, some directions
and some spreads, so you can *draw* examples from it. That is how the body
rehearses old classes later without having stored a single one.

### When the body grows, and why growth cannot cost anything

Growth is driven by abstention. When new material arrives, the body first asks
itself which of the new examples it currently declines; those, and only those,
are what a candidate organ is trained on. The candidate then has to earn its
place:

1. it must find thresholds under which it is right about 90% of the time on
   the declined validation examples — if no setting reaches that, it is
   rejected;
2. among the examples it would then accept, right answers minus wrong answers
   must be positive — a candidate that would do net harm is rejected.

Rejections are logged with their reason (*nothing declined*, *no threshold
meets target*, *net gain −3*). Only survivors are frozen and appended.

**Appended** is the operative word. Answers are produced by going down the
organs in the order they were born, and the *first* one whose contract is
satisfied answers. A new organ sits at the end of that queue, so it is only
ever consulted on inputs that every older organ already declined. It cannot
overrule an older answer — not as a matter of training, but of structure. Two
consequences: answers already given stay identical (checked bit for bit; zero
changes across 501 additions, on bodies of up to 100 organs), and since a
declined input already counts as an error in our scoring, filling one can only
help. **Growth alone cannot lower the score.**

Two honest qualifications, both measured rather than assumed.

- **Narrowing can move answers.** When examples of genuinely new classes
  arrive, any old organ that accepts them is making a mistake, so the old
  organs narrow. Narrowing only ever turns "I answer" into "I abstain" —
  never into a different answer — but in a queue an abstention falls through
  to the next organ, which may answer differently. Those changes are counted
  separately from growth, and they are not always a gain: on CIFAR-100 about
  971 of 10,000 test answers move over a run, 240 from wrong to right against
  80 the other way, while on split-MNIST the same mechanism moved 65 to
  right and 138 to wrong.
- **Reorganisations can change everything**, which is why they are the ones
  put to a gate — the next section.

And what the system does not claim: more organs does not mean more accuracy.
Cutting the same stream into a hundred steps instead of ten gives a hundred
narrower organs and a lower final score (0.667 → 0.564 on CIFAR-100). The
guarantee is about what an update cannot undo, not about what it gains.

### Growing up: the lifecycle

A body of frozen parts has a problem of its own: frozen mistakes accumulate.
So the organism goes through a life, loosely modelled on how brains mature —
and every life event is either forbidden, gated, or logged.

1. **Inhibition (vetoes).** A young organ tends to claim things it has never
   seen. When new classes arrive, each old organ learns a small veto — "this
   looks like my territory but isn't" — trained using its own contract as the
   source of positives. Vetoes only ever *shrink* what an organ claims, so
   they can never cause forgetting.
2. **Adolescence.** Once enough organs exist, the body attempts a massive
   reorganisation: it samples training data *from the contracts themselves*
   (no stored data anywhere) and builds one consolidated organ. The attempt
   must pass a verification gate; if it fails, nothing changes.
3. **A prefrontal arbiter.** Late in life, a small judge learns *which organ
   to trust* for each input. Crucially, our judge is only allowed to pick
   among organs whose contracts actually claim the input. An unconstrained
   judge scores higher — but we measured where its extra accuracy comes from,
   and most of it comes from answering where **no contract applies at all**.
   That extra accuracy is real, but it isn't covered by any promise, so we
   don't use it. In the final system, **every single answer is covered by a
   contract** — measured, not assumed.
4. **The acceptance gate.** Every proposed life change (a consolidation, a
   judge, a boundary change, an organ's death) is admitted or refused by a
   gate. We audited our own gate, found two blind spots, repaired it, froze
   it, and then re-tested it on fresh random seeds it had never seen. On those
   fresh seeds it refused every harmful change it was capable of resolving,
   refused no clearly good one, and let through exactly one regression — of
   0.7 points, half the size the gate itself declares as its detection limit.

### What we found

**It works, and we know exactly what it costs.** With zero stored data, the
certified organism beats a replay method that keeps 100 real images per class
on five of the twelve small benchmarks (e.g. split-MNIST 89.3% vs 87.1%).
Insisting that every answer be contract-covered costs 1–8 points on most
benchmarks compared to the unconstrained version — and on the one benchmark
where the unconstrained version had quietly accepted a disastrous change, the
certified version is 11 points *better*. That trade — a little accuracy for
the guarantee that nothing bad gets in — is the whole point.

On the standard benchmark suite for pre-trained vision models (seven datasets,
a public encoder adapted once on the first task and then frozen), the system
comes in two configurations. The **accuracy configuration** puts the readout
the strongest published method (RanPAC) uses under the same contract: on
CIFAR-100 it reaches 90.6%, it is above every published variant of the
adaptive-tuning baseline (APER) on all seven datasets, and it is 1.6 points
below RanPAC there, which makes no promises about retention.
But it re-solves that readout at every step, so over a run it changes between
7 and 105 of its earlier answers per 100 test points (some answers change more
than once). The **stable configuration**
(class prototypes behind contracts) changes none, and scores 86.7% on
CIFAR-100. Getting both at once was tried and costs 20 to 54 points, for a
structural reason: a readout frozen at birth has never seen the classes that
come later. So retention and accuracy are offered as two configurations, each
with its price stated.

**Some daring ideas worked.** Organs can sleep (nightly consolidation matches
one-shot adolescence, if the organism also "dreams" — rehearses its judge).
The organism can publish a service-level agreement on its own future drift and
keep it (5 violations in 54 events against a budget of 25%). A whole organism
can be wrapped up to act as a single organ inside a bigger one, and the
no-forgetting guarantee survives the nesting.

**Some failed, and the failures became laws.**
- *Organ death:* four increasingly careful "redundancy certificates" kept
  approving deaths that quietly hurt individual classes. A per-class
  certificate stops all of it — by retiring nobody. The anatomy that growth
  builds contains no redundant organ; when we built an artificially redundant
  body (every class covered twice), the same certificate correctly retired
  exactly one of each pair.
- *Markets:* letting organs bid to answer queries fails if each organ prices
  itself (it has never seen its own mistakes), works if prices are calibrated
  on a shared pool, and the auction needs a generalist *participant* — not a
  price index. Used at contract-writing time instead of answer time, the
  market genuinely helps where the fallback routing is good, and provably
  cannot help where it isn't: removing a claim from one organ cannot *direct*
  it to the right one.
- *The unifying law:* how much any judge, market or reorganisation can win is
  bounded by the **routing headroom** a body leaves — and our gate's
  accept/refuse decisions turn out to reproduce that law without having been
  told it.

**The honest limits.** Nothing here beats a properly trained ordinary network
that sees all the data at once. The certificates are only as good as the
contracts' fidelity — we exhibit one benchmark where synthetic samples
misrepresent old tasks and only real held-out data catches the damage. The
results ride on frozen public encoders: the system adds small parts on top of
them and inherits their blind spots. And scale so far is modest: at most 300
classes, and everything after feature extraction runs on a CPU.

### Why it matters

Machine learning is scored on accuracy. Deployed systems are also governed by
**change control**: what changed, who approved it, what got provably worse.
This project is a working example of a learning system that lives under change
control — every weight either frozen or gated, every answer covered by a
written promise, every guarantee priced rather than assumed. That axis —
auditable, refusable change — is one no published continual-learning method
currently occupies.

---

## Part 2 — The same story, with the details

### The parts

**Organ.** A small prototypical trunk with spectrally capped layers (a power
iteration keeps each layer's Lipschitz constant below a cap, so evidence is
bounded). It answers with nearest-anchor classification in its feature space.

**Contract.** Per-class feature anchors with margin slack τ and radius ρ, plus
an input-space **territory**: per-class PCA subspaces (means, axes,
eigenvalues, calibration scalars — prototypes, never raw samples) with a cap
ρ_in. An organ *accepts* an input only inside its territory with a determined
margin; accepted answers carry a certified robust radius by interval
arithmetic. All thresholds are **solved, never penalized** — set by
split-conformal calibration on held-out scores, giving finite-sample coverage
under exchangeability. The territory doubles as a generator: sampling
z ~ N(0, I) in the whitened PCA coordinates produces labelled synthetic data
that is exactly what the spec asserts.

**Organism.** A first-accept cascade of frozen organs. Its abstention region
is the *need field*: growth trains a fresh organ only on declined inputs, and
a candidate joins only if it verifiably improves declined-validation accuracy.
Retention under growth is a theorem (the routing prefix depends only on frozen
parameters) and is checked bitwise anyway — 0 flips, every run, everywhere.

**Vetoes (learned inhibition).** Every unsupervised OOD geometry we tested
(centroids, PCA distance, autoencoder, combinations, a corruption
discriminator) saturates at the same false-accept frontier. The working lever
is supervision the organism already owns: train a small capped net on
contract-sampled positives *in whitened territory coordinates* against the
real falsely-accepted samples of the incoming task, and conjoin it with the
accept rule. Exclusion 0.955–0.986 at 6–10% self-loss. Two structural facts:
the raw-space version is a vacuous certificate (it learns "synthetic vs real"
and rejects everything), and scalar threshold-tightening becomes net *harmful*
once vetoes exist (ablation: none 0.408 / scalar 0.742 / veto-only 0.897 /
both 0.819 on split-MNIST).

### The lifecycle, certified

**Adolescence.** Pool contract samples from all organs, build one consolidated
organ, verify, swap-or-revert. At high class counts the consolidated organ
must be **closed-form prototypes** (a trained trunk collapses on a 100-class
pool). It inherits the *merged* territory — a union of PCA territories is a
PCA territory — which keeps it veto-capable; that single property turns
mid-life consolidation from harmful to net-positive. Guarantees can be
renegotiated at adolescence (certified backward-compatible by sampling);
territories cannot be re-derived from generated data — every attempt is
quietly vacuous.

**The arbiter, and why it is territory-respecting.** A small net reads each
organ's evidence vector (distance, margin, territory score, accept bit) and
picks who answers. The exact, additive decomposition of the *unconstrained*
arbiter's gain (fill / re-route / drop partitions summing to Δ) shows: on
digits its gain is genuine re-routing (134 corrections vs 4 regressions), but
8.6% of its answers come from a non-accepting organ and carry 45% of the
advantage; on CIFAR-ViT, outside-territory answers carry **94%** of the
advantage and only 18.3% of its re-routes land inside any territory. It has
stopped being a contracted system. The lifecycle therefore restricts the
arbiter to accepting organs (abstain if none). Measured price: keeps 72% of
the unconstrained gain on digits, 32% on CIFAR-ViT — where the
regime-appropriate answer is topological (contracted prototypes: same
machinery, prototype routing, certified by construction).

**The acceptance gate.** Three checks; a candidate passes only if all pass,
and the verdict names what fired.

1. *Territory* — data-free and exact: every answer must come from an accepting
   organ. It is a **policy, not a detector** (audited: it refuses all 21
   historical unconstrained arbiters, helpful and harmful alike).
2. *Escrow* — the whole held-out validation split, used only by gates, judged
   by a one-sided McNemar test on discordant pairs (z ≥ −1.645), per-class
   flags only past a Bonferroni correction, plus an unconditional 0.25 hard
   floor for single-class collapse. Three failed designs are kept on record
   as named traps: fixed per-class floors (multiple comparisons), a
   train-drawn escrow (flatters every candidate), and per-class subsampling
   (discarded 65% of a small dataset's validation and turned a z = −6.86
   refusal into an accept).
3. *Contract* — the original per-class check on contract samples, retained to
   show what it misses: contract-sample gates certify contract **fidelity**,
   not real behaviour.

*Disclosure and confirmation.* The gate alone was iterated on evaluation
seeds, so its spec was frozen and re-run on seeds never used for any decision:
30 fresh runs, uncertified fraction 0.0000, the known-harmful arbiter refused
3/3 (z = −3.35/−6.34/−7.72), zero clearly-helpful refusals, one accepted
regression of −0.0068 — half the gate's stated resolution (`detectable_drop` =
0.0137 for that escrow). The frozen gate admits no regression above its stated
resolution, refuses every harmful change it can resolve, and refuses no
clearly-helpful one.

### The certified results

| benchmark | growth | + adolescence | **+ certified arbiter** | unconstrained (uncovered) |
|---|---|---|---|---|
| split-digits | 0.877 | 0.888 | **0.912** | 0.964 |
| split-MNIST | 0.892 | — | **0.893** | 0.915 |
| split-KMNIST | 0.805 | — | — | 0.805 |
| split-Fashion | 0.799 | — | — | 0.797 |
| letter (13 tasks) | 0.782 | — | — | 0.788 |
| satimage | 0.832 | — | **0.833** | 0.861 |
| wall-robot | 0.800 | — | — (harmful judge refused) | 0.685 |
| CIFAR-100, frozen R18 | 0.395 | — | **0.470** | 0.483 |
| CIFAR-100, ViT-B/16 | 0.667 | 0.776 | **0.794** | 0.835 |
| sheet: sweeps / shapes | 0.979 / 0.985 | 0.989 / — | — | 1.000 |
| spoken digits (sheet) | 0.772 | — | **0.790** | 0.872 |

Every answer contract-covered (uncertified fraction 0.0000 in all 36 runs);
zero accepted regressions; refusals leave the organism unchanged.

**Against published results** (all ours certified; anchors are final
accuracies where we report final): split-MNIST 0.893 vs the generative
classifier's 0.938 and DGR's 0.913 (both exemplar-free; our earlier edge over
DGR came from the unconstrained arbiter and is retracted), vs iCaRL's 0.946
with 2,000 stored images, and vs same-data strengthened replay's 0.871 — the
comparison that survives certification, +2.2 with nothing stored. CIFAR-100
at ViT-B/16-in21k: contracted prototypes 0.835 strict / 0.844 forced, i.e.
+2.2 over SimpleCIL-last (0.813) and −8.7 to RanPAC (0.922). The unique row is
the certificate axis: verified zero forgetting, conformal coverage, a
confirmed gate, drift SLA (5/54 vs a 25% budget), replaceable parts up to
whole nested organisms.

### The laws

- **Routing headroom** (arbiter − cascade) bounds what any router — judge,
  market, reorganisation — can win. It is benchmark-specific (+0.001 KMNIST,
  +0.082 digits, +0.062 CIFAR-ViT), and the part that matters for a certified
  system is the *contract-covered* share of it. The gate's verdicts reproduce
  this law unprompted, on evaluation and confirmation seeds alike.
- **Duplicate organs buy coverage, never precision.** Concatenating
  independently grown twins raises coverage (0.934 → 0.981) at flat precision,
  zero flips — first-accept's prefix wins every contest, so a twin can only
  fill abstentions. Width is not a substitute; redundancy and the
  unconstrained arbiter are substitutes; redundancy and *certified*
  arbitration are complements exactly where filling was the forbidden channel.
- **Selection cannot be replicated by monotone boundary edits.** Renegotiating
  territories by auction works on digits (market = oracle, +0.0148, fully
  certified, no inference-time trained parts) and fails on CIFAR even for an
  oracle (−0.0045): a veto removes a claim but cannot direct it, and
  first-accept hands the ceded point to birth order, not to the winner. The
  *fallback ratio* (moved-right : moved-wrong) predicts the outcome in advance
  from the contract pool alone.
- **Prices need joint data, not a central model.** Self-priced routing is
  anti-calibrated (an organ has never seen its own negatives); the same tiny
  per-organ price curves calibrated on the shared contract pool cure
  calibration by up to 200× and, after consolidation, tie the central arbiter
  on the flagship. The auction needs a generalist *participant* — a
  gate-refused generalist scoring 0.187 alone still adds +0.09 as a bidder —
  and indexing the reserve to its bid adds nothing.

### What failed, honestly

Weak features break the cascade (frozen R18, 100 classes: prototypes win; the
contract machinery ports, the cascade doesn't). Contract-sample gates certify
fidelity, not behaviour — wall-robot's judge passed every synthetic signal and
regressed 12 points on test; only the real-data escrow catches it. Territories
refitted from generated data are always quietly vacuous. Pixel-space samplers
are trivially discriminable from real data (the gates correctly refuse
consolidation there). Organ death is the most expensive certificate in the
lifecycle: per-class real data or nothing. The unconstrained arbiter is not
mis-gated but *uncertifiable*. And nothing beats a strengthened joint model.

### Since the certified results: the pre-trained-model suite and beyond

The table above is the cascade on its home ground. Later batteries took the
same machinery to the standard pre-trained-model suite (seven datasets on a
frozen ViT-B/16) and off vision; [BENCHMARKS.md](BENCHMARKS.md) has every row.

- **Two configurations, not a slider.** Contracted prototypes — class means
  with territories — equal nearest-class-mean forced accuracy to the third
  decimal on every encoder tried, are order-invariant, and change no answer
  when classes arrive. The **contracted head** (a random-projection ridge
  readout under the same contract) is the accuracy configuration, and it
  re-solves its weights each step. A head made of per-task blocks frozen at
  birth keeps every answer and loses 20–54 points (`B-head-blocks`).
- **A birth step for the substrate.** Adapting the encoder once, on the first
  task only, lifts the head above every published APER variant on all seven
  datasets; the whole table replicates on untouched seeds 5–7 within 1.4
  points. On one dataset the adaptation makes the features slightly worse, and
  the downstream consolidation gate refuses on all three seeds.
- **Off vision, unchanged.** Two intent-classification streams on a frozen
  sentence encoder and raw intrusion-detection records: no code or setting
  changed, growth flips 0, and the head matches or beats same-data replay.
- **Declining, measured.** The certificate's slack works as an
  out-of-distribution score — above kNN, nearest-mean and Mahalanobis scores
  on the same features — but it says where an answer came from, not how far
  outside an input is: at its own operating point the contract still accepts
  most of a near-distribution set.
- **Size.** A grown split-MNIST organism is about 5.5× the single network the
  baselines train. Pruned at birth, before its contract is solved, each trunk
  keeps its accuracy to 99% sparsity; growth flips stay 0 at every sparsity.

### Reproducing

`cgrow/` holds the library (organ, organism, proximity geometries,
adolescence, gates); `exp_*.py` are the experiments; every quantitative claim
regenerates via `claims.py → results/claims.json` (81 claim families).
Development used seeds 0–1, evaluation seeds 2–4, gate confirmation seeds 5–7.
Install with `pip install -r requirements.txt`; feature extraction for the
pre-trained encoders runs on a GPU through the scripts in `gpu/`, and
[README.md](README.md) has the quick start.
