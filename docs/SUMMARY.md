# A neural network that can prove it hasn't forgotten

*Superseded, kept for the record. This dissemination summary was written before the rigor pass, and several of its headline claims were later retracted on stronger baselines: every "beats joint" claim, the arbiter figures (the free arbiter was afterwards found to buy most of its gain by answering outside every contract, and is not used), the split-MNIST number (0.92 here is the uncertified variant; the certified figure is 0.893), and the CIFAR-100 lifecycle 0.83 (the certified value is 0.795). For the current plain-English account read [APPROACH.md](../APPROACH.md) Part 1; for the record with its retractions, [REPORT.md](../REPORT.md).*

*Plain-English summary of the ASDplusContracts project, for dissemination.
Full results: REPORT.md; paper: paper/main.pdf; code walkthrough: TECHNICAL.md.
Every number regenerates from raw records via `claims.py`.*

## The problem

Neural networks learn by rewriting themselves — and rewriting is
indiscriminate. Teach a trained network something new and it silently damages
what it knew. The field of "continual learning" fights this with tricks
(penalties on important weights, or saving old examples to mix back in), and
scores them all on one number: accuracy. But no method can *prove* anything
about what survived. For a medical device, a bank, or any regulated system,
"we updated the model and probably nothing broke" is not an acceptable
sentence.

## The idea

Two earlier projects supplied the halves. One showed that in growing networks
it is *training itself* — touching weights — that destroys prior ability, so
the way to keep something is never to touch it. The other showed that a piece
of a network can carry a **contract**: a few numbers that fully describe what
it does, precisely enough that the piece can be rebuilt from the contract
alone, with no data.

Put together: grow a network as an **organism of frozen parts**. Each part
("organ") learns one task and is then frozen forever, behind a contract that
says which inputs are its territory and how confident it must be. Parts take
turns: the first one whose contract accepts an input answers it; if none
accepts, the system honestly says "I don't know" — and that refusal is
exactly where the next part gets grown. Because frozen parts cannot change,
the system's past answers cannot change: **forgetting is impossible by
construction, and we verify it anyway** — comparing every answer bit-by-bit
before and after every change, across twelve benchmarks, three kinds of
network, and every random seed. The count of silently changed answers, across
the entire study: zero.

Three further inventions make this practical:

1. **The contract is a data generator.** Since a contract describes a part's
   territory, you can *sample* from it — fake but faithful data, no storage of
   anything real. This one property powers everything else: **vetoes** (small
   learned inhibitors that stop an old part from wrongly claiming new
   material — the single biggest accuracy lever we found), **replacement**
   (delete a part, rebuild it from its contract at 92–96% identical
   behavior), and the lifecycle below.

2. **Adolescence.** Like a brain, the organism can undergo one supervised
   reorganization: all contracts generate data, one leaner consolidated
   network is trained on it, old wiring is rebuilt from scratch (on the
   spatially embedded version, with 60% fewer and shorter connections) — and
   a safety gate *verifies the swap before allowing it*. Across the study the
   gate made fourteen decisions; it refused exactly when the swap would have
   hurt.

3. **A prefrontal cortex.** In brains, sensory areas consolidate first and
   the control areas mature last, learning to arbitrate over already-fixed
   parts. We built that: an **arbiter** that answers nothing itself but
   learns which part to trust for each input — trained, again, purely on
   contract-generated data — and is itself verified and frozen last. It was
   accepted on ten of twelve benchmarks and improved every single one it
   joined.

## The results, in three sentences

Learning classes a few at a time with **zero stored data**, the organism
matches or beats methods that store thousands of raw examples on seven of
twelve benchmarks (split-MNIST 0.92 vs replay's 0.89; on a 13-task stream it
even beats a network trained on everything at once), and reaches 0.84 on
CIFAR-100 with a modern pretrained backbone — level with the standard
published method on the same backbone. The strict guarantee has a price that
depends on the data — from zero on easy tasks to sixteen points on hard ones —
and the full "grow → consolidate → arbitrate" lifecycle buys nearly all of it
back (CIFAR-100: 0.66 → 0.78 → 0.83). Everything that failed is reported with
its mechanism: the strict design loses on weak features, one safety gate was
fooled once (on data whose contracts were unfaithful — and we report that
rather than hide it), and contract territories can never be re-derived from
generated data (the attempt always produces a contract that is technically
true and does nothing).

## Why it might matter

Not because it wins accuracy contests — the best specialized methods are
still a few points ahead. Because it demonstrates a property nothing else
has: **a learning system whose entire life is change-controlled.** Every
modification is either impossible, verified before happening, or logged with
a before/after audit — and the system can *forecast its own drift* before
committing to a change. That is the shape of machine learning that
regulators, safety cases, and long-lived deployed systems actually need, and
this project shows it costs far less accuracy than one would guess — provided
you give the system a lifecycle instead of a single frozen design.
