# Scaffolded learning programs for a contracted organism

*Topic write-up: the scaffolding arc (class-order curricula, containers then refinement, progressive consolidation). A different protocol from the benchmark tables -- it uses label information the standard stream does not give. The live numbers are the `B-curriculum`, `B-hier` and `B-progressive` sections of [REPORT.md](../REPORT.md) and section D of [BENCHMARKS.md](../BENCHMARKS.md).*

*Curricula in the developmental sense: tasks of increasing difficulty, and
basic concepts ("containers") before their refinement. Everything here is on
evaluation seeds 2–4 with the frozen gate and no hyper-parameter changed. Both
curricula are a **different protocol** from random-order class-incremental
learning: they use information the standard stream does not provide (a label
hierarchy, or an oracle difficulty computed from all labels up front), so
nothing on this page is compared to the published anchors in `BENCHMARKS.md`.
Numbers regenerate from `results/claims.json` (`B_order`, `B_hier`,
`B_progressive`). Code: `cgrow/curriculum.py`, `exp_lifecycle.py
--order-policy`, `exp_hier.py`, `exp_progressive.py`. Detailed prose:
`REPORT.md`, section B-curriculum.*

## The question

Human development scaffolds learning: simple before hard, containers before
distinctions. Does the organism benefit from the same? Three answers were
possible in advance, and the experiments separate them: a curriculum could
raise accuracy, it could buy the certificate properties (fewer inhibitions,
stability, graded answers) without raising accuracy, or it could do nothing.

## 1. Class order as a curriculum (flat protocol)

Difficulty of a class = its nearest-class-mean error on the training split,
computed before any organ exists. Policies: `easy` (least confusable first),
`hard`, `grouped` (confusable classes share a task: CIFAR-100 superclasses two
per task, greedy confusion pairs elsewhere), `spread` (confusable pairs kept in
different tasks). Contracted prototypes are unchanged under any order by
construction (class means do not depend on order), and so are NCM and
RanPAC's head: a curriculum can only move the cascade route.

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

**Reading.** On CIFAR-ViT, grouping confusable classes into the same task is the
one policy that moves the cascade: growth 0.667 → 0.698, the
certified lifecycle 0.794 → 0.799, a third fewer vetoes;
the arbiter then had nothing left to route and was refused 3/3, as the
routing-headroom law predicts. Easy-first and hard-first do nothing useful. On
the MLP benchmarks no policy helps consistently: effects sit inside the ±2 point
order band already measured, except Fashion (grouped −5.9, hard −4.1) and
KMNIST (grouped +2.2).

## 2. Containers first, refinement inside them (coarse-to-fine)

CIFAR-100 supplies the hierarchy (20 superclasses × 5 fine classes). Stage A
grows 20 container organs (four tasks of five superclasses) by need with
vetoes, exactly as the flat protocol grows class organs. Stage B refines two
superclasses per task: a child over the five fine classes is built only from
the points the container organism routes to that superclass, so its territory
is nested in the parent's by routing; containers are never touched and children
are frozen at birth. A query gets a fine answer when the child accepts, the
coarse label alone when it abstains (a *graded* answer), or an abstention when
no container accepts.

| configuration (CIFAR-100 ViT-B, seeds 2–4) | coarse strict | fine strict / forced | fine precision | vetoes (containers + within) | fine flips | children params |
|---|---|---|---|---|---|---|
| taxonomy containers → trained-trunk children | 0.774 | 0.626 ± 0.015 / 0.696 | 0.789 | 30 + 0 | 0 + 0 within | 4.27M |
| feature-clustered containers → trunk children | 0.735 | 0.598 ± 0.005 / 0.667 | 0.785 | 30 + 0 | 0 + 0 within | 4.27M |
| **taxonomy containers → prototype children** | 0.774 | 0.665 ± 0.020 / 0.714 | 0.806 | 30 + 0 | 0 + 0 within | 0.08M |
| taxonomy containers → trunk children, two-step (3 + 2) | 0.774 | 0.611 ± 0.022 / 0.635 | 0.763 | 30 + 138 | 0 + 0 within | 8.54M |
| taxonomy containers → prototype children, two-step (3 + 2) | 0.774 | 0.644 ± 0.017 / 0.677 | 0.776 | 30 + 140 | 0 + 0 within | 0.08M |
| flat cascade, random order (reference) | — | 0.667 / — | — | 250 | 0 | 2.14M total |
| contracted prototypes, any order (reference) | — | 0.835 / 0.843 | — | 0 | 0 | — |

**Reading.**

1. *Containers cost accuracy, and the ceiling says why.* A 20-way nearest mean
   over superclasses reaches 0.849 on these features while the 100 fine means
   mapped to superclasses reach 0.911: a container that summarises five classes
   as one prototype loses six points at the coarse level and nothing downstream
   recovers it (hierarchical nearest-mean ceiling 0.777 against 0.843 flat).
   Feature-clustered containers are worse (coarse ceiling 0.822), and containers
   that keep five unnamed sub-modes recover the coarse level (0.891) but name
   fine classes at only 0.760 after refinement. On this substrate the
   superordinate level is not where learning should start.
2. *What containers buy.* About 30 vetoes at the container stage where the flat
   cascade needs about 250; children need none when refined in one step
   (a partition, not an exclusion); no fine answer already given changes across
   ten refinements; and a fifth of inputs receive a certified coarse answer when
   the fine level declines. A child's abstention also flags where the container
   itself is unsure: those coarse-only answers are right 57% of the time
   against 77% overall.
3. *The child level is prototype territory.* Prototype children (anchor = mean
   of the routed points per fine class, PCA territory, thresholds solved on the
   routed validation points, no trunk) score 0.665 fine strict against
   0.626 for trained-trunk children, with 0.08M parameters instead of
   4.27M. This is the adolescence finding again, one level down.
4. *Refinement in steps composes with inhibition.* Two-step refinement (three
   fine classes, then two, inside each container) runs the flat protocol's
   machinery within a container: 138 within-container vetoes over 20
   containers, zero within-container flips, and fine strict 0.611
   against 0.626 one-step. With prototype organs in both steps: 0.644 fine strict, 140 within-container vetoes, 0 within flips.

## 3. A curriculum of consolidation (flat CIFAR-ViT, random order)

When should the organism consolidate what it has grown? Each proposal passes
the adolescence gate; a refusal leaves the anatomy as it is.

| schedule | final strict / forced | proposals accepted | share answered by last C | organs at end | growth flips outside consolidation |
|---|---|---|---|---|---|
| once, at the end (adolescence) | 0.776 ± 0.003 / 0.777 | 3/3 | 0.76 | 11 | 0 |
| once, after task 5, then grow | 0.700 ± 0.007 / 0.702 | 3/3 | 0.39 | 11 | 0 |
| **progressive**: after tasks 2, 4, 6, 8, 10 | 0.771 ± 0.017 / 0.772 | 13/15 | 0.70 | 14 | 0 |

**Reading.** Progressive consolidation is free but not better: per seed −2.4,
0.0 and +1.1 against one consolidation at the end, every accepted proposal
raising strict on the classes seen at the time by 2.7 to 6.1 points, two of
fifteen proposals refused. Its value is an organism that is consolidated at
every point of its life at no final cost. Consolidating once mid-life and then
growing behind the consolidated organ is the one schedule that hurts
(0.700): the consolidated organ's union territory cannot take vetoes, so
later classes leak into it.

## What follows

Scaffolding pays in inhibition, stability and graded answers, not in accuracy;
the unit that helps accuracy is task composition (confusable classes together),
not a coarse label; and where a scaffold is used, its children should be
prototype organs. The accuracy lever remains the classifier head: see `B-head`
in `REPORT.md` for the contracted random-projection head.

*Published frames to audit before citing: task-ordering studies in continual
learning; the IIRC benchmark (superclass-first refinement).*
