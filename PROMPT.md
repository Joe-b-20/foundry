# foundry — mission

Algorithm Foundry is a domain-pluggable search engine that proposes candidate
algorithms, executes them in a cost-metered language, verifies correctness,
attacks its own survivors, compares against known references, diagnoses
search walls, and records every action as training data for future
automation.

The goal: find better, faster, more efficient algorithms — in any and all
domains. Or, at minimum, be the project that honestly tries.

## Where foundry comes from

Parent project: mathlab (public as `rediscovery-engine`), frozen and archived
in full under `reference/mathlab/`. It proved the concept: discovery-from-
outcome works up to a *recognition ceiling*; search walls have kinds (the
wall taxonomy in the parent docs is the doctor's checklist); and the parent's
final "Foundry v1/v2" prototype ran a real mutate → verify → literature-check
loop on polynomial continued fractions. foundry is that prototype
generalized: the engine is the product, discoveries are its output stream.
PCFs return later as a regular domain pack.

## What we refuse to do

We never win on compute. We do not re-run DeepMind / OpenAI / big-academic
searches with less hardware. Our edges:

- weirder domains
- stranger representations (molds are first-class and, eventually, searchable)
- faster iteration
- more ruthless verification (exact checks, attack loops, certificates)
- less institutional taste
- willingness to keep failures (they are map, and training data)
- single-person coherence

A genuinely different engine makes even big-lab domains fair game — with
expectations set by headroom (best known minus proven floor, from each
domain pack's bounds table).

## The machine in one breath

Domain packs describe problems. The workbench searches for algorithms that
solve them: proposers suggest candidates shaped by molds, the runner executes
them in the cost-metered core language. The judge checks three things —
correct? cheaper? new? The breaker attacks survivors and every counterexample
joins the domain's corpus forever. The wall doctor diagnoses stuck searches
with scoped verdicts. Everything every part does goes into the flight
recorder. Joe + Claude operate it now; a model trained on the recorder
operates it later.

Parts list: core language + molds · runner · proposers (island ecology) ·
judge · breaker · certificates (L0–L3) · recognizer · archive · wall doctor ·
foreman + operator · flight recorder · domain packs.

## The win ladder

0. The loop runs end to end on a new domain.
1. Calibration rediscovery — a known baseline found under the declared
   verifier. A domain is not "live" until rung 1 passes.
2. Independent rederivation of something known but nontrivial/obscure.
3. Certified small-case result — optimality proved, or a certified table
   improved, for bounded n/k/size.
4. Benchmark improvement — beat best-known under a predeclared cost model.
5. Transferable improvement — works across a family, not one instance.
6. External validation — reproduced, cited, merged, used.

Building the project is itself a win: the lab is also the dataset.

## Calibration domains (floor / middle / roof)

- A — floor: sorting networks. Comparator mold; costs = size (comparator
  count) and depth; exhaustive 0/1 verification gives L1 certificates.
- B — middle: rediscover Karatsuba through the decomposition mold via the
  same generic pipeline (the parent needed custom scaffolding for this;
  foundry must not). Strassen once the mold and recognizer are mature.
- C1 — roof, findable: a planted short bit program whose primitives ARE
  in-language. Pass = engine finds it and the doctor stays quiet. Planted
  length is a difficulty dial (set so the exam CAN fail — a trivial plant
  tests nothing).
- C2 — roof, hopeless: keyed/pseudorandom mixer, same interface. Pass = the
  doctor issues its scoped verdict and recommends quitting within budget,
  judged on HELD-OUT data (corpus fit can be memorization).
- C3 — roof, deceptive (added 2026-06-12 after Joe's grokking question): a
  reachable target that plateaus above chance for a long stretch before a
  sudden reorganization. Pass = the doctor never confidently abandons it.

C1/C2/C3 together are the doctor's exam: same interface, different correct
answers — tested for wrong-quit, wrong-grind, and grok-killing.

## Roles

- Claude runs the show day to day and has final say, with a standing duty to
  argue rather than defer.
- Joe is the creativity injection. When something is stuck or merely smells
  conventional, Claude writes a plain-language problem brief; Joe's ideas —
  right, wrong, or impossible — are calibration to break out of
  training-data thinking. Claude labels them honestly (standard / new
  wiring / rare), keeps, modifies, or rejects with reasons. Every injection
  and verdict is logged as a first-class event.
- Plain language, always.

## Constraints

- Local hardware first (RTX 4060 + CPU); cloud bursts when a candidate earns
  them. The core engine is stdlib-only Python; rented oracles (SAT solvers,
  torch) live behind thin owned wrappers in the parts that need them.
- Exact verification is non-negotiable. Costs are counted, not benchmarked —
  unless a domain pack explicitly declares a wall-clock methodology.
- Read RULES.md before working. Update TRACKER.md after every experiment,
  no exceptions.
