# What a Small Network Can Discover — and What It Took to Believe Our Own Results

*Draft v1 (2026-06-10). One long-form piece, structured so §A/§B/§C can be split into
standalone posts/papers. Every number is artifact-backed (repo: `TRACKER.md` + `runs/`
+ `runs_pod/`; per-claim file references in the consolidation docs 01–10). Author: Joe
Bachir, with two generations of Claude instances as research assistants — which turns
out to be part of the story.*

---

## The project in one paragraph

mathlab asks whether a small, from-scratch neural system — nothing pre-trained, models
of 1k–40k parameters plus symbolic search — can **discover its own mathematical
algorithms** from input→output examples alone, under a single uncompromising test:
**length generalization with exact integer evaluation** (train on short inputs, test on
long ones; a lookup table collapses, a real procedure stays at 1.000; no partial
credit, ever). The far goal — the "moonshot" — was a procedure humans haven't found.
Over ~11 sessions on an RTX 4060 and a few dollars of rented 4090, the project produced
three things: a **rediscovery engine** that reliably finds the canonical efficient
algorithm across a dozen domains; an **obstruction theory** — a map of exactly why a
*novel* procedure never appears; and, unexpectedly, a case study in **how AI-assisted
research goes wrong and how to catch it**.

---

## §A. The science: a rediscovery engine and the walls around novelty

### A.1 What was discovered (all exact, all length-general)

One recipe — *exact-filtered self-imitation*: sample programs from the current policy,
keep only the exactly-correct ones, extract the repeating loop body, verify it across
widths, minimize, distill — discovers and internally runs, from outcome alone:

- **addition/subtraction** as carry/borrow (exact 2-state FSMs, extracted and verified);
- **multiplication and long division** as the schoolbook loops (one 14k-param
  controller, 3/3 seeds, exact to 30 digits);
- the primitives themselves: **addition rebuilt from counting** (carry = the digit-wheel
  rollover), **single-digit multiply from repeated addition**;
- with an efficiency budget as the selector: **GCD → Euclid**, **isqrt → binary search
  or Newton** (depending on the primitives offered), **sorting → bubble or selection**,
  **factorization → √n-bounded trial division**, **matrix multiply → Strassen and
  Laderman** (rank budgets on the operation tensor), **integer multiply → Karatsuba**;
- given freedom over its own number representation: **carry-save addition** — the
  hardware engineers' algorithm, not the schoolbook one.

### A.2 Why it never surprises: the wall taxonomy

The honest framing crystallized mid-project: *"correct + optimal-under-a-budget" is
exactly the criterion by which humans found the canonical algorithms*, so this engine
is a **rediscovery engine by construction**. The interesting science is the obstruction
map — why each route to novelty closes:

1. **Representational** — some functions exceed the substrate (n×n multiplication is
   not finite-state; external memory crosses the wall for reversal but showed no
   benefit for multiplication).
2. **Complexity-class** — for factoring, no efficient algorithm exists to find.
3. **Convergence-by-theorem** — for *regular* operations, Myhill–Nerode uniqueness
   makes rediscovery a theorem, **at a fixed I/O encoding**: every correct
   length-generalizing adder is provably the carry machine re-encoded. The one escape
   is changing the encoding itself — which is precisely how carry-save appeared.
4. **Landscape** — deep computation lives at isolated, mutation-fragile needles
   (Busy-Beaver territory); measured: depth-evolution's yield is heavy-tailed and
   seed-dominated (596–1887 across seeds at identical config), and depth and structure
   are **in tension** — deep halters are near-regular, structured trajectories don't
   halt (verified with an uncontaminated head-track signal after the original
   space-time measure was shown to be sparsity-contaminated).
5. **Learnability** — an efficiently-computable bit-mixer defeats statistical learners
   at R≥1 rounds; but the project's own program search recovers it *exactly* from
   examples while its description is short, and dies at a deeper secret: for
   search-based discovery this is a **description-length wall**, not an op-count wall
   — measured, not asserted (2^24-budget search vs 2^40 key space).

### A.3 The bridge, and where the ceiling actually is

Target-free search needs an intrinsic "interestingness" signal (the **bridge**) to
escape both rediscovery (targets converge) and noise (pure novelty diverges). Six
signal families were validated — compression sophistication, edge-of-chaos, polynomial
invariants, computational depth, learning progress, survival — and the most striking
single result of the GPU phase: **self-replicators emerge from random code under pure
survival selection**, with readable conserved copy-loops.

But every bridge signal was built to flag a *known* notion of structure, and the final
phase located the real ceiling by elimination:

- Target-free evolution on a rich substrate converges to **structured functions that
  match nothing** in an 85-entry named suite (the early claim that it "discovered
  addition" did not survive re-instrumentation — see §B);
- Growing the substrate's reachable composition depth **16×** leaves nearest-named
  similarity flat (~0.58–0.60): the *namedness* of what's found is depth-invariant;
- Yet a purpose-built **function telescope** (ANF degree, polynomial fits mod 2¹⁶,
  encoding-equivalence, equivariances, dependency flow) shows those unnamed functions
  form a coherent class — carry-like triangular nonlinear mixers — whose **algebraic
  degree tracks reachable depth** (2–4 → 8 → 16) even as namedness stays flat.

So the moonshot's binding wall is **recognition**: surfacing structured unfamiliar
objects is easy and target-free; *certifying* that an unfamiliar object is meaningful
— rather than arbitrary-structured — is the unsolved problem, and "human-unknown" is
in the end evidenceable but not provable.

### A.4 The one direction with a defined win — and how far it goes

Identity space (continued fractions, integer relations) is the only regime where
small-compute search has historically produced genuinely human-unknown mathematics
(BBP; the Ramanujan Machine). The project built a two-stage instrument — GPU float64
sweep of polynomial continued fractions with streaming Möbius-proximity prefilter,
then exact PSLQ verification with positive controls, a reject-rational filter, and
literature subtraction — and scaled it from 4.8M to 10.2 **billion** PCFs:

- It independently **rediscovered the Ramanujan Machine's still-unproven 8/(7ζ(3))
  conjecture from outcome** (exact coefficients, verified to 250 digits);
- It reached and verified continued fractions for **Catalan's constant (1/2G)** and
  **8/π²** — and the recognition pipeline then *correctly classified them as known*
  (the Catalan form is exactly the κ=0 member of a published family, matched against
  the fetched paper);
- Past the published coefficient region (3.37B PCFs at heights 20× beyond): only the
  known forms. An irrationality-quality (δ) blind hunt surfaced exactly what theory
  predicts — periodic surds — and nothing else.

The instrument demonstrably operates in the right space; the low-height polynomial-CF
region is now mapped as catalogued; novelty there requires height, constants, or
objects beyond a hobby budget. **No novelty is claimed anywhere in this project.**

---

## §B. The meta-result: an AI audited an AI, and every correction held up

This may be the most broadly useful thing the project produced.

The first ten sessions were run by one model instance (Claude Opus 4.x). At handover, a
successor instance (Claude Fable 5) was instructed to **audit everything before
touching anything**: re-verify claims against code and artifacts, hunt for framings
that exceeded evidence. The audit (consolidation/09) found that the project's honesty
discipline had held at the *number* level — of ~30 re-verified claims, ~22 reproduced
exactly — but leaked at the *reading* level, three times, by the same mechanism: **a
live read of a log tail became a tracker sentence, survived consolidation (which
cross-checked numbers, not readings), and was upgraded into a framing.**

The three reversals, then what re-running showed:

1. **"The memory net fits multiplication but memorized the widths."** The loss curve
   was width-stratified and oscillating; the final printed value (3e-4) was a width-1
   batch. Both architectures fit only trivial widths; in-distribution exact accuracy
   was 0.000. *The claimed phenomenon did not exist.*
2. **"Quality-diversity is the wrong tool — 40× worse."** The MAP-Elites archive used a
   duplicate-index CUDA scatter whose write order is undefined: archived fitness and
   genome could come from different machines — and the run's own log showed the
   corruption (archived best 154; the stored genome doesn't halt). Re-run with a fixed
   archive and seeds: **the verdict inverts** — QD is competitive, and with a
   depth-aligned descriptor it *beats* plain evolution (2139 vs 675). The "6238" that
   anchored a "140×" headline was a single upper-tail draw of a heavy-tailed statistic.
3. **"Target-free evolution discovered addition by gen 50."** No surviving artifact
   contained it; re-running with 10-generation logging across three seeds shows it
   **never happens** — trivial projections appear early and are abandoned for unnamed
   structured functions. The honest result was the *opposite flavor* of the claim, and
   more interesting.

Two further audit findings were *scope* corrections that redirected strategy: the
Myhill–Nerode wall holds only at a fixed I/O encoding (the project's own carry-save
result is the demonstrated escape), and "everything found is a composite of the op set,
so nothing novel can appear" proves too much (every algorithm ever published is such a
composite) — the actual gate was depth-reach × the naming-density of the shallow
region, which the function telescope later measured directly.

What made the audit work — and what we'd recommend to anyone doing AI-assisted
research — distilled into standing rules:

- **Claims need artifacts**: any number cited must exist in a file; live-stdout
  numbers are tagged and uncitable until re-run.
- **Comparative claims need seeds** (or an explicit n=1 tag): extreme-value statistics
  from single runs anchored two false headlines.
- **Re-execute archived winners**: one re-run of a stored genome exposed the archive
  corruption instantly.
- **Render before verdict**; **stratify mixed-condition losses**; **state the scope
  when upgrading an observation to a law** — each rule maps to a specific failure that
  it would have prevented.
- **Append-only logs with dated errata**, under version control.

The arc completed cleanly: every audit reversal was confirmed by re-run; the corrected
framings (heavy tails, descriptor-dependence, the unnamed attractor, the
description-length wall) each *replaced a wrong result with a more interesting one*.

---

## §C. The identity pipeline, as a methods note

(Compressed; the full recipe is `gpu_pcf_hunt.py` + `pcf_quadmine.py` +
`fn_telescope.py`, all runnable.)

1. **Sweep** (GPU, float64): polynomial continued fractions by family — deg-2 grids;
   deg-3 × b=−n⁶ (Apéry class); deg-3 × deg-6 general — via the renormalized
   convergent recurrence; 1.9–10M PCFs/s on one RTX 4090. **Stream every filter**
   (the one OOM in the campaign came from accumulating 3.4B raw survivors).
2. **Prefilter** (GPU): keep survivors within 1e-8–1e-11 of any low-height Möbius
   transform of a constant battery (precomputed table, ~3–5M values), plus a blind
   reservoir as the null arm, plus a δ (irrationality-quality) top-K for the
   blind-approximator hunt.
3. **Verify** (CPU, local — pod CPUs are slow): mpmath/PSLQ per candidate
   ([1, C, v, vC], and quadratic [1, C, C², v, vC, vC²] with algebraic-constant
   guards), reject-rational filter, ≥3-constant triviality filter, then re-verification
   at 250 digits with convergence-matched term counts.
4. **Controls or it didn't run**: every family carries an in-grid positive control
   (4/π; Apéry 6/ζ(3); 30/π²) — a null without a recovered control is declared
   uninterpretable, never "no result."
5. **Reference-subtract before any claim**: search + fetch the literature and match
   coefficients (this correctly demoted all three tail-constant finds to
   rediscoveries).

Total compute for everything in §C: roughly **$12 of rented 4090 plus a laptop**.

---

## Coda: where this leaves the moonshot

Open, and honestly placed. The walls are mapped and measured; the one historically
fertile direction has a working, validated instrument whose low-height region is now
known to be fully catalogued; the recognition problem — telling *meaningful-unfamiliar*
from *arbitrary-structured* — is the single binding constraint in every object space
tested, and one half of it is provably beyond reach (novelty can be evidenced, never
proven). The contribution stands regardless: a sharp account of why outcome-driven
discovery converges to the known, a toolkit that holds itself to exact verification,
and a demonstrated protocol for keeping AI-assisted research honest — including from
its own assistants.
