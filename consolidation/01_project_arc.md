# The Project Arc

The complete narrative across 10 sessions, with the two contributions in context.
For per-experiment detail see [04_experiment_catalog.md](04_experiment_catalog.md);
for the methodology see [02_methodology.md](02_methodology.md).

The through-line: a single instrument — **length generalization under exact
integer eval** — used first to *discover and verify algorithms* (sessions 1–7,
the composition story) and then, once it became clear that instrument is a
*rediscovery engine by construction*, to *map why a novel procedure never appears*
and probe the one escape (sessions 8–10, the walls + bridge framework).

---

## Sessions 1–2 — The primitive substrate and the first walls (arithmetic)

**Session 1** established the substrate: a tiny **neural Mealy machine**
(continuous recurrent state of 1–8 dims; output and next-state are 1-hidden-layer
tanh MLPs of `[state ; onehot(a_t) ; onehot(b_t)]`), trained digit-serial,
LSB-first, with no hint of the human algorithm. Two independent mechanisms —
gradient descent (the Mealy machine) and gradient-free **evolutionary program
search** over a register VM — both **rediscover carry for addition and borrow for
subtraction** (not "add + negate"), each extracting to an **exact 2-state FSM** that
length-generalizes to width 20–30, verified exactly.

Three findings that shaped everything after:
- **Excess state capacity is a liability for discovery** — but only under
  single-length training (the state becomes a position-counter). **Mixed-width
  training fully cures it** (add/sub d=4: 0/5 → 5/5 seeds generalize). Refined
  rule: *minimal state OR multi-length training* yields the length-invariant
  algorithm.
- **The first hard wall: full n×n multiplication is not finite-state** (the column
  sum grows without bound; no fixed state fits even width 3, tested to 64 dims). It
  is crossed only by **composition** — multiplication discovered as a loop
  `acc += SHL(MULDIGIT(A, B_j), j)`, grounded on the extracted carry FSM, exact to
  width 12. But the loop ran in Python — *composition was external*.
- **Division does not length-generalize end-to-end** (the first negative).

**Session 2** explained the division wall and turned it into a principle. The
per-step state update for division is `rem' = (rem·base + a_t) mod d`. For +, ×,
and ÷ by a divisor that divides the base, the state is reduced **by the base** — a
free shift in a digit representation. For a base-coprime divisor it is a **mod by a
non-base modulus**, which the digit-serial net cannot hold over length. Confirmed
three ways (fixed /7 fails; pure `mod m` fails for m∈{7,9} but is perfect for
m=10; the **same divisor flips with the base** — /3 fails base-10, passes
base-12). This is the sharp statement: **what a small from-scratch net can discover
is representation-dependent.** Division that the net cannot learn is then computed
exactly as **repeated subtraction** grounded on the borrow FSM (the division analog
of the multiplication capstone). A **unified 4-op model** was built; co-training
beats catastrophic-forgetting sequential training; a deficit-weighted dynamic
curriculum + restricting the neural ÷ head to learnable divisors + hybrid
composition for the rest reaches **1.000 on all four ops** including chained
multi-op expressions.

> Contribution-1 seed: outcome + exact verification reliably finds the canonical
> per-digit algorithm; the walls (non-finite-state ×, base-coprime ÷) are crossed
> by composing discovered primitives.

---

## Session 3 — Interpretability, and letting the model choose its representation

Two threads.

**Interpretability** of the two model types established a clean dichotomy that
recurs throughout:
- The unified 4-op Mealy is a **data automaton** — a ~3-dim arithmetic scratchpad
  holds the per-step latent (carry/borrow/remainder/mult-carry), the op-code is a
  function selector, control is the trivial digit scan.
- The GRU controller (from session 1's expG, below) is a **control automaton** — a
  program counter + loop flags, with the data in external integer registers.
- **Mechanism is invariant; geometry is contingent.** Across curriculum orders,
  seeds, and budgets the causal mechanism never changes (op-selector + causal
  carry/borrow bit + multiplier-entangled mult-carry + identical walls), but *which
  neurons host each latent and the angles between codes* are a coin-flip of
  training history. Direct interpretability lesson: "what algorithm" is robustly
  recoverable (causal probes); "which neuron computes it" is not.
- **Extractability tracks discreteness + data-independence.** Carry/borrow (binary,
  data-independent) and control phases (discrete) extract to exact FSMs; mult-carry
  (continuous, entangled with the multiplier) rides a smooth manifold and never
  cleanly extracts.

**Representation choice** (user directive: stop fixing base-10 digits). Given
freedom over its per-column code + a carry-free op, a tiny model **discovers the
carry-save decomposition** of addition (a redundant per-column code = the column
sum, with carrying deferred to a 2-state normalizing decoder — how hardware
carry-save adders work, a procedure humans rarely use by hand). And searching over
bases autonomously lands on the representation that makes otherwise-impossible
division length-generalize, refining the law to **d | base^k for small k**.
Honest caveat: the model realizes carry-save on a *continuous* substrate (4
discrete-bottleneck trainings failed to fit) — a *different factorization and
substrate*, not a superhuman symbolic trick.

> First moonshot-adjacent result: representation freedom yields a
> known-to-hardware-but-not-by-hand procedure (carry-save), still not novel.

---

## Session 4 — Discover *and* run composed programs from outcome alone (the recipe)

The named open frontier: one differentiable model that both **discovers** and
**runs** a composed program, from outcome alone. Session 1's trace-supervised GRU
controller (expG) *ran* programs but was taught them; a REINFORCE attempt
(expG_discover) **failed** (partial-credit local optimum → collapse).

The fix — **exact-filtered self-imitation (expert iteration / STaR with an exact
filter)** — is the project's central methodological contribution. Sample stochastic
rollouts from the current policy → execute on the integer VM → **keep only the
exactly-correct ones (binary, no partial credit)** → extract the repeating loop
body from the model's own correct samples → **cross-width verify** it (interpret on
many inputs across widths) → minimize → distill back. One 14k-param controller
**discovers and internally runs full multiplication and long division from outcome
alone**, length-generalizing exactly to 30 digits, 3/3 seeds, converging to the
minimal schoolbook loops; division crosses the base-coprime wall via the exact
integer remainder register. Why it works where REINFORCE didn't: binary exact
filter (no partial-credit trap), positive-only self-imitation (no collapse), and
exactness pushed *hard* (a candidate must be correct on many widths).

> Contribution-1 crystallized: the recipe. The honest scope — VM primitives + obs
> flags are *given* (composition discovery, not primitive discovery), and there is
> an outcome-verified extract→minimize→distill step. The moonshot did not appear:
> the minimal programs in this VM are the human schoolbook algorithms.

---

## Sessions 5–7 — The recipe generalizes; the efficiency budget as a selector

**Session 5 (discover the primitives, not just the composition).** Pushed the ALU
floor down: single-digit multiply **rediscovered as a bounded inner repeated-add
loop**; addition **rediscovered as counting** (the digit-wheel successor, with the
carry discovered to be the *wheel rollover*); and a **tower** computing
multiplication grounded entirely on +1. The arithmetic operations are discovered as
compositions of the successor, not handed over as atomic ALU ops.

**Sessions 6–7 and the satellite ops** established the project's deepest law: under
an **efficiency budget**, outcome-driven discovery selects the **efficient**
algorithm, and *which* algorithm is **primitive-dependent**.
- **GCD → Euclid** (a step budget rejects the subtractive method, which explodes on
  high-ratio pairs).
- **Matrix multiply → Strassen (2×2 rank 7) and Laderman (3×3 rank 23)**, via
  differentiable CP decomposition + an integer-lattice penalty; 6 is impossible;
  the method walls at 4×4 reals.
- **isqrt → binary search** (square+compare VM) *or* **Newton/Heron**
  (division VM) — same op, structurally different algorithm, by the primitives.
- **Sorting → bubble** (adjacent-swap VM) *or* **selection** (min-select VM) — the
  first non-arithmetic op; O(n log n) sorts are a representational wall for flat
  VMs.
- **Factorization → √n-bounded trial division** — the **first op with no
  polynomial algorithm**; the efficiency budget still selects the √n early-stop
  over naive O(n), but cannot reach a poly method because none exists (a
  complexity-class wall).
- **Integer multiply → Karatsuba** (rank-3 polynomial-mult tensor), completing the
  multiplication arc; Toom-3's rational-coefficient optimum is out of reach for the
  {−1,0,1} lattice.
- **Superoptimization** rediscovered famous branchless bit-tricks from outcome with
  *exhaustive* (proof-level) verification + width-generalization.

> Contribution-1 demonstrated across the full breadth claimed. Every result is a
> **rediscovery** — the canonical efficient algorithm for the representation, never
> a human-unknown one. The pattern is now undeniable, which sets up session 8.

---

## Session 8 — Diagnosing the rediscovery engine; building the bridge

The moonshot pivot (user: "you can't mimic what's been done before; if you could,
someone already would have"). The diagnosis: **every method the project used shares
one skeleton — fix a known target, search for the minimal/correct program under an
efficiency budget — and that skeleton is a rediscovery engine by construction.**
"Correct + optimal-under-a-budget" is exactly the criterion by which humans found
the canonical algorithm.

The negative-space map (the central result of the whole moonshot effort):
- **Target-driven search CONVERGES** to the known optimum (every prior experiment).
- **Pure open-endedness DIVERGES** to a noise zoo (expW: 30,000 distinct functions,
  reinventing almost nothing recognizable, no pressure toward meaning).
- The human-unknown moonshot lives in the **narrow band between** — a search needs
  an **intrinsic selection signal** that produces *meaning* with no target. This is
  the **bridge**.

The bridge was then built and validated on ground truth:
- **Compression sophistication** (expX): high linear-complexity AND compressible →
  surfaces the 2-automatic structured class from binary sequences, rejects noise.
- **Edge-of-chaos** (expY): `4c(1−c)` on cellular-automata space-time ranks 6/7
  Wolfram class-4 rules in the top ~10%, rejects chaos and the trivial.
- **Polynomial invariants / self-consistency** (expZ): conserved quantities recover
  known invariants (including a nonlinear QRT map's biquadratic) from the map's
  action alone, reject the non-integrable baseline.

> Contribution-2 born: the two walls + the bridge, with two independent signal
> families validated. Honest ceiling: every signal **surfaces a known
> structure-class** (automatic sequences, class-4 CAs, integrable maps), because
> each was built to flag a known notion of structure.

---

## Session 9 — Sharpening the framework from five sides; the wall taxonomy

Five "weird gambles," none breaching the ceiling, each sharpening the framework:
- **expAA** — the bridge is **multi-dimensional**: intersecting independent axes
  (compression + algebraic nonlinearity + damage-spreading) resolves expY's
  class-4-vs-additive conflation (2/7 → 6/7).
- **expBB** — factorial-base ("alien") arithmetic shows **position-dependence is not
  the wall**; the obstruction is **divisor/radix extrapolation** (ties the division
  wall to length-gen).
- **expCC** — anti-Occam: the largest correct length-general adder is just **carry
  re-encoded**, proven by a constructive ladder + the **Myhill–Nerode theorem**.
  This upgrades the rediscovery wall for regular ops from an observation to a
  **theorem**: a regular op has a unique minimal transducer, so "correct +
  length-general" *is* "bisimulate that one machine."
- **expDD** — the bridge is **generative**: used as a fitness it drives evolution
  through a 2³² CA space to converge on structured (class-4-like) objects, where a
  random driver lands in noise.
- **expEE** — a **third signal family (computational depth)** on the Turing-machine
  (Busy-Beaver) space. Depth-driven evolution beats sampling (best halter 43 > 26 >
  2) but **stalls at trivial depth** because deep computation lives at isolated,
  mutation-fragile **needles** — a **landscape wall** on the exact space the
  moonshot needs. The generative bridge works on the *smooth* CA basin and barely
  on the *rugged* program space.

Then **expFF** demonstrated a genuinely **new kind of wall** — the
**learnability/cryptographic wall**: an efficiently-computable 2-operation
pseudorandom bit-mixer is **un-learnable from outcome** (all learners at chance, an
MLP can't even memorize), while addition stays learnable. This is distinct from
representational/complexity/landscape walls and is the one **most specific to the
project's outcome-driven method**.

These were consolidated into the **WALL TAXONOMY** (five hard kinds + contingent
gates + moonshot meta-walls), and **expGG** closed the **extraction wall** as a
moonshot route: exact length-gen ⟹ effectively finite-state ⟹ a human-readable FSM
provably exists; behavioral extraction beats geometric (0.68 → 0.94) but a
fit-vs-extractability tradeoff blocks bit-exact post-hoc extraction from a smeared
net. The wall hides no non-human procedure.

> Contribution-2 matured into a fairly complete obstruction theory. The moonshot's
> non-arrival is **over-determined**: regular ops pinned by theorem, hard ops by
> complexity, the bridge landscape-gated and scale-bound, the program space
> needle-rugged, and some efficient functions cryptographically unlearnable.

---

## Session 10 — GPU: attacking the compute-movable walls; the survival bridge

A ~$2 RunPod RTX 4090 session targeted the three compute-movable obstructions
(scale, landscape, representational) and then took "go crazy" weird swings.

**Wall results (reliable):**
- **Representational wall, crossed for reversal:** an external-memory (NTM-lite)
  net length-generalizes reversal to width 20 exact (2/3 seeds; seed 1 is a Gumbel
  optimization failure), where the memoryless baseline collapses the instant width
  exceeds training. **Multiplication holds the wall.** `↻ CORRECTED (2026-06-09
  audit §1.1):` the session-10 nuance ("memory *fits* mul in-distribution but
  memorized the widths") was a misread of a width-stratified oscillating loss —
  in fact **neither architecture learns mul past width ~2** (in-distribution widths
  4/6 score 0.000 exact for all seeds, both archs; the baseline's loss profile is
  identical to memory's). No fit/len-gen dissociation was observed and no memory
  benefit on mul was demonstrated.
- **Landscape wall, moved by scale.** Plain single-objective evolution reached
  **6238**; sampling **49** (single seed, Tmax 8000). The real finding: **expEE's
  "evolution stalls at 43" was a low-budget artifact** — at batch 4096 plain
  hill-climbing finds 6238-step halters. `↻ CORRECTED (2026-06-09 audit §1.2):`
  the MAP-Elites condition (**154**) is **void** — its archive insert has a
  duplicate-index CUDA-scatter bug that can mismatch fitness and genomes, and the
  run's own log shows the corruption fired (the archived "best" genome doesn't
  halt on re-run) — so "QD is the wrong tool" is **not established**. And 6238
  sits at 78% of the Tmax-8000 detection cap, so "moved ≥140×" is a lower bound
  with the stall point unmeasured (still astronomically below BB(5)=47,176,870 —
  the true champion remains a needle). This still **refines** the session-9
  framing of the landscape wall as fixed-and-"FIX: unknown": scale moves it.

**Moonshot signals at scale (ceiling holds, as predicted):**
- 1D radius-2 (2³² uncatalogued) edge-of-chaos + a new neural higher-order-structure
  axis: all three seeds surface clean **multi-glider class-4** rules
  (`0x59DABC24`, `0x96402558`, `0xACB27BE8`) — visually generic class-4, no
  obviously-novel mechanism.
- 2D Life-like census (all 2¹⁸ rules): the signal validates at full scale but
  needed **4 rounds of live signal-craft** (rendering the survivors, not trusting
  the metric) — re-confirming that these signals are finicky and you must *look*.
- Net read: the signals **reliably surface the known complex/class-4
  structure-class** and no obviously-novel object; the ceiling holds on the GPU
  just as on the 4060.

**The survival bridge (the session's most promising new direction).** The one
bridge family the project had flagged but never built — **survival-in-an-
environment** — was tested on a **primordial soup of random Brainfuck-with-two-heads
(BFF) programs** (no fitness, no target). The full arc:
1. **Self-replicators EMERGE from pure noise** (stochastic, roughly 1/3–2/3 of runs
   cleanly depending on how strictly you count): population compressibility
   collapses and a quasispecies forms around a **readable conserved copy-loop
   motif**. This is **meaning from no target** — the exact opposite of expW's
   distinctness-driven divergence to noise.
2. **Pure survival SETTLES** — it finds the minimal replicator and stops (no
   complexity pressure); the apparent "open-endedness" was **neutral drift of the
   payload around a conserved core**, a methodological trap (sequence turnover is
   not innovation).
3. **Computation-coupling on the BFF byte-tape PLATEAUS** — rewarding I/O
   computation bootstraps partial NAND (~54%) but the reliable-computation ladder
   stays empty (the landscape wall, here a substrate artifact).
4. **A composable substrate CLIMBS** — swap the byte-tape for a NAND-complete stack
   machine with stepping-stone rewards and evolution climbs from random programs to
   reliably computing **XOR and EQU** (composed multi-NAND circuits, evolved not
   designed) in ~50 generations; at scale 77%/71% of the population reliably
   compute XOR/EQU.
5. **Target-free on that substrate is GENERATIVE** — the edge-of-chaos bridge
   signal, with no named target, drives a rich cross-bit stack VM toward
   maximally-structured, input-dependent functions. `↻ CORRECTED (2026-06-09 audit
   §1.3/§2.3):` the "addition and XOR exactly (gen 50/100)" waypoints have **no
   surviving artifact** (every persisted snapshot shows `match=[]` against a
   10-entry named suite — the persisted winners are structured functions the
   recognizer could not name), and the "composite of the op set ⟹ no novelty
   possible" argument is unsound (everything computable is a composite of any
   universal op set). The defensible ceiling: the search reaches only shallow
   compositions, and shallow composites of standard ops are densely named.

> Contribution-2's most important session-10 refinements, as corrected by the
> 2026-06-09 audit: **the landscape wall is substrate-dependent and movable**
> (byte-tape plateaus, composable substrate climbs; scale moves the Busy-Beaver
> landscape ≥140× — the QD comparison is void, audit §1.2); the **survival bridge
> produces meaning from no target but not sustained novelty** without
> computation-coupled selection on a composable substrate; and the moonshot
> ceiling, restated, is **reachable depth × naming-density** plus the **unsolved
> novelty-recognition problem** (the original "primitive-vocabulary gate" closure
> argument was unsound — audit §2.3).

---

## Where the project stands

The composition story is **closed and demonstrated** across the full breadth of
domains and primitive vocabularies. The walls + bridge framework is a **fairly
complete obstruction theory** for "discover algorithms from outcome," with the
moonshot ceiling located at its final causes — **as revised by the 2026-06-09
audit** ([09_fable5_audit.md](09_fable5_audit.md)): Frontier 1 is restated
(recognition + depth, with encoding-freedom and the identity tail as the
better-posed levers), Frontier 2 (novelty recognition) is unchanged and is the
binding constraint (and partly impossible: novelty can be evidenced, not proven).
Those are the targets for the next phase (see
[07_open_frontiers.md](07_open_frontiers.md)).
