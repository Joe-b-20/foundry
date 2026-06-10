# Methodology — the recipe, the discipline, the failure modes

This is the "how," distilled so a successor does not re-derive it. Two engines did
the work: a **discovery recipe** (exact-filtered self-imitation, for sequential /
policy algorithms) and a **bilinear tensor-rank search** (for the multiplication-count
algorithms). Plus the discipline that kept the results honest, and the catalog of
failure modes and signal-craft bugs that were caught.

---

## 1. The instrument: length generalization under exact eval

The single binary test that organizes everything:

> Train on short inputs (width ≤ 3–5 / short lists / small numbers); test on long
> ones (to width 20–30, length 50, numbers to 10³⁰). Eval is **exact integer
> match — no partial credit**. A memorized lookup table collapses on long inputs; a
> genuine procedure stays at 1.000.

Consequences that recur:
- **It is a test of "real algorithm."** Used as the headline metric everywhere.
- **It is also a *selector*.** When several locally-correct candidates exist, only
  the length-generalizing one survives cross-width verification (it disambiguates
  bubble-sort's ambiguous memoryless state; it picks Euclid over subtractive,
  binary-search over linear, the √n-bound over naive trial division).
- **It forces an efficiency budget.** Under a per-instance step cap, the inefficient
  method busts the cap at moderate width and never generalizes — so the budget
  selects the efficient algorithm. This is the project's deepest law.

Implementation: `core_data.py` (`to_digits`/`from_digits` LSB-first codecs,
`exact_result`, `length_gen_report`). Ground truth is exact Python integer
arithmetic.

---

## 2. The discovery recipe — exact-filtered self-imitation

This is the reusable engine (reference implementation: `expJ_selfdiscover.py` over
the `expG_controller.py` VM + GRU controller; reimplemented per-op in expM/O/P/K/Q/R/S).

One model both **discovers** and **runs** the program. The loop:

1. **Sample** K stochastic rollouts per problem from the *current* policy
   (op-masked legal actions, ε-exploration), over a small integer-register VM.
2. **Execute exactly** on the VM and **keep only the exactly-correct rollouts**
   (binary filter — the whole answer must equal ground truth; no partial credit).
3. **Self-imitate** the kept programs by supervised cross-entropy (positive-only —
   never a negative gradient).
4. **Extract the repeating body** from the model's own correct samples: split each
   correct rollout into loop-delimited segments (e.g. `INC_J`-delimited cycles); a
   run of a flag-gated op (`SUB_D`) becomes a single `while ge: SUB_D` token.
5. **Cross-width verify** the candidate body by *interpreting* it (expand the loop
   per problem) on many inputs across widths — keep the body whose repetition is
   exactly correct across widths (this is the length-gen test used as a filter).
6. **Minimize** (MDL): greedily drop ops while the body still verifies, yielding the
   minimal clean program.
7. **Distill** the clean body back across widths (mixed-width → length-gen).
8. The final model **runs greedily with zero scaffolding** and length-generalizes —
   that is the honest eval.

**Curriculum**: width 1 ignites; advance to genuine non-trivial widths as greedy
accuracy climbs. A small replay buffer of verified programs guards against
forgetting.

**Why it beats REINFORCE** (which failed in `expG_discover` — partial-credit local
optimum then collapse):
- **Binary exact filter, never a shaped/partial-credit reward** (REINFORCE's trap —
  a wrong program that gets some answer digits right gets rewarded).
- **Positive-only self-imitation** — no destabilizing negative gradients, so no
  late collapse.
- **Exactness pushed hard** — a candidate "algorithm" must be exactly correct on
  *many* inputs *across widths* (interpret-verify), which extracts the
  length-generalizing loop from noisy one-off-correct rollouts and rejects
  width-specific flukes.
- **MDL/minimize** biases toward the minimal real algorithm.

**Two kinds of target** the recipe handles:
- A **sequential program** with loops (mul, div, isqrt, factorization) — extract the
  loop body.
- A **memoryless reactive policy table** (gcd, sorting, factorization) — read off the
  greedy obs→action map and verify it across sizes. For these, the "algorithm" is
  one or a few table entries; the science is in the *selection criterion*, not the
  search difficulty.

---

## 3. The tensor-rank engine — for multiplication-count algorithms

For bilinear maps (matrix multiply, integer/polynomial multiply) the natural
"algorithm" axis is the **scalar-multiplication count = tensor rank**, not a
sequential program. Reference: `expL_matmul.py`, scaled in `expN_matmul.py`, reused
by `expN_ladder.py` and `expT_karatsuba.py`.

- Frame the product as a rank-R CP decomposition of the operation tensor T:
  `c = Σ_r W_r (U_r·a)(V_r·b)`; it computes the product **exactly for all inputs**
  iff `Σ_r U_ri V_rj W_rk == T_kij` (an algebraic identity → length-gen automatic).
- **R = the efficiency budget.** Sweep R from naive down to the impossibility
  boundary.
- Fit by gradient (loss = tensor reconstruction error) with an **annealed
  integer-lattice penalty** `(x³−x)²` that pulls coefficients onto {−1,0,1}; round;
  **verify the exact tensor identity** (all entries, integer arithmetic) + on random
  integer matrices.
- **Batch the restarts as a leading dim** (the key scale enabler — hundreds of
  parallel inits in one vectorized einsum + Adam step), plus a polish/re-anneal of
  the best near-integer candidates.

This is deliberately **not** AlphaTensor's RL/transformer setup (per the "don't mimic
published work" rule). It rediscovers Strassen (2×2 rank 7), Laderman (3×3 rank 23),
the rectangular optima, and Karatsuba (rank-3 polynomial-mult); it walls at 4×4
reals (optimizer cost + the recursive structure a flat search can't represent) and
the GF(2)/Toom-3 rational-coefficient regimes.

---

## 4. Extraction — reading the discovered procedure out

The project's second requirement (extract + verify the procedure), and its
recurring asymmetry:
- **2-state algorithms (carry, borrow) extract to exact FSMs** by sign-bit
  discretization (`expA_mealy.extract_fsm`) — binary and data-independent.
- **Control phases (the GRU controller) extract to an exact control FSM** — phases
  are intrinsically discrete (a program has discrete steps).
- **>2-state arithmetic latents (mult-carry, remainder) do not cleanly extract** —
  they ride a continuous (carry × data) manifold (mult-carry is entangled with the
  multiplier). Geometric k-means stalls (~0.94 for mult). `fsm_extract.py` is the
  general k-means/majority-vote extractor.
- **Two routes to a clean FSM**: (a) **bake discreteness in during training** — a
  sign-STE makes the FSM exact *by construction* (BFS over reachable hypercube
  vertices), at a cost: hard discreteness *hurts* >2-state fit (`expD_discrete.py`);
  (b) **behavioral extraction post-hoc** — on-distribution carry-read + majority
  vote over the net's real transitions beats geometric (0.68 → 0.94) but is not
  bit-exact on a smeared net (`expGG_extraction.py`).
- **The verdict (expGG):** exact length-gen ⟹ effectively finite-state ⟹ a
  human-readable FSM provably exists (Myhill–Nerode). Un-extractability is a
  continuous-storage artifact + a fit-vs-extractability tradeoff, **not** a hidden
  non-human procedure.

---

## 5. The discipline (from `.claude/RULES.md`)

These are not decoration — they are why the results are trustworthy.

- **The tracker is sacred.** Every experiment — works, fails, partial, abandoned —
  gets an entry with concrete numbers. Search it before starting anything to avoid
  repeating a dead end.
- **Eval is exact.** No partial credit on arithmetic. *Wanting to relax this is a
  tell that an approach isn't actually working.*
- **Abandonment is first-class (no-grind).** If you've tuned the same core approach
  >3 times without signal, mark it ABANDONED with a one-sentence reason and try
  something structurally different. (Applied e.g. to the blind anti-Occam census,
  the GF(2) soft-XOR, the 4×4 matmul wall.)
- **Be weird.** Ask the published-paper answer, then deliberately try a stranger one
  that fits the small/narrow/exact-verification constraints. Use the conventional
  approach only as a baseline.
- **Honesty about uncertainty.** Pre-register predictions and say when you're unsure;
  report failures as failures, not "partial validation"; say "ambiguous" and what
  would resolve it. *The prior version of this project failed because elaborate
  narratives got built on unverified results.*
- **Separate discovered from scaffolded.** Every "works" entry states the
  irreducible **floor** (the given primitives, obs flags, validity filters, control
  affordances) vs what was genuinely **discovered from outcome**. This is the single
  most important honesty habit in the project.
- **Start small, scale on signal.** Smoke configs exercise every code path; scale is
  the same code with bigger numbers. (Session 10 lesson: smoke then *measure pace*
  before a multi-seed campaign — the neural OOM, the 2D signal-craft, and the QD
  launch-bound were all scale-only and invisible in 4060 smokes.)

---

## 6. Validity filters (scaffolding that constrains *validity*, not the algorithm)

Several discoveries needed filters on *which* correct rollouts to learn from, to
kill trivial programs that pass the exact check by coincidence. These constrain
**validity** (read your input; don't subtract past zero; be a genuine memoryless
policy) — they do **not** specify the algorithm. They are documented scaffolding,
and every "works" entry names them:

| Experiment | Validity filters | What they reject |
|-----------|------------------|------------------|
| Division (expJ) | must contain `GETDIGIT` (read the input); `SUB_D` must be ge-guarded (only when valid) | "SUB_D STOREQ" gaming quotient=1 without reading A |
| isqrt (expO) | `consistent` (the `TAKE` agrees with the `le` flag); freshness rule (a `TAKE` must follow a fresh probe) | inconsistent / off-probe takes |
| Sorting (expQ/R) | `clean` (never SWAP an in-order pair); `consistent` (induced obs→action map is a function) | the "always SWAP" degenerate collapse; exploration noise masquerading as a policy |
| Factorization (expS) | `clean` (FACTOR only when `div`=1); `consistent` | the "declare-prime gamble" (EMIT_N immediately — correct for primes, wrong for composites) |

The general pattern: **consensus/majority vote alone is not enough** when a locally
correct but non-generalizing action exists. Propose candidate tables from the
model's own top actions and let the **cross-width/size exact verify** pick the
generalizing one (the "isqrt lesson," reused for sorting and factorization).

---

## 7. Documented failure modes (the recipe's real edges)

- **Ignition** is the live risk: will random sampling at width 1 ever hit an
  exactly-correct program to seed self-imitation? Deeper programs (carry-addition
  from counting) needed a **no-carry warmup** to ignite the core loop before forcing
  carries.
- **The naive-attractor** without cross-width verify: raw self-imitation ignites the
  *easy/short* attractor (linear scan for isqrt; "always SWAP" for sorting) because
  the efficient leap needs two simultaneous mutations exploration won't find. The
  cross-width verify of the extracted body — not the policy gradient — does the
  selecting.
- **REINFORCE on a shaped reward**: partial-credit local optimum → collapse. Do not
  use (see `expG_discover`); use exact-filtered self-imitation.
- **Hard discreteness hurts >2-state fit**: a sign-STE makes extraction exact but the
  net fits worse for mult/div; the net *prefers* the smooth manifold there.
- **Catastrophic forgetting** in sequential multi-op training; cured by rehearsal /
  deficit-weighted dynamic curriculum.

---

## 8. Signal-craft bugs caught (the bridge's real edges)

The intrinsic-meaning ("bridge") signals are **finicky and representation-
dependent**. Each of these was a real bug caught by a noise baseline or by actually
rendering the survivors — they are the single most reusable warning for the
frontiers. **Do not re-introduce them.** (Cross-ref [08_what_not_to_redo.md](08_what_not_to_redo.md).)

1. **Bit-packing artifact (expX).** Ranking by raw `1 − zlib_ratio` with bits stored
   one-per-byte: 7/8 bits are always zero, so zlib fake-compresses *everything* to
   ~0.24 and a random-noise baseline scored 0.76. Fix: pack bits 8-per-byte (random
   → genuinely incompressible) and weight by linear complexity. The noise baseline
   is what exposed it.
2. **IC-contaminated column-LC (expY).** `(1−c) × center-column-LC` ranked class-2
   *periodic* rules on top, because a pure shift rule copies the random initial
   condition down the column → fake-high linear complexity, and `(1−c)` rewards
   *maximal* compressibility (the opposite of complexity). Fix: edge-of-chaos
   `4c(1−c)` on the whole space-time, which peaks at *intermediate* compressibility.
3. **TM space-time sparsity contamination (expEE).** Edge-of-chaos compression
   doesn't transfer to Turing-machine space-time: the head touches one cell per
   step, so the diagram is ~blank and highly compressible (c ≈ 0.09) regardless of
   the computation's complexity — every machine sits at the same low c. A
   TM-appropriate structure measure (on the head-track or output tape) is needed.
4. **Residual-novelty rewarding maximal compressibility (GPU prep).** The
   "structured-but-not-named" score as a *pure subtraction*
   `general_bpc − named_bpc` rewards maximal compressibility, so it ranks
   frozen/blinking rules on top, not complex ones. Fix: the validated
   intermediate-compressibility `4c(1−c) × damage-band` finder (the damage band
   needs **both** cuts — a high cut kills chaos *and* a low cut kills frozen
   healers; an early version had only the chaos cut). Caught on the 4060 before
   spending GPU money.
5. **Sequence diversity as a false innovation metric (alife).** In the primordial
   soup, sequence turnover stayed ~1.0 and the auto-metric printed "OPEN-ENDED" — but
   that turnover was **neutral drift of the payload around a conserved functional
   core**, not innovation. The functional-complexity signals (ops-count, conserved-
   motif content) said SETTLED. **Use functional, not sequence, metrics for
   open-endedness.**

Plus the live session-10 corrections: the **2D census** needed four rounds of
signal-craft (B0 global-flashing noise fooling the damage measure → an activity
cut; complex-looking *static debris* → an activity floor; B0 rules having no
quiescent background → a `lut[all-dead]==1` gate). The recurring lesson: **"looks
complex by a metric" ≠ "is dynamically complex" — you must render and look.**

---

## 9. The honest meta-lesson

The recipe (Contribution 1) is a **rediscovery engine by construction**: "correct +
optimal-under-a-budget" is exactly how humans found the canonical algorithm, and for
regular operations this is a *theorem* (Myhill–Nerode uniqueness, expCC). The bridge
(Contribution 2) is the only escape from that, but every validated bridge signal was
*built to flag a known notion of structure*, so it resurfaces known classes. The two
frontiers ([07_open_frontiers.md](07_open_frontiers.md)) are precisely the two places
where this methodology, as it stands, cannot reach a human-unknown object — and
naming them that precisely is itself the methodological payoff.
