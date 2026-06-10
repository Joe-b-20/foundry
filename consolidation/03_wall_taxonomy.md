# The Wall Taxonomy (refined, post-session-10)

The map of **every obstruction the project hit**, sorted by kind. The point of the
taxonomy: the walls are not all the same thing, and knowing which kind you face
tells you whether to change the substrate, the budget, the search, or give up.

This is the session-9 taxonomy **with session-10's refinements folded in and marked
explicitly**. Where a later result changed the framing of an earlier wall, that is
called out as `↻ REFINED`.

> **The headline:** the moonshot's non-arrival is **over-determined**. Regular ops
> are pinned to their unique minimal machine *by theorem* (at a fixed I/O encoding —
> see #3's scope); hard ops are pinned by complexity; the escape (the bridge) is
> substrate-/landscape-gated and scale-bound; the program space is needle-rugged;
> and some efficient functions are statistically unlearnable from outcome (#5's
> corrected scope). `↻ 2026-06-09 audit:` the session-10 localization of the ceiling
> at "the primitive-vocabulary gate (#6) + novelty-recognition (#11)" is **restated**
> — #6's closure argument was unsound. The ceiling now reads: **reachable depth (#4)
> × naming-density of the shallow region**, plus **novelty-recognition (#11)**, with
> **encoding freedom** (#3's scope) the one demonstrated escape for regular ops.
> Corrections are marked `↻` below; full detail in
> [09_fable5_audit.md](09_fable5_audit.md).

---

## A. Hard walls (a real impossibility, not a tuning problem)

### 1. Representational (Chomsky / state-growth)
The **substrate cannot represent** the function.
- Full n×n multiplication is not finite-state (session 1, "THE WALL" — the column
  sum grows without bound; tested to 64 state dims).
- O(n log n) sorts need a stack a flat comparison-swap VM lacks (session 7).
- Sub-Strassen 4×4 matmul needs recursive/block structure a flat tensor
  decomposition can't hold (session 6).
- **FIX:** a richer substrate (loops, stack, recursion, external memory).
- `↻ REFINED (session 10, gpu_exp3) — then ↻ CORRECTED (2026-06-09 audit §1.1):`
  external differentiable memory (NTM-lite tape) **crosses** the wall for
  **reversal** (a stack-shaped op — exact length-gen to width 20, 2/3 seeds) where a
  memoryless net collapses. That stands. The session-10 claim that the memory net
  "*fits* mul in-distribution (loss → 3e-4) but memorized the training widths" is
  **contradicted by its own log** (`runs_pod/runs/exp3_mul.log`): the loss is
  width-stratified and oscillating (fit at width 1 only; median CE 0.97–1.65 at
  widths 3–6), the **baseline shows the identical profile**, and exact accuracy at
  the *in-distribution* widths 4/6 is **0.000** for every seed of both
  architectures. **Corrected statement:** memory demonstrated a crossing for a
  stack-shaped op; for multiplication neither architecture learns past width ~2 — a
  plain fit failure, not a fit/length-gen dissociation. No evidence yet that a
  richer substrate helps mul at all.

### 2. Complexity-class
The **efficient algorithm does not exist in any representation.**
- Factoring has no known polynomial algorithm (session 7, expS): trial division is
  exponential in #digits, so "length-gen under a poly budget" is impossible *in
  principle*; the efficiency budget still selects the √n early-stop over naive O(n),
  but cannot reach a poly method because none exists.
- **FIX:** none (it's complexity theory). Sub-exponential improvements (Pollard rho,
  sieves) need machinery this VM can't express and are not polynomial anyway.

### 3. Convergence-by-theorem (Myhill–Nerode uniqueness)
A **regular** operation has a **unique minimal transducer**, so "exact correctness +
length-generalization" *is literally* "bisimulate that one machine" — target-driven
search is **forced to rediscover** the known algorithm.
- Proven in session 9 (expCC): the largest correct length-general adder is just
  carry **re-encoded** (a constructive ladder of bisimilar adders + the theorem).
- This is **why** no regular-op experiment in sessions 1–8 ever surprised — it was
  not Occam/minimality pinning addition to carry; it was *regularity + correctness*.
- `↻ SCOPE (2026-06-09 audit §2.1):` the theorem pins the machine only at a **fixed
  synchronous I/O encoding** (LSB-first digit pairs in, one digit out per step). The
  project's own **carry-save adder** (expI, session 3) is a correct, length-general
  adder *outside* that scope — a different transduction (redundant output code +
  normalizing decoder), not a re-encoding of carry. For regular ops, algorithmic
  variety lives exactly in the encoding choice; this is the only escape route the
  project has *demonstrated* to yield a non-canonical algorithm, and it was dropped
  after session 3.
- **FIX:** leave regular ops / leave target-driven search → the bridge — **or change
  the I/O encoding** (carry-save precedent; signed-digit / residue systems untried).

### 4. Landscape  `↻ REFINED (session 10) — now substrate-dependent and movable`
The interesting objects are **isolated, mutation-fragile needles** (deep computation
= the Busy-Beaver / halting landscape), neither dense enough to **sample** nor smooth
enough to **evolve** toward. The generative bridge that works on the smooth CA basin
(expDD) barely moves here (expEE: depth-driven evolution stalls at 43 steps).

Session 9 marked this wall **"FIX: unknown."** Session 10 changed that on two axes:
- **Scale moves it (heavy-tailed); QD is competitive, not "the wrong tool"**
  `↻ MEASURED (2026-06-10 closure, exp2fix — supersedes the audit §1.2 "void" note)`.
  The closure re-ran with a dedup-safe archive (per-cell argmax, no duplicate-index
  scatter) + mandatory re-execution of every archived winner (all `archive_ok=True`).
  Five runs, n=5, batch 4096, deepest halter:
  | seed/cfg | sampling | evolution | MAP-Elites |
  |---|---|---|---|
  | s1 (span_ones, Tmax 8000) | 51 | 675 | 410 |
  | s2 | 45 | 596 | 201 |
  | s3 | 50 | 1887 | 713 |
  | rt_span descriptor | 51 | 675 | **2139** |
  | Tmax 30000 | 81 | 675 | 258 |
  Two corrections to the session-10 story: **(a)** the buggy "ME 154" was archive
  corruption; with a valid archive MAP-Elites is the **same order** as evolution, and
  with a **depth-aligned descriptor (rt_span) it BEATS evolution 2139 vs 675** — the
  crossing the experiment was built to detect. QD's depth is **descriptor-dependent**
  (the known QD fact), not "the wrong tool." **(b)** Evolution depth is
  **heavy-tailed and seed-dominated**: 596 / 675 / 1887 across seeds at identical
  config, and the original session-10 **6238 was an upper-tail draw** (≈3–10× the new
  median). So "scale moves the wall 140×" → **"~1–2 orders of magnitude, heavy-tailed
  — report the median of ≥3 seeds, never one number."** (Tmax 30000 did not raise
  evolution's best above 675, so at this budget the cap is not binding — the
  detection-cap caveat applies specifically to the old 6238 run.) BB(5)=47,176,870
  remains astronomically out of reach; scale moves the *floor* of reachable depth,
  not the ceiling. The deepest machine across all runs (rt_span ME, 2139) was
  re-verified by re-execution. *Both* audit flags on this experiment (the scatter bug
  and the n=1) were real and each distorted the headline.
- **Substrate-dependence for incremental-complexity evolution.** Coupling
  reproduction to computation **plateaus** on the BFF self-modifying byte-tape
  (gpu_metabolism: bootstraps partial NAND ~54% but the reliable-computation ladder
  stays empty) because there is no smooth fitness path from partial → reliable →
  composed. **Swap to a composable substrate** (NAND-complete stack machine,
  gpu_avida) and the same idea **climbs** from random programs to reliably computing
  **XOR/EQU** in ~50 generations. **Refined statement:** the landscape wall for
  *incremental complexity growth* is **substrate-dependent and movable** — choose a
  substrate with composable primitives + a smooth task-reward gradient.
- **FIX (updated):** scale (for raw depth/sampling-rarity) and substrate choice (for
  incremental complexity). It is no longer "FIX: unknown" — though the deepest
  Busy-Beaver needles remain out of reach.

### 5. Learnability / cryptographic  `(NEW in session 9, expFF)`
The algorithm **exists and is efficient**, but the function's input→output map
exposes **no structure exploitable from examples** (pseudorandom / one-way), so
**outcome-driven discovery can only memorize** — held-out/length generalization is
impossible by cryptographic hardness.
- A 2-operation reversible bit-mixer demonstrates it for **statistical learners**:
  at R≥1–2 rounds, linear, kNN, and a small MLP are all at chance while addition
  stays learnable (MLP 0.756); the rounds knob (R=0 → 100%, R=2 → 50%) shows it's
  the function, not the learners. (The "MLP can't even memorize" observation is a
  capacity artifact — a ~4k-parameter MLP vs 64k random target bits — not extra
  evidence of pseudorandomness.)
- `↻ SCOPE-CORRECTED (2026-06-09 audit §2.2):` the demonstration covers
  **statistical / local-generalization learners**, not "outcome-driven discovery"
  per se. The project's own engine is exact-filtered **program search**, and the
  demo function has a ~30-bit description (shift 7, multiplier 0x9E37, R rounds) —
  short-program enumeration with exact checking against examples finds it easily
  (a wrong short program passes 4,000 16-bit exact checks with probability ~0).
  For this project the cryptographic wall is properly a **description-length
  wall**: it binds when the function's secret/description exceeds what the search
  can enumerate (real keyed crypto), not at "2 ops." **Demonstrated** the same day
  (`expFF_search.py`, `runs/expFF_search.log`): depth-2 exhaustive search recovers
  the exact mixer program from examples (held-out 1.000; zero false positives on a
  random-permutation control), and dies at a depth-4 secret (~2^39.8 key space vs
  ~2^24.3 budget) — the wall is at the enumeration boundary.
- This remains the wall most specific to outcome-driven *statistical* learning.
- **FIX:** none for statistical learners; for exact-filtered program search the
  wall begins at description lengths beyond enumeration reach.

---

## B. Contingent walls / gates (movable, not fundamental)

### 6. Primitive-vocabulary gate  `↻ RESTATED (2026-06-09 audit §2.3)`
The **primitive set gates which algorithm is found** (solid, sessions 1–7) — but the
session-10 claim that it **bounds novelty** was unsound as argued.
- Examples (the solid half): clamped-subtract blocks borrow (signed arithmetic fixes
  it, session 1); a division-VM finds Newton-isqrt while a square+compare-VM finds
  binary-search-isqrt (session 7); a sort VM picks bubble vs selection by its
  primitives; the {−1,0,1} lattice reaches Karatsuba but not Toom-3's
  rational-coefficient optimum.
- `↻ REFINED (session 10, gpu_avida_oe) — then ↻ RESTATED (audit §1.3/§2.3):` the
  target-free edge-of-chaos bridge on a rich cross-bit stack VM (ops: nand, and, or,
  xor, add, sub, shl, shr) *generatively* evolves structured computation with no
  named target — that stands. Two corrections. **Evidence:** the "discovered
  addition/XOR exactly (gen 50/100)" waypoints have **no surviving artifact** (the
  stdout logs show `match=[]` at every persisted snapshot, against a 10-entry named
  suite); the persisted best organisms are structured functions the recognizer
  **could not name**. **Logic:** "every discovered function is a composite of the op
  set, so no human-unknown procedure can appear" proves too much — *every*
  computable function is a composite of any universal op set (human-unknown
  algorithms included; NAND's closure is all of boolean space). The defensible
  finding: **the search reaches only short/shallow compositions, and the shallow
  region of standard vocabularies is densely named.** The ceiling is *reachable
  depth × naming-density* — i.e. the landscape/depth problem (#4) plus
  novelty-recognition (#11) — **not closure**.
- **Not a hard wall — but not the lever as originally stated:** making shallow
  composites *un-named* is trivially achievable with exotic primitives and buys
  nothing by itself (un-named-by-obscurity ≠ interesting). See the **revised
  Frontier 1** in [07_open_frontiers.md](07_open_frontiers.md).

### 7. Optimizer-cost
The representation **can** express the answer but the **search** can't reach it at
feasible compute.
- 4×4 matmul: restarts grow steeply toward the true rank; no exact even at naive R=64
  on a 4060 (session 6). The GF(2) soft-XOR relaxation dead-ends (only naive 2×2,
  ~80 min/rank at 3×3).
- A compute wall, not an impossibility — a better optimizer / discrete search might
  cross it.

### 8. Extraction  `(probed and CLOSED as a moonshot route in session 9, expGG)`
The **net length-generalizes** but its continuous state is a **smooth manifold**, not
crisp states, so a bit-exact symbolic FSM can't be read off post-hoc (>2-state
mult/div extraction stalls at ~0.94).
- **Verdict:** the wall is **tooling, not fundamental, and hides no non-human
  *function*** — for a regular op the human-readable FSM exists because the
  **function is regular**, independent of the net. `↻ precision (2026-06-09 audit
  §2.4):` the net-side step ("no drift to w30 ⟹ the net's *mechanism* is
  effectively finite-state") is a plausibility argument, not a proof; whether the
  net's internal *procedure* is FSM-like remains exactly as open as the extraction
  gap. Behavioral extraction (on-distribution carry-read + majority vote) beats
  geometric (geometric 0.68@w3; behavioral 0.94@w1, decaying to 0.715@w30 — each
  method's best width) but a **fit-vs-extractability tradeoff** blocks bit-exact
  extraction from a smeared net. For guaranteed exact extraction, **bake
  discreteness in during training** (a sign-STE makes the FSM exact by
  construction) rather than extracting post-hoc.

---

## C. Meta-obstructions to the moonshot specifically

### 9. The two walls + the bridge  `(session 8, expW; sharpened sessions 9–10)`
- **Target-driven search CONVERGES** (rediscovery — this is walls #2, #3).
- **Pure open-endedness DIVERGES** (a noise zoo — distinctness has no notion of
  meaning).
- The moonshot lives in the **narrow band between** = the **bridge**: an intrinsic-
  meaning signal that surfaces structure with no target. The bridge is real,
  **multi-dimensional** (expAA), and **generative** (expDD), but **landscape-gated**
  (expEE) and **ceiling-bound** (every validated signal was built to flag a *known*
  structure-class). See [05_bridge_signal_families.md](05_bridge_signal_families.md).
- `↻ REFINED (session 10):` the **survival** family is the first target-free
  open-endedness in the project to produce **meaning rather than noise**
  (self-replicators emerge from random code — the expW-divergence counterpart). But
  **pure survival SETTLES** (no sustained novelty), and sustained complexity growth
  needs **computation-coupled selection on a composable substrate** (and even then
  the functions are vocabulary-bounded, #6). So the bridge produces meaning from no
  target but not *sustained novelty* without the right substrate + coupling.

### 10. Scale
The genuinely-novel object (a new identity, a new structure class) lives in a **tail
reachable only beyond a single 4060/4090 session**.
- Session 10's GPU run confirmed the prediction precisely: the signals work cleanly
  at scale (2³² and 2¹⁸ uncatalogued CA spaces) but **reliably surface the *known*
  class-4 structure-class** — gliders cleanly in 1D, Life-like rules in 2D — with no
  obviously-novel object. Ceiling holds.

### 11. Novelty-unprovability (epistemic)  `← the moonshot ceiling's other half`
Even if found, **"human-unknown" can be EVIDENCED but never PROVEN** — an irreducible
wall. A find can only ever be "structured + not in the references I know," never
"provably absent from the human catalog." **This is Open Frontier #2.**

---

## Cross-reference: what changed from the session-9 taxonomy

| Wall | Session-9 state | Session-10 refinement |
|------|-----------------|-----------------------|
| #1 Representational | FIX: richer substrate | External memory crosses *reversal* (2/3 seeds). `↻ audit §1.1:` the "fits mul but doesn't len-gen" reading was **wrong** — neither arch learns mul past width ~2; no demonstrated memory benefit on mul. |
| #4 Landscape | one hard wall marked **"FIX: unknown"** | `↻ closure 2026-06-10:` scale moves it **~1–2 orders, heavy-tailed** (evo median ~675 over 3 seeds; old "6238/140×" was an upper-tail draw). QD is **competitive & descriptor-dependent** (rt_span ME 2139 > evo 675), not "the wrong tool" (the "void" verdict was the archive bug). Composable substrate climbs where byte-tape plateaus. |
| #6 Primitive-vocabulary gate | a movable gate among others | `↻ audit §2.3:` **restated** — the closure argument was unsound; the gate is *reachable depth × naming-density*, largely collapsing into #4 + #11. |
| #9 The two walls / bridge | bridge real, multi-dim, generative, landscape-gated | **Survival bridge** produces meaning from no target (vs expW divergence) but **settles**; sustained novelty needs computation-coupled selection on a composable substrate. |
| #11 Novelty-unprovability | flagged | Unchanged, now named as **half of the moonshot ceiling** (with #6). |

## The two unprobed walls (flagged, never cleanly isolated)
- **Ambiguity / relation wall:** operations with *many valid outputs* (relations, not
  functions) — never tested; outcome-verification may absorb it or may break the
  learning signal.
- (The **extraction wall**, #8, was the other flagged-unprobed wall; session 9
  probed and closed it — see above.)
