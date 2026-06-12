# Experiment Catalog

Every experiment, organized by phase and theme. Status codes follow the tracker:
**WORKS / PARTIAL / FAILED / ABANDONED**. "Discovered" always means
*discovered-from-outcome*; the honest floor (given primitives/obs/filters) is in the
per-experiment tracker entries and summarized in [02_methodology.md](02_methodology.md).
Numbers are exact and cross-checked against the result files (re-audited 2026-06-09;
corrections marked "↻ audit" — see [09_fable5_audit.md](09_fable5_audit.md)).

Quick index: **Part I** = the composition story (arithmetic substrate → the recipe →
its breadth). **Part II** = the moonshot / walls+bridge. **Part III** = session-10
GPU + the survival-bridge arc. **Part IV** = interpretability & audit.

---

## Part I — The composition story

### I.A — Arithmetic substrate and the first walls (sessions 1–3)

| Exp | Domain / attack | Method | Result | Status |
|-----|-----------------|--------|--------|--------|
| **expB** progsearch | addition; subtraction | gradient-free GP over a digit-serial register VM | **carry** discovered (exact, provably length-general). Sub: clamped-sub blocks borrow → **primitive-vocabulary gate**; signed VM fixes the output but not the borrow flag | WORKS (add); PARTIAL (sub) |
| **expA** mealy | addition; subtraction | tiny neural Mealy machine (1–8 state dims) + FSM extraction | **carry / borrow** (not add+negate) → exact 2-state FSMs, len-gen 1.000 to w20–30. Excess capacity overfits length **only under single-width training**; mixed-width cures it (d4: 0/5→5/5) | WORKS |
| **expA** mul1 | single-digit × multi-digit | Mealy (needs wide g, ~9 states) | NET discovers multiplicative carry, len-gen 1.000; FSM extraction ~0.94 (smooth manifold) | PARTIAL (extraction) |
| **expMul_full** | full n×n multiplication | Mealy, state dims 4/16/64 | **THE WALL** — not finite-state; can't fit even width 3 at 64 dims. Clean negative | WORKS (negative) |
| **expC** compose | full multiplication | GP loop search + grounding on the extracted carry FSM | discovered `acc += SHL(MULDIGIT(A,Bj), j)`; exact to w12. *Composition external (Python loop)* | WORKS |
| **expA** div1 | single-digit-divisor ÷ (MSB-first) | Mealy, 4 varied attempts | does **not** length-generalize (best 0.69@w20). First negative | PARTIAL/neg |
| **expD** divfixed | why ÷ fails | continuous Mealy, fixed divisor | fixed /7 fails too → not the variable divisor. **Net len-gens ÷d iff d divides the base** (/2,/5 pass; /3,/7 fail); base-12 flips them | WORKS |
| **expD** modm | isolate the cause | predict running prefix mod m, no quotient | m=10 (=base) perfect; m=7,9 collapse → the wall is **non-base modular state maintenance** | WORKS |
| **expD** div_compose | ÷ capstone | GP long-division = repeated subtraction on the borrow FSM | exact to w20 incl. base-coprime divisors. `COMBINE` primitive was the unlock (vocabulary gate again) | WORKS |
| **expD** discrete | exact extraction | sign-STE discrete state (FSM exact by construction) | extraction exact for any #states, but **hard discreteness hurts >2-state fit** (mult/div). Fit-vs-extractability tradeoff | PARTIAL |
| **expE** shared | is +/− one shared mechanism? | one op-conditioned Mealy on + and − | needs d≥2; d=1 gives the bit to addition and sub collapses. **Sharing a network ≠ sharing a mechanism** (refuted a pre-registered hypothesis) | WORKS |
| **expF** unified | one model, 4 ops; curricula | UnifiedMealy d=8; co-train / sequential / blended / dynamic | co-train ≫ sequential (catastrophic forgetting); rehearsal + deficit-weighted dynamic + hybrid ÷ → **1.000 on all ops + chained expressions**. Walls are architectural (order-invariant) | WORKS |
| **expG** controller | internal composition | GRU controller emits+runs a program over an integer VM (trace-supervised) | full mul + any-divisor div, len-gen to w20; crosses both walls. *Imitation, not yet discovered* | WORKS |
| **expG** discover | discover composition from reward | REINFORCE (shaped reward, no traces) | **FAILED** — partial-credit local optimum → collapse. Motivates expJ | FAILED |
| **expH** orders | curriculum order effects | 4 orders, identical arch/init/budget | walls order-invariant; hard op (mult) needs **early** introduction for enough rehearsal exposure; mechanism invariant, geometry contingent | WORKS |
| **expI** repr | let the model choose its code | learned redundant code + carry-free op + recurrent decoder | discovers **carry-save** (column-sum code + 2-state normalizing decoder), len-gen to w20 (2/3 seeds). Code is *continuous* (discrete forcing fails); carry **relocatable not eliminable** | WORKS |
| **expI** basechoice | let the system choose its base | outer search over bases | lands on the covering base; **refines the law to d | base^k** (verified /4 clean via 4|100, /8 partial via 8|1000) | WORKS |

### I.B — The recipe: discover *and* run from outcome (session 4)

| Exp | Domain | Method | Result | Status |
|-----|--------|--------|--------|--------|
| **expJ** mul | full multiplication | exact-filtered self-imitation + loop extraction + cross-width verify + distill | discovered+run from outcome, `[GETDIGIT MULDIGIT SHL ADD_ACC INC_J]`, len-gen 1.000 to w20 | WORKS |
| **expJ** div | long division (data-dependent inner loop) | same, + reactive loop extraction + validity filters | discovered `[GETDIGIT COMBINE SUB_D* STOREQ INC_J]`, crosses the base-coprime wall, len-gen 1.000 to w20 | WORKS |
| **expJ** both | both ops, one model | unified controller, 3 seeds | 3/3 seeds discover identical minimal programs; len-gen 1.000 to **w30**; all divisors incl. base-coprime = 1.000 | WORKS |

### I.C — Discover the primitives, not just the composition (session 5)

| Exp | Domain | Method | Result | Status |
|-----|--------|--------|--------|--------|
| **expM** muldigit | single-digit × = repeated add | expJ recipe on a minimal ALU (no MULDIGIT) | discovered `[GETDIGIT ADD_STEP* SHL ADD_ACC INC_J]`, 4/4 seeds, w30=1.000 | WORKS |
| **expM** add | addition = counting | expJ recipe; only primitive is the digit-wheel successor (TICK) | discovered `[LOADA CARRYTICK* TICK* EMIT]`; **carry = the wheel rollover**, 4/4 seeds, w30=1.000 (needed a no-carry warmup to ignite) | WORKS |
| **expM** tower | multiply grounded on +1 | run the rung-1 mul controller on a VM whose adds are the discovered adder | A×B exact, len-gen to w20, with all arithmetic reduced to the digit successor | WORKS |

### I.D — The efficiency budget as a selector; breadth (sessions 5–8)

| Exp | Domain | Selector / which-algorithm | Result | Status |
|-----|--------|----------------------------|--------|--------|
| **expK** gcd | GCD; Euclid vs subtractive | **step budget** rejects subtractive (explodes on high-ratio pairs) | discovered **Euclid** (mod-reduction), 4/4 seeds, exact to w30 | WORKS |
| **expL** matmul | 2×2 matrix multiply | **multiplication-count (rank) budget** | discovered **Strassen** (rank 7, {−1,0,1}); 6 impossible; naive = rank 8 | WORKS |
| **expN** matmul | 3×3 scale | batched CP + lattice anneal | rediscovers **Laderman** rank 23 (needs ~16× restarts vs the easy 24) | WORKS |
| **expN** ladder | rectangular ⟨2,2,3⟩,⟨2,2,4⟩,⟨2,3,3⟩ | same | hits best-known optima 11/14/15; sharp impossibility boundary below each | WORKS |
| **expN** gf2 / phase2 | GF(2) mod-2; 4×4 reals | soft-XOR relaxation / flat CP | **FAILED/ABANDONED** — 4×4 reals = the wall (no exact even at naive); GF(2) finds only naive 2×2, ~80 min/rank at 3×3 | FAILED/ABANDONED |
| **expO** isqrt | isqrt; binary search vs linear | **step budget** (linear busts the cap by w=3) | discovered **binary search** (square+compare VM), 4/4 seeds, exact to w30. *Cross-width verify selects it; raw self-imitation ignites the naive linear scan* | WORKS |
| **expP** newton | isqrt; division primitives | primitive-dependence | discovered **Newton/Heron** (division VM), 4/4 seeds, exact to w30 | WORKS |
| **expQ** sort | sorting; first non-arithmetic op | adjacent-swap VM | discovered **bubble sort** (memoryless table), exact len-gen train≤5 → len 50. Length-gen also *disambiguates* the one ambiguous memoryless state | WORKS |
| **expR** selection | sorting; min-select primitives | primitive-dependence | discovered **selection sort**, 4/4 seeds, exact to len 50 | WORKS |
| **expS** factor | factorization; first NO-poly op | per-instance √n budget | discovered **√n-bounded trial division**, 4/4 seeds, exact to 1e8. **First complexity-class wall** — discovered method is exponential-in-digits | WORKS |
| **expT** karatsuba | fast integer multiply | rank budget on the polynomial-mult tensor | discovered **Karatsuba** (rank 3); recursive multiplier verified O(n^1.585). Toom-3 rank-5 unreachable (needs rational coeffs) | WORKS |
| **expU** superopt | branchless bit-tricks | exhaustive search + **exhaustive (proof) verification** + width-gen | rediscovered 4 Hacker's-Delight gems (incl. a self-found abs variant); width-gen rejects w=3 coincidences. Signed min/max needs >5 ops (overflow-safe) | WORKS |

---

## Part II — The moonshot: the two walls and the bridge (sessions 8–9)

| Exp | What it attacked | Result | Status |
|-----|------------------|--------|--------|
| **expV** cfhunt | identity-space (escape the rediscovery skeleton) | rediscovers classical π/e continued fractions to 210+ digits; **no novel identity** (small sweep re-collects catalogued formulas). Caught + filtered the rational-PCF false-positive trap (98 → 16 genuine) | PARTIAL/neg |
| **expW** openended | pure open-endedness (distinctness, no target) | **DIVERGES to a noise zoo** — 30,000 distinct functions, reinventing almost nothing. Maps the **second wall**: removing the objective is necessary but catastrophically insufficient | WORKS (as negative-space map) |
| **expX** interesting | the **bridge** — compression sophistication | high linear-complexity AND compressible → surfaces the 2-automatic class, rejects noise. *First proxy was broken (bit-packing artifact); the noise baseline caught it* | WORKS |
| **expY** ca | the bridge on ground truth (cellular automata) | **edge-of-chaos 4c(1−c)** ranks 6/7 class-4 rules in the top ~10% (↻ audit §2.6: the 7 rules are **3 symmetry classes** — 110×4, 54×2, 106 — so 6/7 = 2 of 3 independent objects, 106-family missed), rejects chaos+trivial. *First signal (column-LC) was IC-contaminated; fixed* | WORKS |
| **expZ** invariants | second bridge family — polynomial invariants | recovers known conserved quantities (incl. a nonlinear QRT map's biquadratic) from the map's action alone, rejects the non-integrable baseline + cross-prime flukes. Blind spot: rational invariants | WORKS |
| **expAA** multisignal | is the bridge multi-dimensional? | intersecting compression + nonlinearity + damage-spreading resolves expY's class-4-vs-additive conflation (2/7 → 6/7; same symmetry caveat as expY — effective n = 3 classes). **Bridge is multi-dimensional** | WORKS |
| **expBB** factoradic | alien position-dependent carry | position-dependence is **not** the wall (in-range randmix radix len-gens perfectly); the obstruction is **divisor/radix extrapolation** | WORKS |
| **expCC** ladder | anti-Occam: is carry unique? | the largest correct length-general adder is just **carry re-encoded** — proven by a bisimilar-adder ladder + **Myhill–Nerode**. Upgrades the rediscovery wall to a **theorem** for regular ops | WORKS |
| **expCC** census | (the other route) | **ABANDONED** — blind enumeration of distinct functions explodes before reaching carry's depth. Method-negative, not a result | ABANDONED |
| **expDD** evolve | is the bridge generative? | used as a fitness it **drives** a 2³² CA search to converge on structured objects (vs a random driver → noise). **Bridge is generative.** Honest circularity caveat | WORKS |
| **expEE** logicaldepth/evolve | third bridge family — computational depth, on the program space | depth-driven evolution beats sampling (43 vs 26, single seed; ↻ audit: the random-driver "2" is a reporting artifact — it logs the hash-elite's runtime, not the deepest halter that run evaluated) but **stalls at trivial depth** — deep computation = isolated needles. **Landscape wall** on the exact space the moonshot needs. Signal-craft: edge-of-chaos contaminated by TM space-time sparsity | WORKS |
| **expFF** learnability | "do other walls exist?" | **NEW wall** — a 2-op pseudorandom mixer is **un-learnable from outcome** (all learners at chance) while addition is learnable. The learnability/cryptographic wall | WORKS |
| **expGG** extraction | probe the extraction wall (moonshot-adjacent) | **CLOSES it as a route** — for a regular op the human-readable FSM exists because the *function* is regular (↻ audit §2.4: the net-mechanism inference is plausibility, not proof; the net's internal *procedure* stays open by exactly the extraction gap); behavioral beats geometric (0.68@w3 vs 0.94@w1, decaying to 0.715@w30) but bit-exact post-hoc fails. Hides no non-human *function* | WORKS |

---

## Part III — Session 10: GPU + the survival-bridge arc

### III.A — The three compute-movable walls (RunPod RTX 4090)

| Exp | Wall attacked | Result | Status |
|-----|---------------|--------|--------|
| **gpu_exp3** memory | representational (#1) | **reversal CROSSED** — NTM-lite memory len-gens to w20 exact (**2/3 seeds**; seed 1 = Gumbel optimization failure), baseline collapses past train width. **multiplication HOLDS** — ↻ audit §1.1: the "fits in-distribution (loss→3e-4)" reading was wrong (width-stratified loss, fit at w1 only; w4*/w6* in-distribution = 0.000 both archs; baseline profile identical) — **neither arch learns mul past width ~2** | WORKS (rev); wall holds (mul) |
| **gpu_exp2** qd | landscape (#4) | sampling **49** / plain evolution **6238** deepest halter (n=1, Tmax 8000; 6238 = 78% of cap → "≥140×" is a lower bound, stall unmeasured). **scale moves the wall** (expEE's "stalls at 43" was a low-budget artifact). ↻ audit §1.2: the MAP-Elites **154** is **VOID** — duplicate-index CUDA-scatter archive bug; the log shows the stored best genome doesn't halt on re-run — QD verdict unresolved, fixed re-run queued | WORKS (evo/sampling); ME void |
| **gpu_exp1** novelty | scale / moonshot (#10) | 1D radius-2 (2³²) edge-of-chaos + neural axis: all 3 seeds surface clean **multi-glider class-4** rules (`0x59DABC24`/`0x96402558`/`0xACB27BE8`). Ceiling **holds** — generic class-4, no novel mechanism | WORKS (ceiling holds) |
| **gpu_exp1b** ca2d | scale / moonshot (#10) | 2D Life-like census (all 2¹⁸) validated at full scale after **4 rounds of live signal-craft**; surfaces active class-3/4 rules incl. Conway's birth family. ~18k/9k/3k "structured-but-not-named" rules by threshold. Ceiling holds | WORKS (ceiling holds) |

### III.B — The survival-bridge arc (the "go crazy" gambles)

| Exp | Step in the arc | Result | Status |
|-----|-----------------|--------|--------|
| **gpu_weird_soup** | does survival-selection converge to meaning? | **self-replicators EMERGE from random BFF code, no target** — population compressibility collapses, a quasispecies forms around a **readable conserved copy-loop motif**. Stochastic (~1/3 cleanly, ~2/3 show the condensation signature). The expW-divergence counterpart | WORKS |
| **gpu_weird_lprog** | structure-agnostic signal (learning progress) | rejects trivial+noise cleanly as a *distribution* (class-4 median rank 20, trivial 202); class-4-vs-chaos margin is **fragile/budget-sensitive**. 2,776 residual-novel radius-2 candidates flagged (not claimed novel) | WORKS (partial-positive) |
| **gpu_alife** | does pure survival give *sustained* novelty? | **SETTLES** — finds the minimal replicator and stops (flat complexity over 13k epochs); apparent "open-endedness" was **neutral drift** (sequence turnover ≠ innovation). Replication rigorously proven (rep 0.25 vs ctrl 0.037). `alife_l32` was degenerate (condensation without real replication) | WORKS (bounds survival) |
| **gpu_metabolism** | computation-coupling on the byte-tape | **PLATEAUS** — bootstraps partial NAND (~54%) but the reliable-computation ladder stays **empty** (both logic and intrinsic merit). The landscape wall as a **substrate artifact** | WORKS (negative) |
| **gpu_avida** | composable substrate (NAND stack) | **CLIMBS** — evolution reaches reliably computing **XOR/EQU** (composed multi-NAND circuits, evolved-not-designed) in ~50 gens; at scale 77%/71% of the population. The landscape wall is **substrate-dependent and movable** | WORKS (strong positive) |
| **gpu_avida_oe** | target-free on the composable substrate | edge-of-chaos with **no named target** generatively evolves structured input-dependent functions (merit → 1.0). ↻ audit §1.3: the "addition/XOR exact (gen 50/100)" waypoints have **no surviving artifact** (`match=[]` at every persisted snapshot; 10-entry named suite); the "composite ⟹ no novelty" ceiling argument is unsound (§2.3) — restated as *reachable depth × naming-density* | WORKS (generative); claims corrected |

> **Note on gpu_avida_oe artifacts (updated 2026-06-09, audit §1.3):** the discovery
> claims (a+b, a^b exact) were attributed to **live session stdout** — but the
> surviving nohup stdout logs (`runs_pod/oe_s{1,7}.log`) log every 250 generations
> and show `match=[]` at **every** logged generation (0/250/500) for both seeds,
> with no gen-50/100 lines anywhere; the named suite has only 10 entries. The
> waypoint claims are therefore **unverified, not just thin**, and should not be
> cited. What the artifacts *do* show: target-free evolution of structured functions
> the recognizer could not name. A re-run with per-10-gen match logging + a
> ≥50-entry suite is the queued closure experiment.

---

## Part IV — Interpretability & audit (supporting)

These produced no new capability but the causal account underwriting the
"mechanism invariant / geometry contingent" and "extractability = discreteness +
data-independence" claims.

| Files | What they established |
|-------|-----------------------|
| `interp_unified.py`–`interp_unified6.py` | The unified 4-op model is a **data automaton**: ~3-dim shared scratchpad holds the per-step latent; op-code is a causal function selector; carry/borrow are causal ±1 bits; **mult-carry is entangled with the multiplier** (matched-b ~1.0 vs mismatched ~chance) → the smooth-manifold / extraction-failure explanation |
| `interp_controller.py`, `interp_controller2.py` | The GRU controller is a **control automaton**: hidden state is a program counter (autonomous period-5 mul cycle under constant obs); the div inner loop is a flag-gated fixed-point attractor (count-invariant → len-gen); control FSM extracts exactly; op-routing is asymmetric (mul latches the op, div re-reads it) |
| `interp_orders.py` | Across orders/seeds/budgets: **mechanism is rigidly invariant; geometry (which neurons, what angles) is contingent** — you cannot read "which neuron owns which op" off the weights |
| `audit.py` | Session-1 verification: loads saved checkpoints and *measures only* (no training); confirmed add/sub/mul1 numbers, the 200/200 exhaustive carry-FSM check, and that no division checkpoint exists |
