# What NOT to Redo

Failed approaches, intractable directions, things abandoned per the no-grind rule, and
the signal-craft bugs that were caught. Each entry says **what failed**, **why**, and
**what to do instead**. The point of the no-grind rule: these are recorded so they are
not rediscovered on paid time or paid context.

---

## A. Methods that failed (use the alternative instead)

### REINFORCE / policy-gradient for composition discovery
- **What:** discover the composed program from a scalar reward (digit accuracy + exact
  bonus), no traces (`expG_discover.py`).
- **Why it failed:** the shaped `0.6·digit_match` reward is a **partial-credit trap**
  (rewards a wrong program that gets some digits right); policy-gradient smears credit
  across the episode; negative advantages destabilize → climbs to a ~0.225 local
  optimum then **collapses** to 0.000.
- **Instead:** **exact-filtered self-imitation** (`expJ_selfdiscover.py`) — binary
  exact filter, positive-only imitation, cross-width verify. This is the project's
  central working recipe.

### Hard discrete state (sign-STE) for >2-state ops
- **What:** force the recurrent state onto {−1,+1}^d to get exact FSM extraction for
  mult/div (`expD_discrete.py`).
- **Why it failed:** hard discreteness **hurts fit** for >2-state algorithms — the net
  *prefers* the smooth manifold (annealing confirms: the soft fit is destroyed by
  hardening). Extraction becomes exact but accuracy drops (mul1 0.74 vs continuous 1.0).
- **Instead:** for guaranteed exact extraction, **bake discreteness in during
  training** and *accept the fit cost*; do **not** try to post-hoc bit-exact-extract
  from a smeared net (expGG confirmed the fit-vs-extractability tradeoff is real, and
  that the smeared net hides no non-human procedure anyway).

### Discrete carry-save code via gradient descent
- **What:** train a clean *discrete* redundant (carry-save) digit code (expI).
- **Why it failed:** **4 different discrete-bottleneck trainings** (straight-through
  hard-from-start, soft-anneal-to-0.1, soft-warmup+ST, Gumbel) all failed to fit; only
  the *soft continuous* code fit. GD strongly prefers the smooth redundant manifold.
- **Instead:** accept that the discovered carry-save code is continuous (it still
  length-generalizes); don't grind on forcing it discrete.

### End-to-end neural division by a base-coprime divisor
- **What:** learn ÷7, ÷3 etc. directly with a digit-serial net (expA_div1, expD).
- **Why it failed:** the per-step state is a **mod by a non-base modulus**, which the
  net cannot hold over length (confirmed by the pure `mod m` isolation). Not a tuning
  problem — a representational fact.
- **Instead:** **compose** division from repeated subtraction (expD_div_compose) or use
  an **exact integer remainder register** (expJ/expG) — both cross the wall.

### QD (MAP-Elites) for raw Busy-Beaver depth — verdict VOID, do not cite (↻ audit §1.2)
- **What:** use quality-diversity to cross the landscape wall (`gpu_exp2_qd.py`).
- **What actually happened:** the run's MAP-Elites arm is **invalid** — `insert()`
  scatters with duplicate cell indices (non-deterministic on CUDA), so a cell's
  archived fitness and genome can come from *different* machines; the run's own log
  shows the corruption fired (archived best 154; the stored genome re-runs to
  runtime=8000, non-halting). Single seed; descriptor and archive-capacity
  confounds untested. The previous "QD is ~40× worse / the wrong tool" conclusion
  is **withdrawn**.
- **What still stands:** plain evolution 6238 vs sampling 49 at matched budget —
  scale moves the wall **≥140×** (lower bound: 6238 is at 78% of the Tmax-8000
  detection cap and the planned high-Tmax verify pass never ran).
- **Do not redo:** the *buggy insert pattern* (duplicate-index scatter for
  archives/elites) — and never report an archived winner without re-executing its
  stored genome. A fixed 3-seed re-run is the queued closure experiment.

### Memory-augmented net for multiplication length-gen
- **What:** crack full multiplication's length-gen with an NTM-lite tape
  (`gpu_exp3_memory.py --op mul`).
- **Why it failed (↻ corrected, audit §1.1):** neither memory nor baseline learns
  mul past width ~2 — the loss is width-stratified (fit at w1; median CE 0.97–1.65
  at w3–6; identical profile for both architectures) and exact accuracy at the
  **in-distribution** widths 4/6 is 0.000 for every completed seed. The session-10
  "fits in-distribution (loss→3e-4) but memorized the widths" reading was a misread
  of the oscillating loss tail; no fit/len-gen dissociation exists in this data.
- **Instead:** memory **does** crack reversal (a stack-op, exact to w20, 2/3 seeds).
  For multiplication, the discovered composition (expJ) is the route, not a flat
  memory-net trained from outcome.

### Computation-coupled evolution on the BFF byte-tape
- **What:** Avida-style merit (reward I/O computation) on self-modifying byte-tape
  programs (`gpu_metabolism.py`).
- **Why it failed:** **plateaus** — bootstraps partial NAND (~54%) but the
  reliable-computation ladder stays **empty** (both logic and intrinsic merit). No
  smooth fitness path from partial → reliable → composed on a byte-tape (the landscape
  wall as a substrate artifact).
- **Instead:** use a **composable substrate** (NAND-complete stack, `gpu_avida.py`) —
  it climbs to XOR/EQU in ~50 generations.

### Pure survival-selection for *sustained* novelty
- **What:** run the primordial soup long and expect open-ended complexity growth
  (`gpu_alife.py`).
- **Why it failed:** **settles** into the minimal replicator and stops (flat complexity
  over 13k epochs); the apparent "open-endedness" was **neutral drift** (sequence
  turnover, not innovation).
- **Instead:** couple survival to computation **on a composable substrate**; and
  **measure open-endedness with functional metrics** (conserved-motif content,
  computational depth), never sequence diversity.

---

## B. Intractable / walled directions (a *different* mechanism is needed, not more tuning)

### 4×4 real matrix multiplication via flat CP + lattice
- **Walled (expN_phase2):** no exact integer decomposition even at **naive R=64** at
  feasible 4060 compute; the sub-Strassen R=49 is a **recursive (block) structure a
  flat tensor search cannot represent**. Needs a recursive/block search — a different
  mechanism, not "scaling this method." (Optimizer-cost wall #7 + representational wall
  #1.)

### GF(2) mod-2 matmul via soft-XOR relaxation
- **Walled (expN_gf2):** finds only **naive** 2×2, never sub-naive, and is
  pathologically slow (~80 min/rank at 3×3). The famous AlphaTensor mod-2 results are
  **not reachable** by this differentiable approach. Needs a discrete/SAT or
  straight-through binary optimizer.

### Toom-3 (rank-5 polynomial multiply) via {−1,0,1} lattice
- **Walled (expT):** the {−1,0,1} integer lattice reaches rank **6**, not the
  field-optimal **5** — Toom-3 needs **rational** coefficients (eval at 2, interpolate
  with /2, /6) the lattice cannot express. A genuine coefficient-restriction boundary.

### Blind program census for anti-Occam
- **Abandoned as intractable (expCC_census):** the space of distinct
  small-arithmetic functions **explodes** before reaching carry's depth-3 DAG (base5
  gen2 → 3895 funcs capped; base2 gen3 caps at 300k without enumerating carry).
  "Enumerate all programs" is the wrong tool — useless intermediate functions dwarf the
  useful ones.
- **Instead:** the **constructive ladder + the Myhill–Nerode theorem** (expCC_ladder)
  is the right instrument — it *proves* the result (largest correct adder = carry
  re-encoded) without enumeration.

### Small-sweep identity hunt
- **Negative (expV):** a small continued-fraction sweep only **re-collects the
  catalogued** π/e formulas; the genuinely-new identities live in the tail (higher
  degree, larger coefficients, under-explored constants) reachable only with much
  larger search. The pivot is right (identity space has no convergent optimum, so it
  *can* harbor the unknown) but needs scale. **And:** the **rational-PCF false-positive
  trap** (a PCF converging to a rational matches *every* constant at once via the
  value-independent Möbius triviality) must be filtered — it collapsed "98 identities"
  to 16 genuine. Use the reject-rational filter; force the pure-Python mpmath backend
  (gmpy2 segfaults on pathological PCFs).

### Signed branchless min/max in ≤5 ops (superopt)
- **Negative (expU):** the famous one-line sign-trick min/max is **overflow-incorrect**
  even within w=3, so it never matches the target; the genuinely width-general
  overflow-safe min/max needs **>5 ops** in that op set. A clean finding that branchless
  min/max is harder than the popular one-liners suggest — not a bug to chase.

---

## C. Signal-craft bugs caught (do NOT re-introduce these)

These are the bridge-signal bugs caught by a noise baseline or by rendering the
survivors. They are the single most reusable warning for the frontiers — every one
made a signal measure an *artifact* instead of the intended structure. (Full detail in
[02_methodology.md](02_methodology.md) §8.)

1. **The residual-novelty score that rewarded maximal compressibility** (GPU prep): a
   *pure subtraction* `general_bpc − named_bpc` ranks frozen/blinking rules on top, not
   complex ones. Use the validated intermediate-compressibility `4c(1−c) ×
   damage-band` finder (the damage band needs **both** a high-cut for chaos *and* a
   low-cut for frozen healers).
2. **The IC-contaminated column-LC** (expY): `(1−c) × center-column-LC` lets a shift
   rule copy the random initial condition down the column → fake-high linear
   complexity → periodic rules rank top, *and* `(1−c)` rewards maximal compressibility.
   Use `4c(1−c)` on the full space-time.
3. **The bit-packing artifact** (expX): storing bits one-per-byte makes zlib
   fake-compress everything (the always-zero padding bits) → a random noise baseline
   scores 0.76. Pack 8 bits/byte and weight by linear complexity. **A noise baseline is
   mandatory.**
4. **TM space-time sparsity contamination** (expEE): the head touches one cell/step, so
   the diagram is ~blank and everything compresses to c≈0.09 — edge-of-chaos does not
   transfer to Turing-machine space-time. Use a TM-appropriate structure measure
   (head-track / output tape).
5. **Sequence diversity as a false innovation metric** (alife): turnover stayed ~1.0
   and the auto-metric printed "OPEN-ENDED," but it was **neutral drift** of the payload
   around a conserved functional core. Use functional-complexity metrics
   (conserved-motif content, ops-count, depth), never sequence churn.

Plus the meta-lesson behind all five: **these intrinsic signals are finicky and
representation-dependent; "looks complex by a metric" ≠ "is dynamically complex" — you
must render and look** (the 2D census needed 4 rounds of this on paid GPU time).

---

## D. Reconciliations with the tracker (gaps filled at consolidation; superseded in
part by the 2026-06-09 audit — [09_fable5_audit.md](09_fable5_audit.md) takes
precedence where they overlap)

Cross-checking the result files against `TRACKER.md`'s last GPU entries (which were
written mid-campaign) surfaced these. None contradicts a headline; they tighten the
honest record:

- **exp1 seeds 2, 3 were marked "PENDING"** — they **completed**. Top survivors:
  s1 `0x59DABC24` (interest 1.370), s2 `0x96402558` (1.656), s3 `0xACB27BE8` (1.524).
  ↻ audit §1.4: the "all clean class-4" verdict here was issued **without renders**;
  re-rendering (`runs/audit_exp1_s*_grid.png`) shows s2 clean class-4 but s3
  **mixed** (top survivor plausibly class-4-ish; several of its top-8 are dense
  chaotic textures). Ceiling conclusion unaffected. (Beware
  `runs_pod/exp1_r2_s3.log` — a 224-byte red herring; the real run is in
  `runs_pod/runs/`.)
- **QD: "154 vs 73"** — both real. The MAP-Elites run plateaued at depth **73** for
  gens 64–191 then jumped to **154** at gen 192 (final). A script comment captured the
  plateau (73); the tracker body's **154** is the correct final value. Verdict
  unchanged: sampling 49 / ME 154 / evolution 6238 — the landscape wall holds for QD,
  scale (evolution) wins.
- **exp3 multiplication** — 5/6 seeds completed (memory seed 2 interrupted, no final
  summary table); **all completed seeds len-gen 0.000** at all widths, memory and
  baseline alike; the missing seed would not change it. ↻ audit §1.1: the
  parenthetical this entry used to carry ("fits in-distribution, does not
  length-generalize") was **wrong** — neither arch fits past width ~2; w4*/w6*
  in-distribution exact = 0.000 too.
- **exp3 reversal** — the "memory crosses the wall" result is **seed-dependent: 2/3
  seeds perfect, seed 1 collapses** (Gumbel optimization variance), mean 0.667. State
  it as 2/3, not a uniform accuracy.
- **Primordial soup emergence is stochastic** — roughly 1/3 of runs emerge cleanly
  (em4: confirmed copy-loop; em3: condensation without a confirmed single-pair
  replicator; em2: null). `alife_lowmut` is a strong positive (rep 0.25 vs ctrl 0.037);
  `alife_l32` was degenerate. The automated single-pair replication test **under-counts**
  vs the population-level zlib-collapse + conserved-motif evidence — the tracker already
  flagged this ("the test read 'weak' — too strict").
- **gpu_avida_oe artifacts are thin** — the a+b/a^b discovery claims came from **live
  stdout**; the persisted `oe_log.json` (seeds 1, 7) has 3 coarse snapshots with the
  `match` field unpopulated. Real result, lighter on-disk evidence. Re-run with finer
  snapshots if revisiting.

These are recorded in the final `TRACKER.md` entry as well.
