# Bridge-Signal Families

The **bridge** is the project's name for an **intrinsic "interestingness" signal that
surfaces meaningful structure with no target** — the only escape from the rediscovery
engine (target-driven search → known optimum) on one side and the noise zoo (pure
distinctness → divergence) on the other. See [03_wall_taxonomy.md](03_wall_taxonomy.md)
#9 for the framing.

Six families were validated. Each is genuinely independent (they measure different
things) and each was validated against ground truth before use. **The shared
ceiling:** every one of them was *built to flag a known notion of structure*, so all
six **surface known structure-classes** and none has produced a human-unknown object.
The shared craft lesson: these signals are **finicky and representation-dependent** —
the failures below are mostly cases where the signal measured an artifact instead of
the intended structure (see the signal-craft bugs in [02_methodology.md](02_methodology.md) §8).

---

## 1. Compression sophistication (Bennett-logical-depth flavor)

- **Idea:** interesting = **weak models fail AND a strong general compressor
  succeeds**. A sequence is sophisticated iff it has high linear complexity (no short
  LFSR / not periodic — Berlekamp–Massey) **and** zlib compresses it (structured, not
  pseudo-random). Score `= LC_norm × (1 − zlib_ratio)`.
- **Object space:** binary sequences from tiny rules (expX).
- **Surfaces:** the **2-automatic structured class** (Thue–Morse / Rudin–Shapiro
  motifs), cleanly separated from noise (rejected at −0.086) and from simple
  (low-LC, low score).
- **Misses / fails:** the **first proxy was broken** — raw `1 − zlib_ratio` with bits
  stored one-per-byte made a random baseline score 0.76 (zlib fake-compresses the
  always-zero padding bits) and ranked the *simple*, not the sophisticated. Fixed by
  8-bits-per-byte packing + LC weighting. **A noise baseline is mandatory** for this
  family — it is exactly what caught the bug.
- **Files:** `expX_interesting.py`.

## 2. Edge-of-chaos (the validated operational workhorse)

- **Idea:** interesting = **intermediate compressibility**. Score `= 4c(1−c)` where
  `c` = compression ratio of the space-time pattern; peaks at `c = 0.5`
  (class-1/2 too compressible → 0; class-3 chaos incompressible → 0; class-4 in
  between → 1). This is the operational form of family #1 that actually works on
  dynamical objects, and it is the signal reused most across the project (expY,
  expAA, expDD, and the GPU 1D/2D hunts).
- **Object space:** cellular automata (256 elementary; 2³² radius-2; 2¹⁸ Life-like 2D)
  and, target-free, evolved I/O maps (gpu_avida_oe).
- **Surfaces:** **Wolfram class-4** rules — 6/7 in the top ~10% on the elementary-CA
  ground truth (↻ 2026-06-09 audit §2.6: the 7 rules are **3 symmetry classes** —
  110×4, 54×2, 106×1 — so this is 2 of 3 independent objects, 106-family missed);
  clean multi-glider rules in the 2³² space at GPU scale (s1 inspected live; the
  s3 top-set re-renders **mixed** — audit §1.4, `runs/audit_exp1_s*_grid.png`).
- **Misses / fails:**
  - Cannot split **class-4 from additive/Sierpinski-fractal** rules — both are
    intermediate-c. (Resolved by intersecting with other axes — family handoff to #3
    of the multi-signal stack, expAA.)
  - **IC contamination** (the original expY signal `(1−c) × column-LC`): a shift rule
    copies the random initial condition down the center column → fake-high LC →
    periodic class-2 rules rank top. Fixed by switching to `4c(1−c)` on the full
    space-time.
  - **TM space-time sparsity** (expEE): the head touches one cell/step, so the diagram
    is ~blank and everything sits at c ≈ 0.09 — the structure term does not transfer
    to Turing-machine space-time. Needs a TM-appropriate measure (head-track / output
    tape).
  - **2D needs the damage band with both cuts** + activity/quiescent gates (gpu_exp1b,
    4 rounds of live signal-craft) — the raw signal ranked global-flashing noise and
    static debris on top until calibrated by *rendering* the survivors.
- **The damage axis** (used alongside `4c(1−c)`): a 1-cell perturbation spread,
  banded ~[0.10, 0.32] (1D) / centered ~0.12 (2D) — high-cut kills chaos
  (saturates ~0.5), low-cut kills frozen healers (~0).
- **Files:** `expY_ca.py`, `expAA_multisignal.py`, `expDD_evolve.py`,
  `gpu_exp1_novelty.py`, `gpu_exp1b_ca2d.py`, `gpu_avida_oe.py`.

> The exact formulas, for reference:
> 1D scale hunt: `interest = 4c(1−c)·exp(−((dmg−0.18)/0.12)²)`, then
> `novelty = interest·(1 + clamp((c − neural_bpc)/c, 0, 1))` (the **neural tilt** — a
> shared next-row predictor that can't memorize per-rule maps, so it only rewards
> rule-agnostic higher-order structure zlib misses).
> 2D: `4c(1−c)·exp(−((dmg−0.12)/0.09)²)`.
> Target-free Avida: `merit = (dep_a+dep_b)·(0.3 + 4c(1−c))·(1 + dep_a·dep_b)`.

## 3. Polynomial invariants / self-consistency (conserved quantities)

- **Idea:** a dynamical map `f` over a finite phase space `(Z/p)²` is interesting iff
  it admits a **low-degree polynomial conserved quantity** `φ` with `φ(f(s)) = φ(s)`
  for all `s`. Target-free (never specify which φ), and exact pushed to a **proof**:
  enumerate the entire phase space and solve the GF(p) null space of
  `[φ(f(s)) − φ(s) = 0]`. Signal = dimension of the non-constant invariant space.
- **Object space:** polynomial maps over `(Z/p)²` (expZ).
- **Surfaces:** **integrable maps** — recovers known invariants (rotation → x²+y²,
  shear → y, and the genuinely-**nonlinear QRT map's biquadratic H** from the map's
  action alone), rejects the non-integrable baseline (Hénon, random) with **zero
  cross-prime flukes** (p=23 and p=101).
- **Misses / fails:** a **polynomial-only blind spot** — genuinely-integrable
  nonlinear maps (McMillan/Lyness type) carry **rational** invariants the
  polynomial null-space cannot see, so the target-free scan surfaced only the
  trivially-integrable **linear** sub-family. (Clean next step: a rational-invariant
  search with an unknown numerator and denominator.) Linear maps are
  non-discriminating (every SL2 map preserves a quadratic form).
- **Why it matters:** structurally **independent** of compression (it measures a
  conserved structure of the *dynamics*, not statistical regularity of an *output*) —
  proves the bridge band is not an artifact of one signal.
- **Files:** `expZ_invariants.py`.

## 4. Computational depth (Bennett logical depth / Busy-Beaver)

- **Idea:** interesting = **a short rule that computes a long time before producing
  structure**. Score `= log(runtime) × 4c(1−c)` on the space-time of an n-state
  Turing machine run from a blank tape.
- **Object space:** n-state 2-symbol Turing machines (expEE).
- **Surfaces:** **deeper computation than sampling can reach** — depth-driven
  evolution beats sampling (best halter 43 vs 26, single seed, monotone climb;
  ↻ 2026-06-09 audit: the random-driver "2" is a reporting artifact — it logs the
  hash-elite's runtime, not the deepest halter that run evaluated). The depth
  *direction* is real.
- **Misses / fails:** **stalls at trivial depth** (43 vs BB(5) = 47,176,870) because
  deep computation lives at **isolated, mutation-fragile needles** — the **landscape
  wall** (#4). Also the structure term is contaminated by TM space-time sparsity (see
  family #2). This is the family that revealed the generative bridge is
  **landscape-gated**: it works on the smooth CA basin and barely on the rugged
  program space. *(Session-10 refinement, corrected 2026-06-09: scale moves this
  wall ≥140× [lower bound — Tmax-capped, n=1]; the QD comparison is **void** per
  the archive-corruption bug — see [03_wall_taxonomy.md](03_wall_taxonomy.md) #4
  and 09_fable5_audit.md §1.2.)*
- **Files:** `expEE_logicaldepth.py`, `expEE_evolve.py`, `gpu_exp2_qd.py` (the
  scaled MAP-Elites-vs-evolution-vs-sampling follow-up).

## 5. Learning progress (Schmidhuber compression-progress)

- **Idea:** **structure-agnostic** interestingness — interesting = a predictor's loss
  **drops a lot with a little training** (learnable-with-effort), naming **no
  structure type at all**. `lp = bpc_after_warmup − bpc_best`. This is the most
  principled attack on the project's ceiling ("every signal is pointed at a *known*
  structure").
- **Object space:** cellular automata (validated on the 256 elementary; hunted in
  2³²), via a tiny per-object predictor too small to memorize noise (gpu_weird_lprog).
- **Surfaces:** **rejects trivial AND noise** cleanly as a *distribution* (class-4
  median rank 20, chaos 25, trivial 202 of 256) — which is the genuinely hard half of
  "interestingness without a target." 2,776 residual-novel radius-2 candidates flagged.
- **Misses / fails:** **fragile at the class-4-vs-chaos boundary** — some class-3
  chaos has exploitable *early* determinism, so at larger orbit / longer training,
  chaos out-ranks class-4. It separates interesting-from-trivial+noise as a
  distribution, **never rule-by-rule**. (The closest the project came to
  structure-agnostic; still not enough to *recognize* novelty — see Frontier #2.)
- **Files:** `gpu_weird_lprog.py`.

## 6. Survival-in-an-environment (artificial life)

- **Idea:** no explicit interestingness signal at all — let **selection-by-survival**
  (and, later, computation-coupled merit) decide what persists. The bridge as
  *ecology*, not as a scored metric.
- **Object space:** self-modifying byte-tape programs (Brainfuck-with-two-heads, BFF);
  later a composable NAND-stack VM.
- **Surfaces / the arc (session 10):**
  - **Pure survival → MEANING from no target:** self-replicators emerge from random
    code (a quasispecies around a readable conserved copy-loop) — the **opposite** of
    expW's distinctness-driven divergence to noise. This is the single most
    moonshot-relevant *qualitative* shift in the project.
  - **Computation-coupled on a composable substrate → complexity growth:** evolution
    climbs from random programs to reliably computing **XOR/EQU**.
  - With the **edge-of-chaos** signal as target-free merit on that substrate →
    generatively evolves structured input-dependent functions (↻ 2026-06-09 audit
    §1.3: the "addition/XOR exact" waypoints are unsupported by surviving artifacts
    — `match=[]` at every persisted snapshot; the persisted winners are functions
    the 10-entry recognizer could not name; re-run queued).
- **Misses / fails:**
  - **Pure survival SETTLES** — finds the *minimal* replicator and stops; no pressure
    toward more complexity. (And the naive "open-endedness" reading was **neutral
    drift** — sequence turnover is not innovation; use functional metrics.)
  - **Computation-coupling PLATEAUS on the byte-tape** (partial NAND, empty ladder) —
    the landscape wall as a substrate artifact; needs a composable substrate.
  - Even on the composable substrate, the search reaches only **shallow
    compositions**, which for standard vocabularies are densely named (↻ 2026-06-09
    audit §2.3: the original "bounded to composites of named ops" closure argument
    was unsound; see the revised Frontier #1).
  - **Emergence is stochastic** (~1/3 of runs cleanly; the automated single-pair
    replication test under-counts vs the population-level zlib-collapse + conserved-
    motif evidence).
- **Files:** `gpu_weird_soup.py`, `gpu_alife.py`, `gpu_metabolism.py`,
  `gpu_avida.py`, `gpu_avida_oe.py`.

---

## Cross-cutting summary

| Family | Surfaces | Where it fails / its blind spot | Independent of |
|--------|----------|---------------------------------|----------------|
| 1 Compression sophistication | automatic / structured binary sequences | bit-packing artifact; rewards *simple* if mis-formed (noise baseline catches it) | — (the prototype) |
| 2 Edge-of-chaos | class-4 CAs, gliders | can't split class-4 from additive-fractal; IC contamination; TM sparsity; 2D needs activity gates | the workhorse; reused everywhere |
| 3 Polynomial invariants | integrable maps (incl. nonlinear QRT) | polynomial-only — misses rational invariants; linear maps non-discriminating | compression (measures dynamics, not output) |
| 4 Computational depth | deeper-than-sampling computation | stalls at needles (landscape wall); TM sparsity | a different axis (runtime, not pattern) |
| 5 Learning progress | trivial+noise rejection (structure-agnostic) | fragile at class-4-vs-chaos; distributional only | structure-agnostic by design |
| 6 Survival-in-an-environment | meaning from no target (replicators, evolved XOR/EQU) | settles; byte-tape plateaus; vocabulary-bounded | not a scored metric — ecology |

**The unifying ceiling:** families 1–5 were each *designed to flag a known notion of
structure*, so they resurface known classes by construction. Family 6 is the one that
produced a qualitatively new outcome (meaning from no target) and is therefore the
**most promising direction**, but it too is **vocabulary-bounded** and produced no
*sustained* novelty. The two things missing, as restated by the 2026-06-09 audit: a way to reach
**deep-but-structured composites** (the shallow regions of standard vocabularies are
densely named — the original "closure escapes the named operations" framing was
unsound), and an operational way to **recognize** structure that was *not* pre-named
(which is partly epistemically impossible — novelty can be evidenced, not proven).
