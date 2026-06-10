# mathlab — Consolidation (start here)

This directory is the project's hand-off document, written at the consolidation
point after 10 sessions. It is meant to be read **on its own**: a successor
(Fable 5) should be able to understand the project's state, methodology, and
accumulated findings from these documents, with the original code and
`TRACKER.md` available as reference but not required for understanding.

The discipline that ran the experiments runs this writeup: no overclaiming,
discovered separated from scaffolded, honest about scope and ceilings. This is
documentation, not a sales pitch.

---

## What the project is (one paragraph)

The goal (`PROMPT.md`): a small neural network that, given math problems,
**discovers its own internal algorithms** for solving them, and from which we can
**extract** those algorithms and verify them exactly. The organizing instrument
is **length generalization as the exact "real algorithm vs lookup table" test** —
train on short inputs, test on long ones, with **exact integer eval and no partial
credit**. A memorized table collapses on long inputs; a genuine procedure stays at
1.000. The far goal ("moonshot") was a procedure humans haven't found; the project
was structured so each stage produces something useful even if the moonshot does
not materialize. It did not. What materialized instead are the two contributions
below.

---

## The two contributions (stated cleanly and honestly)

**1. The composition story — a reliable recipe for discovering known efficient
algorithms from outcome alone.**
Outcome-driven discovery (exact-filtered self-imitation) + exact verification +
an efficiency budget reliably finds the known efficient algorithm across many
domains and primitive vocabularies: addition, subtraction, multiplication,
division; GCD → Euclid; isqrt → binary-search *or* Newton (by primitives);
sorting → bubble *or* selection (by primitives); matrix multiply →
Strassen/Laderman; integer multiply → Karatsuba; factorization → √n-bounded
trial division; the arithmetic **primitives themselves** rebuilt from counting
(addition from the digit-successor, multiply from addition); and emergent
computation (XOR/EQU) evolved on a composable substrate. Every result is verified
exactly and length-generalizes. The honest framing (proven as a *theorem* in
session 9, see below): for these targets this recipe is a **rediscovery engine by
construction** — "correct + optimal-under-a-budget" is exactly the criterion by
which humans already found the canonical algorithm.

**2. The walls + bridge framework — a map of why the moonshot did not fall, and
the one escape route.**
The project mapped **five hard walls** (representational, complexity-class,
convergence-by-theorem for regular ops, landscape, learnability/cryptographic),
**contingent gates** (primitive-vocabulary, optimizer-cost, extraction), and
**moonshot meta-walls** (the two walls, scale, novelty-unprovability). The one
escape from rediscovery — an **intrinsic "interestingness" signal** (the *bridge*)
that surfaces meaning with no target — was validated across **six signal families**
(compression sophistication, edge-of-chaos, polynomial invariants, computational
depth, learning progress, survival-in-an-environment). The bridge is real,
**multi-dimensional**, and **generative** (it can drive a search, not just filter),
but every validated signal was *built to flag a known notion of structure*, so it
resurfaces known structure-classes. The moonshot ceiling — **restated by the
2026-06-09 audit** ([09_fable5_audit.md](09_fable5_audit.md) §2.3) — is **reachable
depth × naming-density of the shallow composition region** plus the **unsolved
novelty-recognition problem** (the original "primitive-vocabulary gate" closure
argument was unsound).

---

## The two open frontiers (what comes next)

- **Frontier 1 (revised by the 2026-06-09 audit).** Originally "primitive design for
  non-named operations"; restated: reach **deep-but-structured,
  recognizable-as-novel** objects. The shallow composition layer of every substrate
  tried (digit VMs, register VMs, bilinear tensors, word-ops, BFF byte-tape,
  NAND-stack, cross-bit stack) is densely **named**, and un-namedness alone is
  trivially manufacturable. Better-posed levers: encoding freedom for regular ops
  (carry-save precedent, theorem-licensed), depth-while-structured search, the expV
  identity tail. See the revised Frontier 1 box in 07.
- **Novelty recognition.** Operationally distinguish "genuinely unfamiliar
  structure" from "known but unrecognized." The project's standing position:
  novelty is **evidenceable but not provable**. Every validated signal was pointed
  at a *named* structure-class; the closest to structure-agnostic (learning
  progress) is fragile at the class-4/chaos boundary.

See [07_open_frontiers.md](07_open_frontiers.md) for the full state of each.

---

## How to read this directory

| Doc | What it covers |
|-----|----------------|
| [01_project_arc.md](01_project_arc.md) | The full session-by-session narrative; the two contributions in context. |
| [02_methodology.md](02_methodology.md) | The recipe (exact-filtered self-imitation + extraction + cross-width verify + distill), the discipline rules, the validity filters per experiment, documented failure modes and signal-craft bugs. |
| [03_wall_taxonomy.md](03_wall_taxonomy.md) | The refined wall taxonomy in its current (post-session-10) form. **Read this to understand why the moonshot has not fallen.** |
| [04_experiment_catalog.md](04_experiment_catalog.md) | Every experiment (expA–expGG + the session-10 GPU and weird-gamble runs), organized by what it attacked. |
| [05_bridge_signal_families.md](05_bridge_signal_families.md) | The six bridge-signal families: what each surfaces, what each misses, where each fails. |
| [06_code_map.md](06_code_map.md) | Code organization, dependencies, how to run, configs that matter, where results live, the orchestration scripts. |
| [07_open_frontiers.md](07_open_frontiers.md) | The two frontiers: what is known, what prior attempts established, where they hit their limits. |
| [08_what_not_to_redo.md](08_what_not_to_redo.md) | Failed approaches, intractable directions, things abandoned per no-grind, the signal-craft bugs to not re-introduce. |
| [09_fable5_audit.md](09_fable5_audit.md) | **Successor audit (2026-06-09).** Two evidence-contradicted headline claims (exp3-mul "fits", the QD comparison), one unsupported discovery claim (avida_oe a+b), scope corrections to walls #3/#5/#6, a reframing of Frontier 1. Docs 01–08 carry the corrections in place (marked "↻ audit"); 09 holds the full detail + the verification ledger. |

**Suggested reading order for a successor aiming at the frontiers:**
README → `03_wall_taxonomy` → `07_open_frontiers` → `02_methodology` →
`05_bridge_signal_families` → `08_what_not_to_redo`. Use `04` and `06` as
reference when you need the detail of a specific experiment or how to run it.

---

## Ground rules that still hold (from `.claude/RULES.md`)

- **TRACKER.md is sacred** — every experiment, including failures, gets an entry.
- **Eval is exact** — `2+3 == 5`, no partial credit. Wanting to relax this means an
  approach isn't actually working.
- **Abandonment is first-class** — if you've tuned the same core approach >3 times
  without signal, abandon it; try something structurally different.
- **Be weird** — ask the published-paper answer, then try a stranger one that fits
  the small/narrow/exact-verification constraints.
- **Honesty over metrics** — separate *discovered-from-outcome* from *scaffolded*;
  say "ambiguous" when it is; do not manufacture novelty claims.

---

## Status at consolidation

- Hardware: results were produced on an RTX 4060 (local) and one ~$2 RunPod RTX
  4090 session (session 10). Env: conda `mathlab`, py3.11, torch 2.6 cu124 (local) /
  2.4.1 cu124 (pod).
- The moonshot is **open**; its non-arrival is **over-determined** and mapped.
- All result numbers in these documents were cross-checked against the result
  files (`runs/`, `runs_pod/`); where session 10's GPU campaign finished work that
  `TRACKER.md` had marked "pending," the resolved numbers are recorded here and in
  the final tracker entry. See `01_project_arc.md` §Session 10 and
  `08_what_not_to_redo.md` for the specific reconciliations.
- **2026-06-09:** a successor audit re-verified the load-bearing claims against
  code + artifacts and corrected three *readings* the number-check had missed
  (exp3-mul "fits in-distribution", the QD comparison, avida_oe's a+b/a^b
  waypoints) plus three wall scopes (#3 encoding, #5 learner class, #6 closure).
  Corrections are applied in place across 01–08, marked "↻ audit"; the full
  findings + verification ledger are in `09_fable5_audit.md`.
