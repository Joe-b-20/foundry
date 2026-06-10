# mathlab — Architecture (locked-down reference)

This documents the arithmetic architecture as of session 2. It is the stable base we build
on. The headline limitation it makes explicit — **composition is external (Python glue), not
internal to the model** — is what motivates the next experiment (expG, the neural controller).

## 1. Philosophy / the one metric

The whole project is organized around a single binary test:

> **Length generalization is the exact test of "real algorithm vs lookup table."**
> Train on short numbers (≤ width 3–5), test on long ones (to width 20+). Eval is **exact
> integer match** — no partial credit. A memorized table collapses on long inputs; a genuine
> procedure stays at 1.000.

Two design pressures follow from the goal ("discover a procedure AND extract it"):
- **Discovery** — the model finds the procedure; we never supervise the human algorithm, only
  (input → answer) digit by digit.
- **Extraction** — small, discrete, inspectable state, so the learned procedure reads out as a
  finite-state machine or a short program we can verify exactly.

## 2. Number representation (`core_data.py`)

- Fixed `base` (default 10). Numbers are digit sequences.
- **LSB-first** for +,−,× (carry/borrow propagate low→high). **MSB-first** for ÷ (long division
  goes high→low). Direction is intrinsic to each op.
- Ground truth is exact Python integer arithmetic. `length_gen_report` is the standard harness.

## 3. The core primitive: a tiny neural Mealy machine

`NeuralMealy` (`expA_mealy.py`) / `UnifiedMealy` (`expF_unified.py`):
```
out_t   = g(state_{t-1}, input_t)        # output digit logits
state_t = f(state_{t-1}, input_t)        # next recurrent state (tiny: 1–8 dims)
```
- `g`, `f` are 1-hidden-layer tanh MLPs. Input = `[state ; onehot(a_t) ; onehot(b_t) ; onehot(op)]`.
- The state is a **deliberately tiny** continuous vector. It self-quantizes into a finite-state
  machine, which we extract by discretizing the state and verifying exactly.
- `UnifiedMealy` adds the op-code input so **one network** does several ops (selected, not composed).

### What this primitive discovers (all verified exact, length-gen 1.000 to w20+)
| op | discovered procedure | extracted form |
|----|----------------------|----------------|
| + | **carry** (2-state FSM) | exact 2-state FSM |
| − | **borrow** (2-state FSM; NOT add+negate) | exact 2-state FSM |
| × by 1 digit | multiplicative carry (~9 states) | net generalizes; FSM ~0.94 (smooth manifold) |
| ÷ by `d` with `d | base` | base-modular remainder (carry-like) | clean |

## 4. The walls (proven limits of a fixed-state machine)

These are **architectural**, not training problems. No schedule or capacity fixes them:
- **Full n×n multiplication is not finite-state** — the column sum grows without bound (a theorem
  about the unbounded-width family). Empirically, training did not even fit width 3 at 64 state
  dims (`expMul_full.py`) — note (2026-06-09 audit): width-3 mult is a finite function and *is*
  representable; the width-3 failure is an optimization observation at that budget, the
  representational wall proper is the unbounded-n statement.
- **Division by a base-coprime divisor is not learnable end-to-end.** The per-step state update
  `rem' = (rem·base + a_t) mod d` is a **mod by a non-base divisor**; +,×,÷(d|base) all reduce
  their state via the base (a free shift), which the net can hold, but a non-base modulus drifts.
  Confirmed three ways: fixed-divisor /7 fails; the pure `mod m` task fails for m∈{7,9} but is
  perfect for m=10; and **the same divisor flips with the base** (/3 fails base-10, passes base-12;
  /5 the reverse). ⇒ **Discoverability is representation-dependent.** (`expD_divfixed.py`,
  `expD_modm.py`)

## 5. Composition layer — CURRENTLY EXTERNAL (Python glue)  ← the limitation

The walls are crossed by **composing the discovered primitives** — but the composition is hand-
written Python, not in the model:
- **Full multiplication** = `acc=0; for j: acc += SHL(MULDIGIT(A, B_j), j)` (`expC_compose.py`).
  Grounded on the extracted **carry** FSM (muldigit = repeated add). Exact to w12+.
- **Division by any divisor** = long division whose inner step is **repeated subtraction** using
  the model's own subtraction (`make_hybrid_qdiv` in `expF_unified.py`). Exact to w20 for all
  divisors 2–9, including the neural-wall ones (/3,/7,/9).
- The GP in `expC`/`expD_div_compose` *discovers* these loop bodies, but a Python interpreter runs
  the loop.

**This is the gap:** the model is a collection of primitive solvers; the loops, accumulation, and
routing that turn primitives into algorithms live in Python. The model does not develop
compositional solutions on its own. → addressed in `expG_controller.py`.

## 6. The unified 4-op model & training regimes (`expF_unified.py`)

One `UnifiedMealy` (d=8, hidden=96, ~8k params) over +,−,×(1-digit),÷(1-digit), each fed in its
natural digit order. Division neural head restricted to learnable divisors {1,2,5}; all other
divisors handled by the hybrid composition.

| regime | what | result (mixed-test exact) |
|--------|------|---------------------------|
| **Co-training** | random op every step | 0.854; +,− perfect, × strong |
| **Sequential** | one op per phase | 0.100 — **catastrophic forgetting** (only last op survives) |
| **Blended** | curriculum + rehearsal (focus 0.60) | 0.753 — forgetting cured; later ops under-trained |
| **Dynamic** | rehearsal + balancing phase weighted ∝ deficit | best: +,−,× ≈ 1.0, ÷{1,2,5}=1.0 |

Findings: rehearsal cures catastrophic forgetting; a deficit-weighted dynamic schedule also fixes
the exposure imbalance (auto-pours training into the worst op). A partially-unlearnable task
(coprime ÷) **interferes** with the learnable part through the shared net — restricting the neural
÷ head to {1,2,5} removes the interference.

**Complete system** (neural +,−,× and ÷{1,2,5}; hybrid ÷ for the rest): **1.000** on a fully-mixed
bag (all divisors), **1.000** on the 21-case edge suite, and **0.995–1.000** on chained
expressions that combine all four ops in one problem (to 10-digit operands).

## 7. Evaluation suite (`expF_unified.py`)

- **Per-op length-gen** — exact accuracy at widths 1…20.
- **Mixed bag** — each problem one random op across all four.
- **Edge cases** — 21 curated: carry/borrow chains (999+1, 1000−1), identities (×0, ×1, +0, −0),
  a−a, a<d, 0÷d, division walls, length edges.
- **Chained expressions** — one problem applies a random permutation of {+,−,×,÷} left-to-right,
  feeding the model's own output forward (tests composition end to end).

## 8. File map

| file | contents |
|------|----------|
| `core_data.py` | codecs, exact ops, length-gen harness |
| `expA_mealy.py` | NeuralMealy + FSM extraction (carry/borrow) |
| `expA_mul1.py`, `expA_div1.py` | single-digit ×, ÷ |
| `expB_progsearch.py` | gradient-free program search (register VM) |
| `expC_compose.py`, `expC_fsm_primitives.py` | multiplication = composition (Python loop) |
| `expD_discrete.py` | discrete-state (STE) extraction; fit/extract tradeoff |
| `expD_divfixed.py`, `expD_modm.py` | the division diagnostic + representation-dependence |
| `expD_div_compose.py` | division = repeated subtraction (Python loop) |
| `expE_shared.py` | shared +/- mechanism study |
| `expF_unified.py` | unified 4-op model, regimes, full eval, hybrid ÷ |
| `expG_controller.py` | **next:** internal composition (learned controller) |
| `runs/*.pt` | saved checkpoints |
| `TRACKER.md` | full experiment log (read this for the narrative) |

## 9. Reproduce

```
wsl bash -lc 'bash /home/joebachir20/math_lab/run.sh <script.py> [args]'
```
Env: conda `mathlab` (py3.11, torch 2.6 cu124), RTX 4060. Models are tiny (<10k params).
