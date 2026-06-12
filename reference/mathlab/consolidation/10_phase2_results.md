# 10 — Phase-2 Results (Fable 5 autonomous block, 2026-06-10)

The audit (09) corrected the *record*; this phase tested the audit's open questions on
GPU. Three **closures** (re-running the experiments the audit flagged, with the bugs
fixed) and three **new experiments** (the audit's revised frontier directions). Read
09 first; this assumes it.

Every number here is in a named artifact under `runs_pod/phase2/` or `runs/`, and every
reported winner was re-executed from its stored genome (the RULES amendment). The TRACKER
entries dated 2026-06-10 are the primary record; this is the synthesis.

---

## A. The three closures — what the fixes changed

### A1. exp2 QD (landscape wall #4) — the session-10 verdict was overturned
The audit found the MAP-Elites archive used a duplicate-index CUDA scatter (fitness and
genome could come from different machines). Fixed (per-cell argmax) + every archived
winner re-executed. Five runs, n=5 TMs, deepest halter:

| run | sampling | evolution | MAP-Elites |
|---|---|---|---|
| s1 (span_ones, Tmax 8000) | 51 | 675 | 410 |
| s2 | 45 | 596 | 201 |
| s3 | 50 | 1887 | 713 |
| **rt_span descriptor** | 51 | 675 | **2139** |
| Tmax 30000 | 81 | 675 | 258 |

- **"QD is the wrong tool" (session 10) is withdrawn.** With a valid archive MAP-Elites
  is the same order as evolution, and under a depth-aligned descriptor (rt_span) it
  **beats** evolution (2139 vs 675). QD's depth is descriptor-dependent — the known QD
  fact, now measured.
- **Evolution depth is heavy-tailed**: 596 / 675 / 1887 across seeds; the session-10
  "6238" was an upper-tail draw. "Scale moves the wall 140×" → **"~1–2 orders,
  heavy-tailed, report the median of ≥3 seeds."**
- Both audit flags on this experiment (the bug and the n=1) were real and each distorted
  the headline. Cleanest possible vindication of the new RULES.

### A2. avida_oe (target-free computation) — the "a+b discovered" claim is disconfirmed
Re-ran with an 85-function named suite (was 10), gen-10 snapshots (was 200), top-5
described, a persisted first-match table. 3 seeds, 1000 gens.
- The session-10 "discovered ADDITION (a+b exact) by gen 50, XOR by gen 100" **never
  reproduces.** The only early matches are trivial projections {a, b} and XNOR, all
  abandoned by ~gen 100.
- The converged winners (edge≈1.0) **match no name** — e.g. seed 7's top-3 all compute
  the same map, nearest-named 0.545 (≈chance). Target-free search converges to
  **structured *unnamed*** functions.
- This **retires the old wall-#6 framing** ("everything found is a named op/composite")
  at the artifact level: the search does not resurface named ops; it resurfaces unnamed
  structured composites. The ceiling moves to **recognition** (Frontier 2).

### A3. expFF_search (learnability wall #5) — done before the GPU block, measured not asserted
The audit argued #5 was demonstrated only for *statistical* learners. Exact-filtered
program search (the project's own engine) **recovers the "un-discoverable" mixer exactly
from outcome** (depth-2 search, held-out 1.000, zero false positives on a random-
permutation control), and dies at a depth-4 secret (≈2^39.8 key space vs 2^24.3 budget).
**Wall #5 is a description-length wall, not an op-count wall.**

---

## B. The three new experiments — the audit's revised frontiers, tested

### B1. depthstruct — depth and structure are in TENSION on the program space
Closed the expEE signal-craft gap (edge-of-chaos on the sparse TM space-time was
contaminated; replaced with a head-track move-bit measure). 3 conditions × 2 seeds,
render-verified.

| cond | s1 rt/track | s2 rt/track |
|---|---|---|
| depth | 307 / 0.000 | 596 / 0.053 |
| depthXtrack | 307 / 0.000 | 596 / 0.053 |
| trackonly | 30000 / 1.000 | 30000 / 1.000 |

- depthXtrack converges to the **identical** machine as depth (both seeds): the structure
  term is **inert among halters** because deep halters have ~zero head-track structure.
- trackonly finds a **structured non-halting** walker (rendered: a sawtooth/triangular
  drift) — genuinely edge-of-chaos, but never halts.
- **Deep + structured + halting** (the Bennett-logical-depth object) is **landscape-
  inaccessible**: depth-evolution finds regular deep halters, structure-evolution finds
  structured non-halters, and nothing the search reaches is both. Sharpens wall #4 with
  an uncontaminated signal.

### B2. avida_loop — naming-density is DEPTH-INVARIANT
Added bounded loops to the cross-bit VM so a fixed genome reaches composition depth up to
24·maxit. Pre-registered rule: similarity falls with depth ⇒ depth gates naming; flat ⇒
recognition is the ceiling. 2 seeds × maxit {1,4,16}.

| reachable depth | edge | top_named_sim | named_in_top5 |
|---|---|---|---|
| 24 | 0.998 | 0.579 | 0 |
| 96 | 1.000 | 0.595 | 0 |
| 384 | 1.000 | 0.603 | 0 |

- **Flat** (0.579 → 0.603) across a 16× depth increase, nothing named at any depth. By the
  pre-registered rule: **recognition is the ceiling, not depth or vocabulary.**

### B3. PCF identity hunt — the instrument REACHES THE RAMANUJAN-MACHINE TAIL
GPU float64 prefilter → mpmath/PSLQ verify, positive controls + reject-rational filter.
- **deg-2 grid** (4.82M PCFs): only classical π/e/surd families (rediscovery). The
  tail-constant null is **grid scope** — ζ(3)-class identities need deg-3 numerators with
  b=−n⁶, structurally absent from deg-2.
- **deg-3 × b=−n⁶ family** (|c|≤55, 151.8M PCFs): beyond the Apéry control, the hunt
  **independently rediscovered a Ramanujan-Machine conjecture from outcome**:
  aₙ=(2n+1)(3n²+3n+1), bₙ=−n⁶ → **8/(7·ζ(3))**, verified to 250 digits, coefficients
  matching the 2021 RM paper (a recent, non-classical, **still-unproven** result).
- **ζ(2) family** (b=+n⁴, |c|≤55): only the 30/π² control (rediscovery).
- The instrument provably operates in the exact regime where the only historical
  human-unknown finds live.
- **Past the published region** (|c|≤120, **3.37B PCFs**, 20× past the RM coefficients):
  an **honest null** — only the known forms reappear (Apéry 6/ζ(3), the RM 8/(7ζ(3))
  conjecture + mirror; the ζ(2) family gives only 30/π²). No novel identity.
- **Phase 3 — the general sweep** (`--family gen6`, **10.2B PCFs**, A deg-3 × B deg-6,
  every family shape at once + δ/irrationality-scoring + a quadratic-relation re-mine):
  the instrument **reached two more tail constants from outcome** — verified (119-digit)
  CFs for **Catalan (1/2G**, a_n=3n²+3n+1, b_n=−2n⁴) and **8/π²** — bringing the tally to
  **three rediscovered tail constants (ζ(3), Catalan, π²)**. Both reference-subtracted to
  **known** families (the Catalan form is exactly the κ=0 member of
  [arXiv:2210.15669](https://arxiv.org/abs/2210.15669), fetched and matched; π² is
  classical). **No novel identity** — but the *full Frontier-2 recognition pipeline ran
  end-to-end and correctly* (detect verified structure → fetch literature → match
  published family → classify KNOWN → no claim). The δ-hunt honestly surfaced only the
  trivial periodic surds (best approximators by construction); the quad re-mine found
  only classical *e*-family quadratic relations. A novel find needs **larger coefficient
  height** (RM's deeper results exceed |c|=9), a constant outside the battery (MZVs,
  L-values), or a non-CF object. The detect/verify/reference-subtract tooling is built
  and reusable. *(Routing per Joe: GPU stage-1 on pod, all PSLQ verification on local
  cores; streaming OOM fix validated at billion-scale.)*

---

## C. The synthesis — what the autonomous block establishes

1. **The audit's corrections held up under re-run.** Every flagged claim that was re-run
   either reversed (QD), was disconfirmed (avida a+b), or was explained and bounded
   (exp3-mul earlier; the deg-2 PCF null = grid scope).
2. **Recognition (Frontier 2) is the binding moonshot wall — triangulated three ways.**
   Across CA-function space (avida_oe), program space (avida_loop), and identity space
   (PCF): surfacing structure is easy and target-free; *certifying novelty* is the wall.
   The primitive-vocabulary framing (old #6) is retired. Depth is not the gate (B2);
   vocabulary is not the gate (A2); the wall is a structure-agnostic recognizer +
   reference subtraction, with the irreducible epistemic caveat (evidenceable, not
   provable).
3. **The identity direction is the one with a working path to genuine novelty.** It has a
   historical base rate of human-unknown finds, and the scaled instrument now provably
   reaches that tail (the RM-conjecture rediscovery). This is the single most promising
   concrete moonshot lead the project has — gated by scale + reference subtraction, not
   by a wall.
4. **The new RULES paid for themselves immediately.** Re-execute-archived-winners caught
   the QD corruption live; comparative-claims-need-seeds caught the evolution heavy tail;
   render-before-verdict confirmed the depthstruct tension; stream-don't-accumulate (an
   OOM bug caught and fixed mid-run) is a new entry for the trap list.

**The honest moonshot status is unchanged but its geometry is sharper:** the wall is
recognition, the lead is identity space, and the instrument for the lead now works.
