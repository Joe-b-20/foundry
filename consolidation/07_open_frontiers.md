# The Two Open Frontiers

The moonshot — a small system discovers a math/computational procedure humans don't
know, that we can read — is **open**, and the framework now locates its remaining
ceiling at exactly **two places**. These are the targets for the next phase.

The two are not independent: a genuine moonshot needs **both** a primitive vocabulary
whose closure escapes the named operations (Frontier 1) **and** a way to recognize
that what it found is genuinely unfamiliar (Frontier 2). Solving one without the other
is insufficient — vocabulary without recognition gives you un-named functions you
can't certify as novel; recognition without vocabulary gives you a detector that only
ever sees composites of known ops.

---

## Frontier 1 — Primitive design for non-named operations

> ### ⚠ Revised 2026-06-09 ([09_fable5_audit.md](09_fable5_audit.md) §2.3) — read this box first
>
> The original statement of this frontier rested on an unsound argument ("every
> discovered function is a composite of the op set, so no human-unknown procedure
> can appear" — but *every* computable function, including every future human
> discovery, is a composite of any universal op set). The corrected reading of the
> session-10 evidence: target-free search reaches only **short/shallow
> compositions**, and the shallow region of standard arithmetic-logic vocabularies
> is **densely named**. The ceiling is **reachable depth × naming-density**, not
> closure — and "design primitives whose composites are un-named" is trivially
> satisfiable (any random S-box qualifies) and buys nothing by itself:
> un-named-by-obscurity ≠ interesting. The non-trivial residue of Frontier 1 is:
> find vocabulary/search pairs whose **reachable, bridge-flagged region contains
> objects that are un-named AND interesting** — which makes it largely downstream
> of Frontier 2 (recognition), plus the depth problem (wall #4). Three better-posed
> directions the audit promotes:
> **(a) encoding freedom for regular ops** — the one theorem-licensed,
> already-demonstrated escape (carry-save, expI; signed-digit/residue systems
> untried — see the corrected wall #3);
> **(b) deep-but-structured search** — reach compositions beyond the shallow named
> layer while retaining structure (the depth-signal family's open problem);
> **(c) the identity-space hunt at scale** (expV) — unchanged, and still the one
> regime with a historical base rate of human-unknown finds.
> Also note: the launch-point experiment's "discovered a+b/a^b exactly" waypoints
> are unsupported by surviving artifacts (audit §1.3); its persisted winners are
> structured functions the 10-entry recognizer could **not** name — itself the
> cleaner motivation for Frontier 2.

**The goal (as originally stated):** find primitive families whose **composition
space spans outside the human-named arithmetic-logic regions**, so that target-free
search can surface structure that is not just a composite of operations we already
have words for.

### Where the ceiling was located (session-10 final result)

`gpu_avida_oe.py` ran the validated **target-free edge-of-chaos bridge** (no named
function ever specified) on a **rich cross-bit stack VM** (ops:
`nand, and, or, xor, add, sub, shl, shr` + stack manipulation — cross-bit ops
propagate carry, so the function space is vast and largely un-named, not just the 16
bitwise booleans). It **worked generatively**: from random programs it discovered
**addition and XOR exactly, then climbed to maximally-structured functions** that
match no simple named op.

**The original "pinpointed ceiling" claim is retracted as stated** (see the box
above and audit §1.3/§2.3): "every discovered function is a composite of the
primitive op set" is true of all computation and bounds nothing; and the
addition/XOR waypoints lack surviving evidence (`match=[]` at every persisted
snapshot). What the run actually established: the generative machinery works
target-free on this substrate, the merit climbs to maximally-structured functions,
and the persisted winners are functions a (10-entry) named suite could **not**
identify. The honest ceiling is reachable **depth** × **naming-density** ×
**recognition** — wall #6 restated.

### What primitives *have* been tried (all close into named functions)

| Primitive family | Experiments | Closure (what it composes into) |
|------------------|-------------|--------------------------------|
| Digit-serial register ops (carry/borrow/successor) | expA–expM | the digit-serial arithmetic operations (+, −, ×, ÷, and the carry/borrow primitives themselves) |
| Whole-number register ops | expK (MOD/SUB/SWAP), expO (AVG/NEXT/TAKE), expP (NEWTON), expS (FACTOR/INC) | Euclid, binary-search, Newton, trial division — named efficient algorithms |
| Comparison/structure ops on lists | expQ (SWAP/ADV/RESET), expR (SETM/INCJ/PLACE) | bubble sort, selection sort |
| Bilinear tensor coefficients {−1,0,1} | expL/N/T | Strassen, Laderman, Karatsuba (named optimal decompositions) |
| Word-level bit ops | expU | Hacker's-Delight branchless tricks (named) |
| BFF self-modifying byte-tape | gpu_weird_soup/alife/metabolism | self-replicators (a known ALife phenomenon); computation plateaus |
| NAND-complete stack | gpu_avida | the 16 boolean functions (XOR/EQU…) |
| Rich cross-bit stack (+,−,^,&,\|,shift) | gpu_avida_oe | composites of named arithmetic/logic ops |

The pattern: **every substrate's closure is the named functions.** Even the "richest"
one (cross-bit stack) composes `+, −, ^, &, |, shift` into arithmetic/logic composites
that all have names or are obvious compositions of named ops.

### What has *not* been tried (candidate directions)

These are genuinely open and on-theme (weird, exploits exact verification, fits the
small/narrow constraints). None has been attempted:

- **Primitives over an exotic algebraic structure** whose composition closure is *not*
  the standard integer/boolean operations — e.g. operations on a non-commutative
  structure, a tropical/min-plus semiring, a finite field with non-standard
  embedding, or a custom group where "natural" compositions have no schoolbook name.
- **Number representations beyond the ones probed.** Factoradic (expBB) showed
  position-dependence is handleable but radix-extrapolation is the wall. Untried:
  redundant/signed-digit systems *as the substrate for new ops* (not just carry-save
  for addition), continued-fraction or Stern-Brocot representations, residue number
  systems (CRT) where "natural" primitive ops compose into procedures with no
  positional analog.
- **Primitives that are themselves un-named functions.** Instead of `nand` (whose
  closure is the boolean functions), seed the VM with a small set of *structured but
  un-named* base functions (e.g. a fixed pseudo-random-but-low-complexity mixing
  primitive that is *not* one-way — to avoid wall #5) and ask what their composition
  space contains.
- **Higher-type primitives** — primitives that take/return functions (combinators), so
  the composition space is over *programs* rather than over *values*, where the named
  region may be a smaller fraction.
- **The expV identity-space direction at scale.** `↻ ADVANCED (2026-06-10,
  gpu_pcf_hunt):` this is no longer just a lead — it is **demonstrated to reach the
  tail**. The GPU hunt (millions–billions of polynomial continued fractions, float64
  prefilter → mpmath/PSLQ verify, with positive controls and a reject-rational filter)
  did two things: (a) at deg-2 it reproduced only the **classical π/e/surd families**
  (rediscovery, as predicted — the tail null there is *grid scope*, not absence); and
  (b) at the **deg-3 × b=−n⁶ family** it **independently rediscovered a Ramanujan-
  Machine conjecture from outcome alone**: aₙ=(2n+1)(3n²+3n+1), bₙ=−n⁶ → **8/(7·ζ(3))**,
  verified to 250 digits, coefficients matching the 2021 RM paper (a *recent,
  non-classical, still-unproven* result). So the instrument provably operates in the
  exact regime where the only historical human-unknown finds live. The remaining work
  is **scale past the published coefficient region** + the recognition/reference-
  subtraction discipline (Frontier 2). Identity space (continued fractions, integer
  relations) is the one regime where modest-hardware search has historically produced
  genuinely human-unknown mathematics (BBP, the Ramanujan Machine); the new identities
  live in the tail (higher-degree PCFs, larger coefficients, under-explored constants:
  Catalan, ζ(3), γ, MZVs). This is the most concrete "where novelty has actually been
  found before" lead — now with a working scaled instrument — but it needs scale and it
  hits Frontier 2 (you can verify an identity to N digits but not prove it's unknown).

### The honest open question (revised 2026-06-09)

Not "design a primitive set whose composites are un-named" (trivially satisfiable,
worthless alone) but: build a **search + vocabulary pair** whose reachable region
(at honest budgets) contains objects that are simultaneously (a) flagged by the
bridge signals, (b) **deeper / structurally richer than the shallow named layer**,
and (c) checkable against a serious reference catalog — then point the generative
machinery (expDD/avida_oe, which works) at it. Priority candidates: encoding-freedom
substrates for regular ops (theorem-licensed, carry-save precedent); depth-rewarding
search on composable substrates (the wall-#4 problem); the expV identity tail. Each
lands immediately in Frontier 2's recognition problem, which is the binding
constraint.

---

## Frontier 2 — Novelty recognition

**The goal:** operationally distinguish **"genuinely unfamiliar structure"** from
**"known but unrecognized."** This is the second half of the moonshot ceiling, and
part of it is provably impossible.

> ### Frontier 2 is now the BINDING constraint — triangulated three ways (2026-06-10)
> Three independent phase-2 experiments converged on the same conclusion: **surfacing
> structure is easy and target-free; certifying novelty is the wall.**
> - **oe_fix** (target-free edge-of-chaos on the cross-bit VM, 3 seeds, gen-10
>   logging): the search converges to **structured functions that match no name** in
>   an 85-entry suite — and the session-10 "discovered a+b/a^b" claim **never
>   reproduces**. The attractor is unnamed-structured, not named ops.
> - **loop sweep** (reachable depth 24→96→384, pre-registered rule): nearest-named
>   similarity is **depth-invariant** (flat ~0.58–0.60, `named_in_top5=0` at every
>   depth). Making the substrate reach 16× deeper compositions did **not** move the
>   discovered functions relative to the named region — so depth/vocabulary is **not**
>   the gate.
> - **PCF hunt**: structure (verified identities) is found readily; the question
>   "is this verified identity *new*" is answerable only as reference-subtraction
>   (the 8/(7ζ(3)) hit was structured, verified, and — by literature check — **known**).
> Across CA-function space, program space, and identity space the bottleneck is the
> same: a **recognizer** that flags "structured AND not in the references," with the
> permanent caveat that the final step is evidenceable, not provable. The
> primitive-vocabulary framing (old wall #6) is retired; **recognition is the wall.**

### The standing position (wall #11)

**Novelty is evidenceable but not provable.** A find can only ever be certified as
"structured **and** not in the references I know," never "provably absent from the
human catalog." This is an irreducible epistemic wall. So Frontier 2 is really two
sub-problems:
1. **The tractable half:** recognize *structure* without having pointed the detector
   at a *named* class (so it can flag the genuinely-unfamiliar, not just resurface
   known classes).
2. **The irreducible half:** certify "not in the human catalog" — which can only ever
   be *evidence* (true to N digits / not in a reference set), never proof. Manufacture
   no novelty claims.

### Where intrinsic-signal approaches surfaced vs missed

The core problem with every validated bridge signal: **it was built to flag a *known*
notion of structure**, so by construction it resurfaces known classes (edge-of-chaos →
class-4 CAs; invariants → integrable maps; depth → Busy-Beaver; compression → automatic
sequences). A signal pointed at a named class cannot recognize an unnamed one.

The two attempts that pushed toward structure-agnosticism:
- **Learning progress (`gpu_weird_lprog`)** is the closest the project came: it names
  *no* structure type (interesting = learnable-with-effort) and **cleanly rejects
  trivial + noise as a distribution** (the genuinely hard half). **But** it is
  **fragile at the class-4-vs-chaos boundary** (some chaos has exploitable early
  determinism and out-ranks class-4 at larger budgets), and it separates
  interesting-from-trivial only *distributionally*, never rule-by-rule. So it can
  *rank* but not *recognize*.
- **The survival bridge** (gpu_weird_soup/avida) produces *meaning* with no named
  target (self-replicators, evolved XOR/EQU) — a different and promising route,
  because the "recognition" is implicit in survival rather than in a scored metric.
  But what survives is vocabulary-bounded (Frontier 1) and the *novelty* of what
  survives is still un-certifiable.

### What this implies for an approach

- A novelty-recognizer must be **structure-agnostic** (not pointed at a named class) —
  learning-progress is the prototype; the open work is making it rule-by-rule reliable
  and pairing it with a richer object space.
- It needs a principled **catalog/reference** to subtract the known — and an honest
  accounting that this can only ever produce *evidence* of novelty. (expV's discipline
  is the model: it found the rational-PCF false-positive trap by checking that a
  "discovery" matching *every* constant at once is the value-independent triviality.)
- **Beware the labeling artifacts** that masquerade as novelty *or* as familiarity:
  - expX's "0/26 recognized" was a 5-entry reference list artifact (the LC/zr numbers
    proved genuine structure; the label set was just incomplete).
  - The recurring lesson: **"looks complex by a metric" ≠ "is dynamically complex"** —
    you must render and inspect (gpu_exp1b needed 4 rounds of this).
  - **Sequence diversity is a false innovation metric** (alife: turnover was neutral
    drift). Use functional-complexity metrics.

### The honest open question

Build a detector/procedure that recognizes structure **without** having been pointed at
a named class, reliably enough to flag a candidate **rule-by-rule** (not just shift a
distribution), paired with an honest reference-subtraction that yields *evidence* of
unfamiliarity. Combined with Frontier 1 (a vocabulary that can actually produce
unfamiliar structure), that is the moonshot's remaining path — with the permanent
caveat that the final step ("human-unknown") is evidenceable, never provable.

---

## A note on scale (wall #10)

Both frontiers interact with scale. Session 10 confirmed the signals work cleanly at
GPU scale but still surface *known* classes — so scale alone is not the missing
ingredient; **vocabulary and recognition are.** Scale matters as a multiplier *once*
the vocabulary points somewhere new (a larger search of a non-named composition space)
and for the identity-space direction (where the novel objects genuinely live in a tail
reachable only beyond a single session). Budget scale for *after* the frontiers move,
not as a substitute for moving them.
