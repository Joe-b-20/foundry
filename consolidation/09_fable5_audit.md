# 09 — Fable 5 Critical Audit (2026-06-09)

First act of the successor instance: a critical audit of the project before any new
experiment. Method: read all of `/consolidation`, `PROMPT.md`, `.claude/RULES.md`,
`ARCHITECTURE.md`, the full moonshot arc of `TRACKER.md` (sessions 8–10 line by line,
earlier sessions selectively), then re-verify the load-bearing claims against the
**code** (what the scripts actually compute) and the **artifacts** (`runs/`,
`runs_pod/`, `runs_pod/runs/`), including re-running analyses on the stored logs and
re-rendering stored space-times. Everything below cites the file it was checked
against. A verification ledger (what checked out, what didn't) is at the end.

The headline first, because the discipline demands it both ways:

> **Most numbers are faithful and most caveats were self-flagged.** The honesty
> culture is real: of ~30 claims re-verified, ~22 reproduce exactly from artifacts.
> But the audit found **two evidence-contradicted headline claims** that became
> consolidation framings (exp3-mul "fits-but-memorizes"; the QD comparison, which
> sits on a provable archive-corruption bug), **one unsupported discovery claim**
> (avida_oe's "addition by gen 50"), and **three logical/scope holes** in the wall
> taxonomy (the Myhill–Nerode scope, the learnability-wall scope, and the
> primitive-vocabulary-gate argument — the last of which materially redirects Open
> Frontier 1). None of these kills a contribution; all of them change what the
> docs should say.

---

## Part 1 — Evidence-contradicted claims (highest severity)

### 1.1 exp3-mul: "the memory net FITS mul in-distribution" is false on its own log

**The claim** (TRACKER session-10 GPU entry; propagated to `03_wall_taxonomy.md` #1
`↻ REFINED`, `01_project_arc.md` §10, `04_experiment_catalog.md` III.A,
`08_what_not_to_redo.md` §A, and the auto-memory): *"at 40k steps the MEMORY net
FITS mul in-distribution (loss → 0.0003, where the baseline stays ~1.4 and never
fits) yet length-generalizes 0.000 — it MEMORIZED the training widths."* This became
the refined wall-#1 statement: *"richer substrate enables **fitting** the hardest
non-regular op, not discovering."*

**What the artifact says** (`runs_pod/runs/exp3_mul.log`; train_w=[1..6],
test_w=[4,6,...,20], so **w4/w6 are in-distribution**):

- The training loss never converges — it **oscillates** between ~0.0001 and ~1.9 to
  the last step (step 39000: 1.63; step 40000: 0.0003). The "loss → 0.0003" was the
  final printed line of an oscillating series.
- The oscillation is **width-stratified**. The width sampled at each step is
  reproducible (`random.Random(777)` in `gpu_exp3_memory.py:227`); joining widths to
  the logged losses for memory seed 0 gives: width 1 median CE **0.0006** (fit),
  width 2 median **0.20** (partial), widths 3–6 medians **0.97–1.65** (not fit;
  chance ≈ 2.30). The net fit width 1, partially width 2, and **never widths 3–6**.
- The **baseline has the same profile** (min logged loss 0.0001; 47/81 logged losses
  > 1.0 for *both* architectures) — the claimed memory-vs-baseline fitting gap does
  not exist.
- Exact accuracy at the **in-distribution** widths 4 and 6 on fresh samples is
  **0.000 for every completed seed, both architectures** — so "fits in-distribution"
  is false even setting the loss reading aside, and "memorized the training widths"
  is unsupported (it didn't master them in any sense, on training-distribution data).

**Corrected statement:** external memory crosses the wall for reversal (2/3 seeds,
verified in `exp3_rev.log` — that result stands). For multiplication, *neither*
architecture learns beyond width ~2; there is **no evidence memory helps mul at
all**, and no "fits-but-doesn't-discover" phenomenon in this data. The wall-#1
refinement should revert to: *memory demonstrated a crossing for a stack-shaped op;
for mul it produced nothing — fit failure, not a fit/length-gen dissociation.* The
"fits, fails len-gen" dissociation was the *interesting* version of the result; the
data contains the boring version.

**How it happened:** a log-tail skim during a live GPU session (the final loss line
+ a glance at a mid-run baseline) became a tracker sentence, survived consolidation
(which checked the *len-gen tables* against the log — those were right — but not the
*loss* claim), and got upgraded into a taxonomy refinement. Lesson for the rules:
a per-batch loss series where batches differ by width is not a convergence curve.

### 1.2 gpu_exp2: the MAP-Elites archive has a real corruption bug; "QD is the wrong tool" is confounded

**The claim** (`03_wall_taxonomy.md` #4, `08_what_not_to_redo.md` §A, TRACKER):
sampling 49 / MAP-Elites 154 / evolution 6238 at matched budget ⟹ *"QD does NOT
cross the landscape wall — ~40× worse than plain evolution for raw depth... QD is
the wrong tool."*

**The bug** (`gpu_exp2_qd.py:150–159`, `insert()`): the per-generation archive
insert scatters candidates by cell id with duplicate indices —
`cand_fit[cells[order]] = fit[order]` and `cand_gen[cells[order]] = flat[order]`.
On CUDA, index-assignment with duplicate indices is **non-deterministic** (no
last-write-wins guarantee), and the two scatters can resolve to **different
machines** for the same cell — i.e. a cell's stored fitness and stored genome can
come from different individuals.

**On-disk proof the corruption fired** (`runs_pod/runs/exp2_fast.log`, final
lines): the archive's best fitness is 154, but re-running the genome stored in that
cell gives `deepest machine: runtime=8000 ones=17 span=20` — runtime equal to Tmax,
i.e. the stored "best" genome **does not halt**. The evaluation is deterministic,
so stored-genome ≠ evaluated-genome. Every ME generation then mutates from a
possibly-mislabeled archive, which actively sabotages the QD condition (it breeds
from junk credited with good fitness).

**What survives:** evolution 6238 and sampling 49 are unaffected (no scatter in
those paths) — **scale moving the landscape wall stands**. What does *not* survive
is any quantitative or qualitative conclusion about MAP-Elites: "154" measures a
corrupted archive, single seed, with two further confounds never varied (the
(span, ones) descriptor choice, and archive capacity 676 cells vs evolution's
4096-member population). `08`'s instruction *"use QD only where coverage itself is
the goal"* is not supported by this run. If QD ever matters again, the run must be
redone with a dedup-safe insert (e.g. sort-and-segment-reduce per cell, or
`index_reduce_`/`scatter_reduce` with amax semantics plus a second gather for the
genome).

**Additional caveat on the 6238 number** (affects "moved 140×", not the verdict):
search Tmax was 8000, and the best evolved halter (6238) is at **78% of the
detection cap**; halters deeper than 8000 are invisible to the search by
construction. The planned high-Tmax verify pass of the deepest handful
(`runpod_plan.md`) was never run. So "scale moves the wall 140×" is a **lower
bound** with the stall point unmeasured, not a measurement of where evolution
stalls. (For calibration: 227× the eval budget of expEE bought 145× the depth —
roughly linear returns, which is itself worth knowing.)

### 1.3 gpu_avida_oe: the "discovered addition/XOR exactly" claim has no surviving evidence — and the surviving evidence points the other way

**The claim** (TRACKER session-10 target-free entry; `07_open_frontiers.md`
Frontier 1 launch point; `04` III.B): *"from random programs, edge-of-chaos
evolution discovered ADDITION (a+b EXACT: 3,5→8, 1000,999→1999) by gen 50, then XOR
by gen 100, then climbed to maximally-structured functions that match NO simple
named op."*

**What the artifacts say:** `08` already flagged the persisted `oe_log.json` as thin
(3 coarse snapshots, `match` unpopulated). The audit checked the **nohup stdout
logs** (`runs_pod/oe_s1.log`, `oe_s7.log`), which are the live stdout the claims
were attributed to. They log every 250 generations and show `match=[]` at **every**
logged generation (0, 250, 500) for **both** seeds; there are no gen-50/gen-100
lines anywhere, and the gen-250 probe pairs (3,5→12; 255,200→510; 1000,999→16) are
not addition. If a+b appeared, it appeared and vanished inside the first
250-generation logging gap of some run whose output didn't survive — possibly an
interactive pre-nohup run. **The waypoint claim is unverifiable and should be
withdrawn or re-run** (a re-run with per-10-gen match logging costs minutes).

Two further problems with the same entry:

- The "named" detector is a **10-entry suite** (`gpu_avida_oe.py:126–128`: a+b,
  a−b, a^b, a&b, a|b, nand, a, b, a<<1, (a+b)^a). The claim "climbed to functions
  that match no simple named op" means "matched none of these ten." The project
  already learned this exact lesson at expX ("0/26 recognized was a 5-entry
  reference-list artifact"); it recurred here in the load-bearing position.
- Note what `match=[]` at every surviving snapshot actually means: the persisted
  best organisms are functions the named-suite **does not recognize**. The on-disk
  evidence, taken at face value, is "target-free evolution produced structured,
  input-dependent, *unidentified* functions" — which is the **opposite-flavored**
  result from "everything found is named," and was arguably the more interesting
  honest finding. It was framed as ceiling-confirmation instead. (The ceiling
  *claim* has an independent logical problem — see 2.3.)

Minor: `06_code_map.md` says "quote these exactly" and then quotes the oe merit
without the `× (0.5 + 0.5·nov)` factor that is in the code
(`gpu_avida_oe.py:106,114`).

### 1.4 Smaller factual corrections

| Claim | Where | Artifact | Correction |
|---|---|---|---|
| lprog "680 residual candidates" | TRACKER weird-gambles entry | `runs_pod/runs/weird_lprog/lprog_survivors.json`: `n_residual: 2776`; log line 72 says 2776 of top-8192 | 2,776 (the consolidation docs already use the right number; the tracker was never corrected — note the two "680"s in the log are substrings of scores like `0.6808`) |
| expEE "depth-evolution beats... random control (43 > 26 > **2**)" | TRACKER expEE; `04`; `05` family 4 | `expEE_evolve.py:82–84`: for the random driver, the reported runtime is the **hash-maximizing** machine's runtime, not the deepest halter the run evaluated | The "2" is a reporting artifact (the random-driver run evaluated ~3,600 machines and surely *saw* halters ≈ sampling's 26; it just doesn't track them). The honest contrast is depth-evolution 43 vs sampling 26 — a 1.65× gap, single seed — plus the monotone climb. "Provably beats sampling+random" overstates a modest n=1 result. |
| expGG "behavioral beats geometric (0.68 → 0.94)" | `03` #8, `02` §4, `04` | `runs/expGG.log`: geometric **w3** 0.680; behavioral **w1** 0.943 decaying to 0.715@w30 | The two numbers are each method's best width. At matched widths the gap is real but smaller (w3: 0.68 vs 0.87). Also the script's own printed diagnosis is *"behavioral extraction ALSO failed — evidence the mechanism resists finite-state extraction"*; the "closes the wall" verdict is an interpretive overlay on a negative extraction result (the overlay is defensible — see 2.4 — but the doc should not imply the experiment itself succeeded). |
| exp1 s2/s3 "all clean class-4" | `08` §D, final TRACKER entry | No render artifacts exist for s2/s3; audit re-rendered (`runs/audit_exp1_s{1,2,3}_grid.png`) | s2: supported (glider-rich, clean). s3: **mixed** — the top survivor is plausibly class-4-ish, but several of its top-8 are dense chaotic-looking textures (mean density 0.617). The blanket verdict was issued metric-only, violating the project's own render-and-look rule at the exact place the rule was coined. Ceiling conclusion unaffected. |
| "the model cannot even fit width 3 (tested to 64 state dims)" | `ARCHITECTURE.md` §4, `03` #1, session-1 entries | `expMul_full.py` trains width 3 for 8k steps and reports exact acc | Width-3 mult is a finite function; a 64-dim continuous-state Mealy can *represent* it. The honest split: "not finite-state over all n" is a theorem about the **family**; the width-3 failure is an **optimization** observation at one budget. The docs conflate representability with trainability in this one phrasing. |
| expV "98 → 16 genuine identities, verified to 210+ digits" | TRACKER expV | No expV artifact exists anywhere in `runs/` | Numbers are tracker-prose only. No reason to doubt them, but they are unverifiable — the only headline result with zero on-disk trace. |

---

## Part 2 — Logical and scope holes (the framings)

### 2.1 The Myhill–Nerode wall (#3) is true only at a fixed I/O encoding — and the project itself found the escape it forgot to mention

`expCC_ladder.py` is sound: the `bisim_to_carry` BFS checks the inductive invariant
over every reachable state × digit pair, which **is** a proof of all-lengths
correctness (the docstring attributes the proof to "exhaustive simulation + a
finite-reachable-state check," which alone wouldn't prove it — the bisimulation
homomorphism is the actual instrument). The theorem invocation is correct **for
deterministic synchronous transducers with a fixed input/output alphabet and
alignment** — LSB-first digit pairs in, one base-B digit out per step.

The hole: `03` states it unconditionally — *"a regular operation has a unique
minimal transducer, so 'exact correctness + length-generalization' is literally
'bisimulate that one machine'... target-driven search is FORCED to rediscover the
known algorithm. FIX: leave regular ops / leave target-driven search."* But change
the **encoding** and the theorem no longer pins you: the project's own
**carry-save adder** (expI, session 3) is a correct, length-generalizing addition
procedure that is *not* a re-encoding of the 2-state carry machine — it computes a
different transduction (redundant output code) composed with a decoder. The
session-3 docs even celebrate it ("a procedure humans rarely use by hand").

So wall #3 has a **third fix the taxonomy omits: change the I/O encoding** — and it
is the only escape route for regular ops that the project has *actually
demonstrated to produce a non-canonical algorithm*. The algorithmic variety for a
regular op lives exactly in the encoding choice (carry-save, signed-digit,
residue systems...). This thread was dropped after session 3 and never connected
to the theorem when the theorem arrived in session 9. It is arguably the most
underrated open direction in the project (it overlaps `07`'s untried "redundant /
signed-digit / residue number systems" bullet, which is listed there as a
Frontier-1 idea without noting that the theorem *licenses* it specifically).

### 2.2 The learnability wall (#5) is demonstrated for statistical learners, not for "outcome-driven discovery" — and the project's own engine would crack the demo function

`expFF` is internally solid (numbers verified; the rounds-knob logic is good). The
overreach is in the generalization. The panel (`expFF_learnability.py`) is ridge
regression, kNN, and a 128-hidden full-batch numpy MLP — three **statistical**
learners. The conclusion drawn (`03` #5): *"outcome-driven discovery can only
MEMORIZE... FIX: none from outcome (would need the program, not examples)."*

But the project's central method is not statistical learning — it is
**exact-filtered program search**. The demo function is a *2-op, ~30-bit
description* mixer (shift constant 7, multiplier 0x9E37, R rounds). Given a VM
whose primitives include xor-shift and mul-mod-2^16 (the analog of giving the
adder VM digit ops), enumerating or sampling short programs and exact-checking
against 4,000 examples finds it essentially immediately — a wrong short program
passes 4,000 16-bit exact checks with probability ~0. Occam search defeats
*short-description* functions from outcome by construction; what it cannot defeat
is **long-secret** functions (real keyed crypto), where the description itself is
the needle. The "2-op" framing, intended to make the wall look sharp, actually
points at the regime where the wall *doesn't* bind the project's own paradigm.

Corrected scope for #5: *a function can be efficiently computable yet expose no
statistically exploitable structure; gradient/kernel/local learners then collapse
to chance (demonstrated). For exact-filtered program search the wall binds only
when the function's description length exceeds what the search can enumerate —
i.e. the cryptographic wall for this project is a **description-length wall**, not
a 2-op wall.* Also, "the MLP can't even memorize" is a capacity artifact (a ~4k-
parameter MLP vs 64,000 random target bits) and shouldn't be cited as evidence of
pseudorandomness.

(A cheap, fun follow-up that would make this precise: run the project's own recipe
on f_R with mixer-adjacent primitives and watch it succeed from outcome; then grow
the constant/key size and watch discovery die at the enumeration boundary. That
would turn a mislabeled wall into a measured one.)

### 2.3 The primitive-vocabulary gate (#6) — the stated argument is circular, and Frontier 1 partly collapses into Frontier 2

The load-bearing sentence (`03` #6, `07` Frontier 1, TRACKER): *"every discovered
function is a composite of the primitive op set, so **no human-unknown procedure
can appear** — the vocabulary bounds the reachable space to composites of known
operations."*

As stated this proves too much, and therefore nothing: **every** computable
function is a composite of any universal op set. The closure of {nand} alone is
all boolean functions — including astronomically many with no name and every
future human discovery; human-unknown algorithms *are* composites of known
primitives (every algorithm ever published is a composite of known machine ops).
If the argument were valid, no substrate could ever produce novelty, including
the ones Frontier 1 proposes — note that `07` itself describes the cross-bit VM's
space as "vast and largely un-named" *and* concludes from the same experiment
that everything reachable is named.

What the experiments actually found is a statement about **search dynamics ×
depth**, not closure: target-free evolution under these merits finds **short,
shallow compositions**, and shallow compositions of standard arithmetic/logic ops
are dense in named functions. The ceiling, properly located, is: *(a)* the search
reaches only the shallow region (this is the **landscape/depth** issue again —
walls #4 and the depth-signal family), and *(b)* in the shallow region,
"structured" ≈ "named" for these vocabularies, so the bridge signals resurface
named things (this is **recognition**, Frontier 2).

Consequences for Frontier 1 as written:

- "Design primitives whose composition space spans outside the named regions" is
  **trivially satisfiable and therefore mis-specified**: pick any random S-box as a
  primitive and its shallow composites are un-named. Un-named-by-obscurity is free;
  it buys nothing. (The factoradic lesson, expBB, was the same shape: "alien" ≠
  "informative".)
- The non-trivial version of Frontier 1 is *find vocabularies where the
  shallow-reachable, bridge-flagged region contains objects that are un-named **and
  interesting*** — and "interesting without being named" is precisely Frontier 2
  (novelty recognition). The two frontiers are not parallel open problems; #1 is
  mostly downstream of #2, plus the depth problem from wall #4.
- The depth direction deserves explicit promotion in the frontier doc: the one
  regime where un-named-but-meaningful objects *provably* live (deep, structured
  computation — Busy-Beaver-adjacent, the expEE/exp2 space) is exactly the regime
  the generative bridge cannot yet reach. "Reach deeper compositions while
  retaining structure" is a sharper open problem than "find exotic primitives."

### 2.4 expGG's "closure" of the extraction wall — right conclusion, wrong-looking proof obligation

The conceptual argument ("exact length-gen ⟹ effectively finite-state ⟹ a
human-readable FSM provably exists") needs care: the existence of an FSM for
single-digit-mult carry follows from the **function being regular** — full stop;
no property of the net is needed. The net-side inference ("no drift to w30 ⟹ the
*net's mechanism* is effectively finite-state") is a plausibility argument, not a
proof (and the experiment's own extraction failure at 0.94/decaying is weak
evidence *against* crisp effective states). The honest closure statement is: *for
regular ops, the readable algorithm exists independently of the net, so
un-extractability can't be hiding a non-human **function**; whether the net's
internal **procedure** is something non-FSM-like remains exactly as open as the
0.71–0.94 extraction gap.* Since the project elsewhere insists procedure ≠
function (carry-save vs carry!), the distinction matters. The practical verdict
(bake discreteness in during training; post-hoc extraction of smeared nets is a
dead end) is unaffected.

### 2.5 The "two walls" are not symmetric pillars

"Target-driven search CONVERGES" rests on ~20 experiments plus a theorem.
"Pure open-endedness DIVERGES" rests on **one** experiment (expW): a straight-line
expression space, distinctness-only archive, 26-entry recognition list, one run.
Session 10 then showed a target-free, objective-free system (the soup) that
**converges to meaning** — which the docs absorbed by renaming the second wall's
scope ("distinctness diverges; survival is the counterpart"). That's the right
update, but `03` #9 and `05` still present "the two walls + the narrow band" as
the central load-bearing geometry, with the divergence side carrying far more
rhetorical weight than its evidence. The honest statement: *one* anti-objective
(behavioral distinctness over a straight-line space) diverged; survival selection
did not; the general claim "removing the objective diverges without an
intrinsic-meaning signal" is a reasonable reading of two data points, not a
mapped wall. (The framework survives — the soup *is* an intrinsic-meaning signal
in the docs' own taxonomy — but a "wall" demonstrated n=1 should be labeled as
such.)

### 2.6 Statistics: the CA validations are symmetry-inflated; comparative claims are mostly n=1

- The "7 class-4 rules" ground truth (`expY_ca.py:75`: {54,106,110,124,137,147,193})
  is **3 equivalence classes** under mirror/conjugation: {110,124,137,193},
  {54,147}, {106}. So expY's "6/7 in the top ~10%" is **2/3 independent objects**
  (it found the 110-family and the 54-family, missed 106), and expAA's
  "2/7 → 5/7 → 6/7" precision climb counts four copies of rule 110. The
  multi-signal result is real but its effective n is 3, not 7 — worth one honest
  sentence in `05` wherever 6/7 appears.
- Single-seed comparative results stated as general lessons: gpu_exp2 (one seed,
  see 1.2), expBB (one seed/condition — self-flagged, fine), the exp1 "ceiling
  holds at scale" (3 seeds, one space — fine as stated), soup emergence rates
  ("~1/3" vs "~2/3" across entries — `08` reconciles honestly). The project's
  *correctness* claims are exact-verified (immune to this), but its
  *comparative* claims (X beats Y, signal A resolves conflation B) almost never
  have seed-level replication. This is the single cheapest class of improvement
  for the next phase.

### 2.7 The recipe's honesty floor is missing one item: the budget is a target in disguise

`02` §6 documents the validity filters and given primitives scrupulously. One
scaffolding item is systematically left off the "given" lists: the **efficiency
budget itself**. The step caps were chosen by the experimenter knowing which
algorithm fits under them (expK's cap is what rejects subtractive GCD; expO's cap
kills the linear scan; the rank budget R *is* the Strassen criterion). The docs do
recognize this at the meta level ("correct + optimal-under-a-budget is exactly the
criterion by which humans found the canonical algorithm" — the rediscovery-engine
theorem), but per-experiment "discovered from outcome alone" entries should say
"outcome + a budget tuned to the known algorithm's profile." It does not weaken
contribution 1 (whose honest framing is now "a rediscovery engine by
construction"); it just finishes the floor-disclosure habit.

---

## Part 3 — What the audit does *not* change

For fairness and to prevent over-correction:

- **The composition story (Part I) stands as documented.** Spot-checks (expBB
  results file, expGG log, expDD log, expFF log, expM/expJ/expK internal
  consistency, session-1 `audit.py` having verified the early checkpoints) found
  no discrepancies. The "rediscovery engine by construction" framing is the right
  one and is *strengthened* by this audit (2.7).
- **The bridge-signal validations (expX/Y/Z/AA/DD) hold** with their self-flagged
  caveats, modulo the effective-n note (2.6). The signal-craft-bug catalog
  (`02` §8 / `08` §C) re-verified cleanly and is the most reusable thing in the
  consolidation.
- **The survival-bridge arc is the best-evidenced part of session 10**: soup
  emergence (rep 0.25 vs ctrl 0.037 — verified in `alife_log.json`), settling
  (flat ops), metabolism plateau (merit 1.07→3.0 then flat, ladder empty —
  verified), avida climb (XOR 12674/16384 = 77%, EQU 11714/16384 = 71% at gen
  3000 — verified in `avida_log.json`). Only the final oe link (1.3) is weak —
  unfortunately it is the most moonshot-relevant link.
- **Wall #1 (representational), #2 (complexity-class), #7 (optimizer-cost) are
  intact.** The reversal crossing is real (2/3 seeds, verified). Wall #4's
  *direction* (rugged needles; evolution≫sampling at scale) is intact with the
  Tmax/seed caveats. Walls #3 and #5 are intact **with the scope corrections**
  (2.1, 2.2). Wall #6 needs restating (2.3). Walls #8–#11 unchanged.
- **Verified-exact numbers**: expFF table, expBB table, qd_result.json values,
  exp1 survivor rules/scores (0x59DABC24/1.370, 0x96402558/1.656,
  0xACB27BE8/1.524), census thresholds (18213/9458/3211), expDD driver table,
  lprog 2776, exp3-rev tables, exp3-mul len-gen 0.000s, metab/alife/avida values.

---

## Part 4 — Where prior strategic readings steered the project (the briefing's specific question)

Three places where a plain-read locked in an interpretation without testing the
competitor:

1. **exp3-mul** → "fits but memorized" (1.1). Competitor reading ("fits only
   narrow widths; training unstable") was checkable in 10 minutes from the same
   log and is correct. The locked-in version mattered: it made wall #1 look
   half-crossed ("fitting achieved, discovery missing"), which made "richer
   substrate" look closer to paying off than it is.
2. **gpu_exp2** → "QD diffuses budget across niches" (1.2). The mechanism was
   asserted from the QD literature's own vocabulary; the competitor readings
   (archive corruption bug; descriptor misalignment; 676-elite capacity vs 4096)
   were never separated. The locked-in version produced a general "QD is the
   wrong tool" rule in `08` from one corrupted, single-seed run.
3. **gpu_avida_oe** → "everything found is named/composite ⟹ vocabulary is the
   ceiling" (1.3 + 2.3). The competitor reading — "the search found *unrecognized*
   structured functions and the 10-entry recognizer couldn't name them" — is what
   the surviving artifacts actually show. The locked-in version wrote Frontier 1's
   research program ("design exotic primitives"); under the corrected reading the
   bottleneck is recognition + depth, which redirects the frontier (2.3).

A fourth, softer one: **expW** → "pure open-endedness diverges" became a wall
rather than a data point (2.5), which framed the bridge as "the narrow band" —
rhetorically load-bearing for sessions 8–9 but never re-examined after the soup
converged.

---

## Part 5 — Recommendations (for discussion with Joe before anything runs)

**Documentation repairs (cheap, do first):**
1. Correct `03` #1 (revert the "fits" refinement), #4 (QD claim → "unresolved;
   buggy run"), #5 (scope to statistical learners / description length), #6
   (restate as depth × naming-density, per 2.3), and `08`'s QD/no-redo entry.
   Add the encoding-freedom fix to #3. Fix the tracker's 680, the `06` merit
   quote, and annotate the avida_oe entry as stdout-unverified.
2. Add an erratum block to `07` Frontier 1 reframing it per 2.3 (recognition- and
   depth-first; primitive exotism is not by itself a lever).

**Cheap experimental closures (each ≤ an evening on the 4060, only if we continue
these lines):**
3. Re-run gpu_avida_oe with per-10-gen match logging + a ≥50-entry named suite +
   persisted matched-function table — settles 1.3 in one run.
4. Fix the ME insert and re-run exp2 (3 seeds, 2 descriptor choices) — settles
   whether QD has any role on needle landscapes.
5. expFF addendum: the project's own program-search recipe vs f_R with
   mixer-adjacent primitives; then scale the key — converts wall #5 into a
   measured description-length wall (2.2).
6. Optional: a high-Tmax (e.g. 10⁶) verify/extend pass for depth evolution, to
   measure where the climb actually stalls (1.2 caveat).

**Direction-level (the audit's main strategic output):** the two frontiers as
written should be revised — Frontier 1 is mis-specified (2.3) and partially
collapses into Frontier 2; the encoding-freedom escape for regular ops (2.1) and
the depth-while-structured problem (2.3) are better-posed candidates, and the
identity-space lead (expV) remains the one direction with a historical base rate
of producing genuinely human-unknown mathematics. To be discussed, not assumed.

**Proposed RULES.md amendments (proposed here, not edited in, per the handoff
instructions):**
- *Claims need artifacts.* Any number in TRACKER/consolidation must exist in a
  file under `runs/`; numbers read off live stdout get an explicit
  `[STDOUT-ONLY]` tag at write time.
- *Comparative claims need seeds.* X-beats-Y claims require ≥3 seeds or an
  explicit `n=1` tag in the entry's Status line. (Exact-verification correctness
  claims are exempt — that's the point of exact.)
- *Render before verdict.* Any "structured / class-4 / clean / settled" judgment
  about a dynamical object requires a rendered artifact saved next to the metric.
  (Promotes the existing lesson to a rule; it was violated at 1.4-exp1 after
  being learned twice.)
- *Loss curves are per-condition.* Never summarize a mixed-width/mixed-task loss
  series by its tail; stratify by the sampled condition before claiming fit.
- *No duplicate-index scatter on CUDA* for archives/elites; use sort+segment or
  `scatter_reduce`, and re-verify any stored winner by re-execution before
  reporting it (the exp2 bug would have been caught by re-running the archived
  best once).
- *Version control.* `git init` + a commit per session. The project currently has
  no history; the consolidation rewrote framings with no diffable record.
- *Scope upgrades explicitly.* When an observation is upgraded to a
  wall/theorem/law, the entry must state the quantifier it holds under (encoding,
  learner class, budget, cap). All three scope holes in Part 2 entered at
  upgrade moments.

---

## Appendix — Verification ledger

| # | Claim | Artifact checked | Verdict |
|---|---|---|---|
| 1 | expFF table (R-sweep, addition control) | `runs/expFF.log` | ✓ exact |
| 2 | expBB 4-condition table | `runs/expBB_results.txt` | ✓ exact |
| 3 | QD 49/6238/154, coverage, plateau 73→154@192 | `runs_pod/runs/exp2_fast/qd_result.json`, `.log` | ✓ numbers; ✗ archive integrity (1.2) |
| 4 | exp2 budget matching | `gpu_exp2_qd.py:239` | ✓ matched evals |
| 5 | expEE 43>26>2, n=5 Tmax 8000 | `expEE_evolve.py` defaults | ✓ 43/26; ✗ "2" is a reporting artifact (1.4) |
| 6 | exp3-rev 2/3 seeds, baseline collapse | `runs_pod/runs/exp3_rev.log` | ✓ |
| 7 | exp3-mul "fits in-distribution, baseline never fits" | `runs_pod/runs/exp3_mul.log` + width reconstruction | ✗ contradicted (1.1) |
| 8 | exp3-mul len-gen 0.000 all widths/seeds | same | ✓ |
| 9 | exp1 survivors (rules, novelty, bpc, damage) | `runs_pod/runs/exp1_r2_s*/survivors.json` | ✓ exact |
| 10 | exp1 s2/s3 "clean class-4" | re-rendered `runs/audit_exp1_s*_grid.png` | s2 ✓ / s3 mixed (1.4) |
| 11 | 2D census 18k/9k/3k | `exp1b_census*.log`: 18213/9458/3211 | ✓ |
| 12 | avida XOR 77% / EQU 71% | `runs_pod/avida_s1/avida_log.json` gen 3000 | ✓ (12674, 11714 / 16384) |
| 13 | metabolism bootstrap→plateau, ladder empty | `runs_pod/metab_*/metab_log.json` | ✓ (1.07→3.0 max→2.33; ladder "" throughout) |
| 14 | soup/alife rep 0.25 vs ctrl 0.037 | `runs_pod/alife_lowmut/alife_log.json` | ✓ (also 0.039, 0.125 variants as stated) |
| 15 | avida_oe "a+b by gen 50 / a^b by gen 100, exact" | `runs_pod/oe_s{1,7}.log`, `oe_log.json` | ✗ no surviving evidence; match=[] everywhere (1.3) |
| 16 | oe merit formula quoted "exactly" | `gpu_avida_oe.py:106,114` vs `06` | ✗ missing ×(0.5+0.5·nov) |
| 17 | lprog 2776 vs tracker 680 | `lprog_survivors.json`, log line 72 | consolidation ✓ / tracker ✗ |
| 18 | lprog class-4 median rank 20 / trivial 202 | `weird_lprog.log` head | ✓ (spot) |
| 19 | expGG 0.68/0.94 + decay | `runs/expGG.log` | ✓ numbers; framing note (1.4) |
| 20 | expDD driver table + winner inspection | `runs/expDD.log` | ✓ exact |
| 21 | expCC ladder proof structure | `expCC_ladder.py` | ✓ sound (bisim BFS is the proof); scope note (2.1) |
| 22 | expY class-4 set, ranks | `expY_ca.py:75` | ✓ code; symmetry note (2.6) |
| 23 | expW 26-op recognition list | `expW_openended.py:129` | ✓ (26 entries) |
| 24 | expV 98→16, 210 digits | `runs/` | ✗ no artifact exists |
| 25 | runs_pod/runs authority + s3 red-herring log | `runs_pod/` listing | ✓ as documented |

*Audit artifacts produced:* `runs/audit_exp1_s{1,2,3}_grid.png` (renders),
the width-stratified loss analysis (reproducible one-liner in 1.1, deterministic
from `Random(777)`).
