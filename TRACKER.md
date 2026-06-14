# TRACKER — foundry

Project memory. Append-only in spirit (errors get dated [ERRATUM] brackets,
never silent rewrites). One entry after every experiment — works, failed,
partial, abandoned. See RULES.md. The parent project's tracker lives at
reference/mathlab/TRACKER.md.

## 2026-06-12 — fork: foundry begins

What I did: forked mathlab (public as `rediscovery-engine`) into this repo
with full history; the entire parent tree is archived under
reference/mathlab/. New mission (PROMPT.md) and rules (RULES.md) at root.
Mission in one sentence: a domain-pluggable search engine that proposes
candidate algorithms, executes them in a cost-metered language, verifies
correctness, attacks its own survivors, compares against known references,
diagnoses search walls, and records every action as training data for
future automation.

Design locked with Joe (full discussion in session logs): two-layer language
(core + molds); recognition by traces not outputs; wall doctor recommends,
never kills (v1); gate order cost-before-heavy-attack with the amendment
that nothing archives without at least a light breaker pass; islands
ecology for evolution (blank-slate / seeded / pressure-chamber / alien-mold,
archive as the only migration channel); win ladder rungs 0-6; calibration
domains = sorting networks (floor), Karatsuba-then-Strassen (middle),
planted-vs-pseudorandom mixer pair C1/C2 (roof). Creativity-injection
protocol: Joe injects, Claude labels (standard / new wiring / rare) and
rules, everything logged.

Context note: the parent's final commits (2026-06-10) already prototyped a
mutate->verify->literature-check loop named "Foundry v1/v2" on polynomial
continued fractions, and triangulated "recognition is the binding wall."
foundry generalizes that prototype; PCFs return later as a domain pack.

Status: DONE
Files: PROMPT.md, RULES.md, reference/mathlab/ (parent archive)

## 2026-06-12 — calibration A v0: walking skeleton, end to end (rung 0 + first blood on rung 1)

What I tried: build step 1 — core language v0 (straight-line, integer-only,
cost-tagged instructions), runner with counter bank + step budget,
comparator-list mold (random/mutate/tidy-by-layering/pour/pretty),
random + hill-climb proposers, judge v0 (fast 0/1-bitmask gate +
canonical verify-on-write through the core runner), flight recorder v0
(append-only JSONL, RunSpec predeclared as event 0), foreman v0
(RunSpec -> loop -> budgets -> report). Domain pack A: sorting networks,
exact + complete verification via the 0/1 principle (fast path: one int per
wire, comparator = AND/OR; trust path: core runner on all 2^n binary
vectors + 25 random integer vectors).

What happened: all four sizes solved and canonically verified in 0.42 s
total [STDOUT-ONLY for the timing; per-run seconds are in report.json]:
- n=3: size 3, depth 3 (bubble size 3), 4832 candidates, 0.06 s
- n=4: size 5, depth 3 (bubble size 6), 4992 candidates, 0.08 s
- n=5: size 9, depth 5 (bubble size 10), 4928 candidates, 0.11 s
- n=6: size 13, depth 6 (bubble size 15), 5152 candidates, 0.13 s
All runs: found_correct=True, verified_canonical=True, certificate
L1-exhaustive-in-bounds, stop reason "correct candidate held through 150
batches without improvement". ~20k events logged (4838/5004/4937/5173
lines). Note: most of each budget was the settle phase shrinking size after
correctness was reached. n=4 found a 5-comparator depth-3 network that is
NOT identical to the classic one used in the module sanity checks — same
cost, different wiring.

What I learned: the loop composes; the fast checker and the core runner
agree everywhere they were compared; throughput is ~40-50k candidates/s
single-core pure Python, so the 4060 is nowhere near needed for this floor
domain. Hill-climb leaves something on the table (n=6 at 13; whether the
true optimum is lower is a shelf question — NOT asserting published numbers
from memory). That gap is the motivation for step 3's evolution/islands and
an exhaustive/SAT proposer, and the shelf will turn "rediscovered a sorter"
into "rediscovered THE sorter, named".

Status: WORKS (n=1 seed per size — comparative claims need more seeds; the
correctness claims are exact and exempt)
Files: engine/ (core_lang, runner, molds, proposers, judge, recorder,
foreman), domains/sorting_networks.py, scripts/run_calibration_a.py,
runs/sorting_networks-n{3,4,5,6}-hill-climb-s0-1781280940/

## 2026-06-12 — step 2: reference shelf (cited bounds) + recognizer v0; calibration n=3..8

What I tried: built the sorting-network reference shelf — three GENERATED
constructions (bubble, insertion, Batcher odd-even mergesort incl. a
sentinel-padding argument for non-power-of-2 n), each self-verified against
the pack's checker at build time so nothing is trusted from memory — plus a
known-bounds table with citations checked today via web search: size
optimality n<=8 Floyd-Knuth (Knuth TAOCP v3 5.3.4), n=9,10 Codish-Cruz-
Filipe-Frank-Schneider-Kamp 2014, n=11,12 Harder 2020; depth optimality
n<=8 Knuth, n=9,10 Parberry 1989, n=11..16 Bundala-Zavodny 2014
(arXiv:1310.6271). Recognizer v0: exact canonical match -> KNOWN(name);
cost-profile comparison; bounds headroom statements; below-proven-bound ->
CONTRADICTS-PROVEN-BOUND (= suspect our own verifier first). NEW is never
emitted by v0 (rule: needs full literature check). Wired into foreman gate 3
+ extended calibration to n=3..8.

What happened (artifacts in runs/sorting_networks-n{3..8}-hill-climb-s0-
1781283145/report.json; total wall time 0.89 s [STDOUT-ONLY]):
- n=3: size 3 depth 3 = proven optimal size+depth; 4832 cands; UNRESOLVED
- n=4: size 5 depth 3 = proven optimal size+depth; 4992 cands; UNRESOLVED
- n=5: size 9 depth 5 = proven optimal size+depth; 4928 cands; UNRESOLVED
- n=6: size 13 depth 6 = +1/+1 above proven optimal 12/5; 5152; UNRESOLVED
- n=7: size 16 depth 6 = proven optimal size+depth; 7968 cands; UNRESOLVED
- n=8: size 22 depth 9 = +3/+3 above proven optimal 19/6; 5728; UNRESOLVED
All L1-verified via the core runner before recognition (verify-on-write).

What I learned: (1) the recognition pipeline works end to end and every
optimality statement now carries its citation. (2) Plain hill-climbing hits
proven-optimal SIZE at n=3,4,5,7 but not n=6,8 — pattern noted, n=1 seed,
no theory claimed. (3) No exact canonical match to any shelf construction:
optimal-cost wirings are plentiful, so rung 1 PASSES on the published table
values (n=3,4,5,7 match proven optima under the declared verifier) but
named-construction rediscovery has not happened; recognizing networks
equivalent up to standard untangling transformations is future recognizer
work. (4) The honest gaps at n=6 (+1) and n=8 (+3) are the work order for
step 3 (evolution/islands + exhaustive proposer).

Status: WORKS (rung 1 on table values for n=3,4,5,7; headroom flagged at 6,8)
Files: domains/sorting_networks_shelf.py, engine/recognizer.py,
engine/foreman.py (gate-3 wiring), scripts/run_calibration_a.py,
runs/sorting_networks-n{3,4,5,6,7,8}-hill-climb-s0-1781283145/

## 2026-06-12 — repo maintenance: GitHub remote + history filter

What: created public remote github.com/Joe-b-20/foundry (Joe approved going
public; reasons: parent already public, timestamped commits = priority
trail, external-validation rung needs visibility). First push was rejected:
the inherited parent history contained one 230.6 MB artifact
(runs_pod/phase2/pcf_main/stage1_survivors.npz) over GitHub's 100 MB hard
limit. Stripped that single blob from foundry's mirror of the history with
git filter-branch (28 commits rewritten — all commit hashes changed,
including the fork-point hashes cited anywhere earlier). The canonical
artifact is untouched in the parent working copy (~/math_lab) and the
parent's own repo; foundry's reference archive simply has a gap where the
one oversized binary was. No other blob exceeds 100 MB on main.

Status: DONE
Files: (history-level change; no working-tree code touched)

## 2026-06-12 — step 3: archive + island ecology + tiny-n certificates; n=6 and n=8 gaps CLOSED outcome-only

What I tried: Archive v0 (content-addressed, verify-on-write re-execution
through the core runner, pareto elites, provenance as a first-class field);
EvolutionProposer (population GA using mold moves, splice crossover,
"pressure" rank = Island C: smallness outranks sortedness, so wrong
intermediates are allowed INSIDE the island — only verified candidates
leave, because the archive is the only bridge and it verifies on write);
islands driver per Joe's injection #1 — blank / seeded / pressure islands;
blank+pressure pull only outcome-only archive entries, so their lineages
stay discovered-from-outcome by induction; Island D deferred until a second
mold exists. Plus independent optimality certificates for tiny n by
exhaustion (enumerate ALL sequences of length optimal-1; monotone-extension
lemma covers shorter lengths).

What happened:
- Certificates (runs/certify_tiny-1781286005/certificates.json): n=3 — none
  of 9 length-2 sequences sorts; n=4 — none of 1296 length-4 sequences
  sorts; Batcher witnesses give the upper bounds. Optimal sizes 1/3/5 for
  n=2/3/4 confirmed with our own machinery, matching Floyd-Knuth.
- BUG FOUND AND FIXED (first islands batch, runs/islands_summary-
  1781286021.json): archive pareto-domination was checked ACROSS provenance
  classes — the seeded Batcher (19/6), admitted at gen 0, blocked every
  outcome-only admission at n=8, silently erasing the outcome-only record
  (outcome-only=None despite live progress). Fix: identity + domination now
  scoped within provenance (key = (provenance, canonical)). The 3-seed +
  provenance-split reporting is what surfaced the bug.
- Fixed batch (runs/islands_summary-1781286099.json; 3 seeds per n):
  n=6 outcome-only reached proven-optimal size 12 in 3/3 seeds (9.8k-14.6k
  evals, 0.15-0.22 s; seed 2 found 12/depth-5 = proven-optimal size AND
  depth, blank island). n=8 outcome-only reached proven-optimal size 19 in
  2/3 seeds (108k-123k evals, ~2.5 s; both finds 19/6 = proven-optimal size
  AND depth; third seed 21 at the 345.6k-eval budget). Seeded island sits
  at Batcher 12/6 and 19/6, and the recognizer labels it KNOWN =
  batcher-odd-even by exact canonical match — the KNOWN path now fires in
  the wild. All outcome-only finds: UNRESOLVED (cost-identical to Batcher
  at n=8 but different wiring).
- The pressure island produced the outcome-only best in 5 of 6 runs: blank
  finds correct ~13/~22, pressure compresses to 12/19.

What I learned: (1) the ecology closes gaps hill-climb couldn't (n=8:
22 -> 19), n=3 seeds; (2) provenance separation must be structural, not
just reported — the archive bug would have quietly converted discoveries
into nothing; (3) cost-exact-but-different-wiring rediscovery is the common
case, so canonical-equivalence-up-to-untangling is the recognizer's most
valuable next feature; (4) exhaustion certificates are cheap and real at
n<=4; n=5 (10^8 sequences) is the SAT proposer's first job.

Status: WORKS (n=6: 3/3 seeds outcome-only at proven-optimal size; n=8:
2/3; certificates n=2,3,4 match the cited table)
Files: engine/archive.py, engine/islands.py, engine/proposers.py
(EvolutionProposer), scripts/certify_tiny_optimality.py,
scripts/run_islands_a.py, runs/ as cited above

## 2026-06-12 — step 4 part 1: PCF domain pack — parent's seeded-walk result REPLICATED through the generic engine (calibration B, part 1)

What I tried: second domain, polynomial continued fractions — the parent's
strongest territory. New organs: engine/numeric.py (owned metered wrapper
over mpmath; eval recurrence, delta, pslq helpers; the parent's hard
lessons baked in: term count scales with precision vterms=max(1400,dps*9) —
their v2 control failure was exactly a fixed term count — plus the rational
trap and the multi-constant >=3 triviality trap); engine/molds_pcf.py (PCF
mold); domains/pcf.py (pack: two-stage verification — 60-dps screen with
Mobius pslq [1,C,v,vC], then 250-dps re-verify with scaled terms and
residual < 1e-238; certificate level "numeric-250-digit, conjecture-grade,
RM-style" — NOT proof, wording bounded accordingly); domains/pcf_shelf.py
(battery of 8 constants, no algebraics; kappa + Table-7 families of
arXiv:2210.15669 per the parent's parameterization, plus rm-8-7zeta3 as the
zeta3-class control). Shelf is self-verified: 21/34 members pass the full
pipeline; 13 rejects (parameter-degenerate cases, e.g. j=0 makes b(1)=0 ->
truncation -> rational under our indexing convention) are logged, not
hidden. Replication experiment: 4 kappa seeds, 14 generations, 8 children/
gen, 3 controls (rm-8-7zeta3, kappa(0,0), kappa(1,0)) re-verified fresh
EVERY generation; any control failure voids the run.

What happened (3 seeds; runs/pcf-replication-s{0,1,2}-1781287{481,517,521}/
report.json; earlier runs in the same prefix are the failed iterations,
kept):
- Controls: 42/42 C+ in every run (294 control checks total across all 7
  runs including the failed iterations; zero failures). Parent's was 18/18.
- All 3 seeds: a mutant walked BY FORM to kappa(k=0,c=2) — a published
  family member not among the seeds (gens 5/7/2; ~3.3 s per run, CPU only).
- Zero unexplained novel-flags in all runs: clean control-gated null,
  matching the parent's v2 endpoint.
- THREE honest catches en route, each its own small wall lesson:
  (1) Mold v1 mutated dense polynomial coefficients -> 109 mutants, zero
  members reached. The published families are parameterized by FACTOR
  shifts; one family step is one move in factored space but a far junk jump
  in coefficient space. Rebuilt the mold factored (the parent's B-shift/
  B-split grammar). Moves define the geometry — the representational wall,
  PCF edition.
  (2) Value-level reference subtraction can only name the Mobius CLASS:
  the whole catalan family is one Mobius orbit, so every hit "matched" the
  first member in list order. Member identity needs STRUCTURAL matching.
  (3) Factored forms are not unique (n^3(n+2) == n^2(n)(n+2)) AND distinct
  published parameterizations can be the same polynomial — kappa(1,0) ==
  table7(i=1,j=1,mu=0) expand identically. Canonical key = dense expansion;
  "reached" requires the polynomial to differ from every seed's polynomial.
  Before this fix the run claimed table7(1,1,0) as reached when it was a
  seed under another name. (Neutral note: the parent's run-1 also reported
  reaching (1,1,0); whether that was the same aliasing cannot be checked
  from here and is not asserted.)

What I learned: the recognizer ladder (canonical/structural first, value
second) is not optional in numeric domains — value equivalence is class-
level only. The engine now demands its first real refactor: judge v0 is
sorting-shaped; pack-polymorphic judging (each pack provides its own gate-1
path) is the next structural change. delta values here are internal-
relative (our formula; not comparable to the parent's absolute numbers
without checking normalization).

Status: WORKS — calibration B part 1 PASSED, 3/3 seeds (controls perfect,
member reached by form, clean null). Part 2 (scoped in-grid sweep
rediscovery with positive control, Apery-class) still ahead.
Files: engine/numeric.py, engine/molds_pcf.py, domains/pcf.py,
domains/pcf_shelf.py, domains/pcf_refs.json (cached verified shelf),
scripts/run_pcf_replication.py, runs/pcf-replication-*

## 2026-06-12 — earned refactor: pack-polymorphic judge + domain registry

What/why: judge v0 hardwired the sorting-network gates; the PCF pack was
the second domain demanding the shape (RULES: no abstraction until a
second domain forces it — this is that moment). Now: engine/registry.py
maps domain name -> (pack, mold); the judge is a thin domain-blind
dispatcher; each pack provides gate1(mold, tidy) -> (score, cost) and
verify_trusted(mold, cand) -> (ok, details) built on engine services (core
runner / numeric engine). Foreman and islands build via the registry
(islands driver still sorting-flavored in its seeds/bounds, stated in an
assert). Regression: all 13 module sanity checks pass; calibration A
reproduces identically (same sizes/depths per seed); islands smoke at n=4
now reaches outcome-only optimal size 5 at gen 10 — pre-provenance-fix it
could not (the seeded Batcher blocked admission), consistent with the
step-3 bug analysis.
Status: DONE
Files: engine/registry.py, engine/judge.py, domains/sorting_networks.py,
domains/pcf.py, engine/foreman.py, engine/islands.py

## 2026-06-12 — step 4 part 2: scoped in-grid sweep — RM 8/(7 zeta3) REDISCOVERED FROM OUTCOME (calibration B COMPLETE)

What I tried: the parent's flagship from-outcome rediscovery (its commit
c104290, run on a GPU pod) at a CPU-sized scope, everything predeclared in
the RunSpec before running: grid a(n)=c0+c1 n+c2 n^2+c3 n^3 with c0 1..8,
c1 1..30, c2 0..60, c3 1..40, b(n)=-n^6 (585,600 candidates; contains both
Apery (5,27,51,34) and the target (1,5,9,6)); stage 1 = vectorized float64
recurrence, 150 terms, renormalized, keep converged values within 1e-9 of
a Mobius net (p+qC)/(r+sC), |coeffs|<=8, det!=0, C in {zeta3, pi^2,
catalan}; stage 2 = full two-stage mpmath verification + structural naming
+ reference subtraction. Discovery is outcome-only: the prefilter sees
battery CONSTANTS only, never shelf forms; naming happens after. Apery =
in-grid positive control (fail -> VOID). Null arm = 100k random vectors
from the disjoint box c3 41..80, prediction zero verified survivors.
Apery's fraction was added to the shelf first (citations: Apery 1979; van
der Poorten 1979), shelf cache rebuilt.

What happened (runs/pcf-sweep-s0-1781288694/report.json [ERRATUM
2026-06-12 audit: wrong timestamp, that path does not exist — the real
artifacts are runs/pcf-sweep-s0-1781288327, s1-1781288353, s2-1781288355]
+ s1/s2 runs):
585,600 candidates -> 581,233 converged -> 23 prefilter survivors -> 2
verified: (5,27,51,34) = KNOWN apery-zeta3 (control C+) and (1,5,9,6) =
KNOWN rm-8-7zeta3, the target, verified at 250 digits and named BY FORM.
The other 21 survivors all dropped as no-match at the 60-dps pslq screen
(float64 prefilter near-misses — the two-stage design doing its job).
Null arm: 0 stage-1 survivors in all 3 seeds (300k samples total). Wall
time 2.9 s CPU. PASS.

What I learned: (1) the parent's pipeline shape (cheap-float prefilter ->
exact verification) ports cleanly and the scoped version costs seconds,
not pods; (2) the Mobius-net prefilter at 1e-9 has a measured-zero false-
positive rate on the null box at this scale, and its 21 in-grid false
positives were all caught downstream; (3) SCOPE of the rediscovery claim:
within the predeclared box, prefilter constants, coeff bound 8, and
tolerances above — no claim beyond it. Calibration B (replicate the
parent's best) is now COMPLETE: seeded walks (part 1, 3/3 seeds) + from-
outcome sweep rediscovery with control and null arm (part 2).

Status: WORKS — calibration B COMPLETE. (Main arm is exhaustive over the
declared grid, so no seed variance there; null arm n=3 seeds.)
Files: scripts/run_pcf_sweep.py, domains/pcf_shelf.py (apery added),
domains/pcf_refs.json (rebuilt), runs/pcf-sweep-s{0,1,2}-*

## 2026-06-12 — Karatsuba REDISCOVERED AND NAMED from outcome (decomposition mold; third domain)

What I tried: the bilinear-decomposition mold — candidates are R products
(u.a)(v.b) plus integer recombination; the induced tensor
T'[i][j][k] = sum_r u_r[i] v_r[j] w_r[k] must equal the target tensor
entry-for-entry. That identity is INTEGER-EXACT and by bilinearity proves
correctness for ALL inputs — certificate "L1-exact-tensor-identity", the
strongest level foundry has issued (plus a poured-to-core cross-check on
50 random integer inputs; this is the first new mold that pours to core).
Target = polymul2 (degree-1 polynomial multiplication; Karatsuba's R=3 vs
naive R=4). Bounds: rank >= 3 PROVEN by the flattening/slice-span argument
COMPUTED EXACTLY in domains/bilinear_shelf.py (integer row reduction shows
the three output slices are independent; slices of a rank-R decomposition
span <= R dims), witness Karatsuba & Ofman 1962 -> R=3 proven optimal, so
the recognizer's CONTRADICTS-PROVEN-BOUND logic applies below it. Search:
two islands (the validated pattern) — blank with exact-first lexicographic
rank, pressure with a STOKE-style blended scalar (l1 + 0.45R + 0.02dl) so
near-miss R=3 candidates survive in-population and the R=4 -> R=3 valley
is crossable through wrong intermediates; exact-only migration every 20
gens. Recognition by canonical key over the target's symmetry group
(a<->b swap + index reversal). EvolutionProposer gained an optional custom
rank `key` (backward compatible). Registry: third domain, "bilinear".

What happened (runs/bilinear-karatsuba-s{0,1,2}-*/report.json):
- 3/3 seeds: exact verified R=3, all named KNOWN by exact canonical match:
  s0 gen 204 (blank) = karatsuba-3 (+,+); s1 gen 173 (PRESSURE) =
  karatsuba-3 (-,-) — the other sign variant; s2 gen 164 (blank) = (+,+).
  Citations attached (Karatsuba & Ofman 1962).
- Naive R=4 rediscovered first in every seed (gens 40/68/48) — the rung-1
  baseline — then compressed to R=3.
- 21-26k evals, 0.4-0.5 s per seed, CPU. ~1.4 s for the whole experiment.

What I learned: (1) FIRST fully-closed named-rediscovery loop: discovered
from outcome -> verified exactly -> named KNOWN with citation (sorting
finds stayed UNRESOLVED wiring-variants; PCF naming was post-hoc by form;
here symmetry-group canonicalization nails identity). (2) The parent
needed custom scaffolding ("Strassen tensor method") for Karatsuba; the
generic pipeline did it with zero domain-specific search code — the
calibration-B "must not need hand-holding" bar, met. (3) Both Karatsuba
sign variants appear across seeds — the solution space structure is
visible in the archive of runs. (4) The blended-scalar pressure rank
works where lexicographic pressure couldn't cross valleys; worth folding
back into the islands driver someday.

Status: WORKS (3/3 seeds; predeclared bar met). The middle calibration
item is done; Strassen (matmul tensor, R=7) plugs into the SAME pack as
data when we want it. Remaining calibration: C1/C2 + the wall doctor.
Files: engine/molds_bilinear.py, domains/bilinear.py,
domains/bilinear_shelf.py, engine/registry.py, engine/proposers.py (key
param), scripts/run_karatsuba.py, runs/bilinear-karatsuba-s{0,1,2}-*

## 2026-06-12 — C1/C2 + wall doctor: EXAM PASSED 6/6 — calibration ladder COMPLETE

What I tried: the roof. Bit-program mold (candidates ARE core programs
over XOR/AND/OR/ADD/MUL on 4 slots; +,*,&,|,^ commute with mod-2^w
reduction, so the unmodified core runner executes them and only the final
output is masked — no bit-width machinery). BitMixer pack: hidden f behind
an I/O interface; C1 = planted program from the mold's own op set
(non-trivial-filtered; length = difficulty dial), C2 = 8-round
rotation+key ARX-style mixer (rotation NOT a primitive; 64-bit key folded
in every round — the parent's expFF class). gate1 = bit-agreement on a
corpus (graded signal); verify = exhaustive, all 2^16 input pairs at w=8
through the core runner (L1-exhaustive). Wall doctor v1
(engine/doctor.py): plateau detection + comparison against one-op
"chance-plus" baselines; verdicts in the agreed scoped format, exactly one
recommendation, RECOMMENDS-ONLY; operator = predeclared logged policy
(accept after 2 consecutive abandon recommendations).

EXAM v1 FAILED HONESTLY, twice over (runs/doctor_exam_summary-1781290969):
planted_len=4 was trivial (random init solved it at gen 0-1 — wrong-quit
direction never exercised) and keyed at a 384-bit corpus let evolution
reach 0.61-0.63 agreement — which was CORPUS OVERFIT, not signal, and the
doctor had no way to tell. Fix, and it is now a permanent doctor feature:
a HELD-OUT set from a distinct stream (search never sees it; the doctor
judges generalization on it — plateau + corpus >> heldout = memorization).
Also planted_len -> 6, corpus -> 128 + heldout 256, mixer -> 8 rounds.

EXAM v2 PASSED 6/6 (runs/doctor_exam_summary-1781291101.json):
- C1 planted, 3/3: found at gens 116/63/475 (real time for the doctor to
  wrongly quit; ZERO false alarms), each verified exhaustively on all
  65,536 pairs. In s0 and s1 the finds are SHORTER equivalents of the
  plants (3 ops vs 5 planted) — compression beyond the plant. (Note: the
  printed per-run "heldout" is the last pre-find diagnostic reading, not
  the found program's; verified_exhaustive subsumes it.)
- C2 keyed, 3/3: corpus best 0.55-0.58 vs heldout 0.507-0.511 vs chance+
  0.507-0.527 -> doctor: scoped verdict, wall-smell
  learnability/pseudorandom, recommendation abandon; operator accepted at
  gens 1400/1000/850. No exact found, obviously.
- Total exam wall time ~16 s.

What I learned: (1) the doctor can now pass both directions of its exam —
it does not quit on findable targets and does not grind on hopeless ones,
within this exam's scope (w=8, this op set, these budgets); (2) the
held-out generalization check is not optional for learned-signal domains —
corpus fit lies (mathlab's stratify/fit lessons, now structural); (3) exam
difficulty dials matter: a trivial C1 tests nothing — planted_len 6 at
gen 63-475 is the useful range for this budget.

Status: WORKS (6/6, predeclared bar). CALIBRATION LADDER COMPLETE:
A floor + B PCFs (walks + sweep) + B-prime Karatsuba + C1/C2 roof.
Files: engine/molds_bits.py, domains/bitmixer.py, engine/doctor.py,
engine/registry.py, scripts/run_doctor_exam.py,
runs/bitmixer-*, runs/doctor_exam_summary-*

## 2026-06-12 — grokking stress-test (Joe's question): does the doctor kill slow-reorganizing searches?

Question (Joe): a search can look dead for thousands of gens then suddenly
reorganize (grokking). Would the doctor kill it first?

Honest pre-answer: the plateau+chance rule COULD kill a search that sits
exactly AT CHANCE then jumps — and that case is information-theoretically
near-identical to the cryptographic wall (no learnable partial signal = the
PRF property), so no outcome-only detector can separate "flat-at-chance
grok" from "dead". Not a fixable bug; a near-fundamental limit.

What I tried (scripts/run_grokking_probe.py): maximize the doctor's chance
to wrongly quit — reachable planted targets, early min_gens=200 so it
diagnoses through the whole run, a deceptive conjunction plant
out=(x^y)*(x+y) (building-block plateau) + random hard len-7 plants, doctor
OBSERVE-ONLY (logs the verdict it WOULD give, never acts). Measure: any
would-abandon STRICTLY before the find?

What happened (runs/grokking-probe-*/report.json): 0/6 runs would have been
killed before the find — INCLUDING a genuine grok (deceptive seed 1: flat
at corpus 0.755 / held-out 0.761 for gens 0-2300, then SOLVED at gen 2303).
Mechanism, confirmed from the trace: held-out 0.76 >> chance+ 0.62, so the
doctor said "switch/raise budget" the entire plateau, NEVER "abandon".
The abandon verdict is gated on held-out AT CHANCE, not on mere flatness —
so a grok carrying above-chance partial signal during its plateau is routed
to patience, not death. (A deceptive seed-2 run parked at 0.907 for 6000
gens without finding: also never abandoned — correctly "switch/raise".)

Hardening applied (motivated by the measurement, not speculative):
- Doctor verdicts now CONFIDENCE-GRADED. A high-confidence abandon requires
  the at-chance plateau to persist past abandon_min_gens (asymmetric
  patience: killing a discovery costs >> running a dead search longer).
  Below that horizon an at-chance plateau is "abandon, low confidence".
- Operator policy tightened: accept an abandon only on HIGH confidence +
  2 consecutive (was: any abandon + 2 consecutive).
- C3 added as a PERMANENT exam case (out=(x^y)*(x+y)): PASS = the doctor
  never issues a high-confidence abandon on this reachable deceptive target.
  Locks in grok-survival against future doctor changes.

Exam v2 (9 universes, runs/doctor_exam_summary-1781292497.json): PASS 9/9.
C1 planted found gens 116/63/475 (0 false alarms); C2 keyed abandoned at
1650/1650/2700 (later than v1's 850-1400 — the asymmetric patience, still
within budget); C3 deceptive never confidently abandoned (found 19/19/492
under the exam's search seed; the slow-grok stress lives in the probe).

OPEN (honest): the residual at-chance-then-jump grok is undetectable from
outcome alone. Next defense to BUILD ONLY WHEN an at-chance grok can be
manufactured to test it: a "motion underneath" signal — population novelty
/ description-length still moving while best-so-far is flat (the parent's
bridge idea). Not built now: would be an untested mechanism for an
unmanufactured case (violates measure-before-claim).

Status: WORKS — doctor stress-tested against grokking; 0/6 wrongly killed;
hardened + C3 locked in. Calibration ladder stays complete (now 9/9).
Files: scripts/run_grokking_probe.py, engine/doctor.py (confidence +
asymmetric patience), domains/bitmixer.py (explicit plant param),
scripts/run_doctor_exam.py (C3), runs/grokking-probe-*,
runs/doctor_exam_summary-1781292497

## 2026-06-12 — self-audit + full-engine re-proof + documentation refresh

What I tried (while Joe thinks over the portfolio): a systematic audit of
everything since the fork — programmatic probes (scripts/run_audit_checks
.py, kept rerunnable), independent re-execution of every archived winner,
claim-vs-artifact reconciliation against this tracker, and a code review
pass. Then built the GAUNTLET (scripts/run_proof_phase.py): one command
that re-runs the entire proof-it-works phase on the current engine — 19
module sanity blocks + calibration A + exhaustion certificates + islands
+ PCF replication (3 seeds) + PCF sweep (3 seeds) + Karatsuba + the 9-
universe doctor exam + the grokking probe + the audit itself.

What the audit found (full report: docs/audit_2026-06-12.md):
- ONE WRONG ARTIFACT PATH: the sweep entry cited runs/pcf-sweep-s0-
  1781288694, which never existed (real: 1781288327/-353/-355). Dated
  [ERRATUM] added in place. This is exactly the error class the claims-
  need-artifacts rule exists for; the audit probe now checks every cited
  path automatically.
- ONE INCOMPLETE EXPLANATION: PCF shelf rejects had TWO mechanisms, not
  one — 9/13 are j=0 -> b(1)=0 truncations, 4/13 (table7 j=2 corners) are
  genuine telescoping rationals (verified values: 6, 4, 10, 60/7). The
  trap caught all 13 correctly; my explanation covered only the first
  mechanism.
- CLEAN: all 63 archived winners re-executed independently, 0 failures;
  sweep findings re-verify through a fresh pack.
- FIXED: machine-readable found candidates now persisted in karatsuba +
  doctor-exam reports (artifact completeness); doctor exam now verifies
  corpus-exact candidates exhaustively AT FIND TIME and continues on
  failure instead of ending the run; PCF verify flags 2-constant
  ambiguity; audit tooling's own bugs (brace expansion, self-matching
  TODO census).
- DOCUMENTED (not hidden): standalone breaker still roadmap; archive
  pareto keys and recognizer module sorting-flavored; delta not comparable
  to RM-published values (unreduced q).

Re-proof: PROOF PHASE PASS — 19/19 sanities, 9/9 stages, 108.6 s
(runs/proof_phase-1781294922/summary.json). The whole ladder reproduces
on the post-audit engine.

Docs refreshed: README rewritten with two up-to-date mermaids (the machine
map incl. roadmap items marked as such; candidate lifecycle), the ladder
results table, and the gauntlet command. PROMPT.md calibration section
now includes C3. RULES.md gains two earned rules (held-out generalization
is the only claim basis in learned-signal domains; exams must be able to
fail) and the run-the-gauntlet-after-engine-changes practice.

Status: DONE — audit clean apart from the items above (all corrected or
documented); full proof phase re-verified on the current engine.
Files: scripts/run_audit_checks.py, scripts/run_proof_phase.py,
docs/audit_2026-06-12.md, README.md, PROMPT.md, RULES.md,
runs/audit-*, runs/proof_phase-1781294922

## 2026-06-12 — PORTFOLIO DOMAIN #1: rsqrt (numerical approximants). Magic constant found from outcome: 0x5F375A87, one step from Lomont's, marginally better on our scope

Domain rationale (Joe picked numerical approximants — exp/log/sqrt/rsqrt/
sin/cos/tanh/sigmoid/erf/softplus/gelu; scoped to start with rsqrt, tanh
next): extremely load-bearing; float32 unary input spaces are FINITE so
verification is EXHAUSTIVE (this domain's 0/1 principle); the polynomial
corner is theory-solved (Remez/minimax — that's bounds machinery, not a
competitor), so headroom lives in bit/float hybrids and the sloppy
ML-activation corner. Citations checked: Quake III 0x5F3759DF (1999);
Lomont 2003 0x5F375A86, bound 1.751302e-3; Moroz/Walczyk/Cieslinski
(arXiv:1802.06302 etc.) modified-Newton class — their exact constants NOT
carried (rules: no shelf numbers from memory).

Built: core language's first real extension — 32-bit BITS-TYPED op subset
(FADD/FSUB/FMUL + ADD32/SUB32/SHR32/SHL32/XOR32; ops interpret a slot's
low 32 bits, so int<->float reinterpretation is free, exactly how the
trick family sees the world; FDIV excluded: float64 intermediates would
double-round division — add/sub/mul are exact). FloatProgMold: programs +
4 CONSTANT GENES (bit-flip mutations; mold-aware crossover mixes constant
genes — the generic sequence splice cannot handle (instrs, consts) pairs).
RsqrtPack: declared scope all float32 in [2^-8, 2^8) (134,217,728 values);
gate1 = shaped search loss; CLAIM metric = exhaustive max rel error vs
float64 reference; verify_trusted = exhaustive sweep + 256-input bit-exact
cross-check against the core runner. Shelf self-verified: our exhaustive
measurement reproduces Lomont's published bound DIGIT-FOR-DIGIT
(1.751302e-3) and Quake's at 1.752339e-3; seed-only at 3.437577e-2.

THREE SEARCH-DESIGN FAILURES en route, each measured and kept:
(1) constant-output attractor: raw max-rel fitness saturates near 1.0 for
"output a tiny constant" while bit-trick exploration scores astronomically
worse -> total collapse (summaries -1781296565, -1781296725; the shaped
log-loss alone did NOT fix it because (2) global truncation selection
culls any temporarily-bad lineage — scaffold seeds with random constants
died in gen 1. Niche protection by structure-freezing fixed retention
(-1781296890). (3) Coordinate search traps: single-bit descent cannot
cross multi-bit boundaries (shift gene stuck at 2; 2->1 is ...10 -> ...01,
two simultaneous flips — summary -1781297088); carry deltas fixed that but
one-gene-at-a-time descent cannot enter the Newton basin (c2~0.5 AND
c3~1.5 AND magic c0 must move together; -1781297202).

ARM B1 (the rung-1 result; structure + DERIVED Newton coefficients (3/2,
1/2 — calculus, not discovered magic) given; THE CONSTANT searched from
outcome — historically faithful to what Lomont's numerical search
optimized): coarse-to-fine 1-D global sweep over all 2^32 constants
(8192-point coarse grid on the sample metric -> recursive refinement ->
2M-point dense grid -> final +-8 fine scan under the EXHAUSTIVE metric).
Result, 3/3 seeds, deterministic: c0 = 0x5F375A87, exhaustive max rel
error 1.751288e-3 — ONE integer step from Lomont's 0x5F375A86 and
MARGINALLY BETTER on our scope/metric (1.751288e-3 vs 1.751302e-3; the
1.4e-8 gap is far above the ~1e-16 reference noise). HONEST FLAG, not a
claim of beating Lomont: his optimum may be over a different scope (all
normals) or a slightly different error definition; whether 0x5F375A87
refines or merely ties his constant needs a read of the primary paper
before anything is said publicly. Label: UNRESOLVED-novel-flag pending
literature check. (~38 s/seed; runs/rsqrt-B1-s{0,1,2}-*,
summary runs/rsqrt_hunt_summary-1781297607.json, PASS predeclared bar
3/3.)

OPEN, honestly: arm B (all four constants jointly from outcome) 0/3 —
needs a coupled-variable optimizer (CMA-style or nested descent); arm A
(open structure from nothing) 0/3 — the constant-output attractor is this
domain's landscape wall; needs diversity machinery (novelty/MAP-Elites),
which is a real next-session research item, and the doctor should learn
this domain's baselines.

Status: WORKS (B1 3/3, predeclared) + two honest OPENs. Next: tanh pack
(Remez bounds machinery), joint-constant optimizer, arm-A diversity.
Files: engine/core_lang.py + engine/runner.py (32-bit subset),
engine/molds_float.py, domains/rsqrt.py, domains/rsqrt_shelf.py,
engine/registry.py, scripts/run_rsqrt_hunt.py, runs/rsqrt-*

## 2026-06-12 — the 0x5F375A87 flag RESOLVED: wins the full domain under our metric (all 2.13e9 positive normal float32)

Primary source read (Lomont 2003 PDF, fetched + text-extracted): his table
tests constants "over all floating point values" against the FLOAT32
reference (float)(1.0/sqrt(x)) and reports 0x5F375A86 best at 0.175124%
after one Newton step (0x5F3759DF at 0.175228%). Our metric uses a
float64 reference — the two differ at the ~6th significant digit, so
neither ranking supersedes the other; ours is stated alongside his.

Decisive sweep (scripts/run_rsqrt_fullscope.py; runs/rsqrt-fullscope-*/
report.json; 187 s): exhaustive max rel error vs float64 reference over
ALL 2,130,706,432 positive normal float32, standard structure + one
Newton (3/2, 1/2):
  0x5F375A87 (ours)    1.751287782e-3   <- best of the tested set
  0x5F375A85           1.751291588e-3
  0x5F375A88           1.751300410e-3
  0x5F375A86 (Lomont)  1.751301558e-3
  0x5F3759DF (Quake)   1.752338672e-3
Scope of the claim, exactly: under THIS metric, 0x5F375A87 beats the
published constant and all +-2 neighbors on the full domain
(L1-exhaustive certificate). The constant was selected by a global
coarse-to-fine search on a 16-octave sub-scope, then confirmed on the
full domain — a constant-exhaustive x input-exhaustive proof of global
optimality was NOT performed (2^32 x 2.13e9). The optimum is extremely
flat (5th-6th digit); the practical value of the improvement is nil; the
demonstrated value is the pipeline: outcome-only search + exhaustive
certificates reproducing and marginally refining a 20-year-old famous
result in ~3 minutes of CPU.

Status: WORKS — flag resolved, claim filed at its honest size.
Files: scripts/run_rsqrt_fullscope.py, domains/rsqrt.py (full-range
scope), runs/rsqrt-fullscope-*

## 2026-06-12 — tanh: weighted-Remez floors + rung-1 calibration lands ON the proven floor (6/6)

Built: engine/remez.py — Remez exchange with the de la Vallee Poussin
bracket (proven floor for the whole degree-d polynomial class; numeric at
40 dps), extended to WEIGHTED equioscillation after the first shelf build
exposed a metric mismatch (absolute-error minimax is ~4x off the pack's
relative metric at tanh(0.25)≈0.245; weight=f fixes it — generalized
equioscillation covers positive weights). TanhPack subclasses RsqrtPack
(truth = float64 numpy.tanh; scope all float32 in [0.25, 8), 41,943,040
values, exhaustive; generalization deferred per the rule of three).
Shelf: weighted-Remez minimax polys deg 3/5/7, float32-rounded, each
exhaustively measured: E_f32 sits 0.04-0.2% above its real-model floor
(deg5: 9.9555e-3 vs floor 9.9495e-3) — exactly the expected rounding
inflation; FPminimax-style float-native fitting noted as future craft.
FloatProgMold gained a parameterized constant pool (Horner deg d needs
d+1 genes).

Calibration (coefficients FROM OUTCOME, Horner skeleton given;
scripts/run_tanh_calibration.py): FOUR optimizer failures on the way,
each diagnosed and kept:
(1) cyclic bit-descent under the max metric from cold: all FAIL at
E~0.6-1.0 — coordinate descent PROVABLY stalls on nonsmooth objectives
(max-of-kinks); (2) two-phase shaped-then-max: still stuck — the shaped
loss's 1e6 clip created a gradient-free plateau over garbage-coefficient
space; UNCLIPPED log1p restored slope everywhere (fix now in the packs);
(3) plain least-squares init: lands in-basin but 1.4-1.8x off minimax
(the L2-vs-minimax gap); (4) Lawson IRLS at 40 iterations + shaped-phase
polish: 1.13x off — Lawson converges linearly (needs patience) and the
shaped polish pulls a near-minimax point AWAY toward the mean-optimum.
Final recipe: Lawson IRLS (150 iters, relative-error rows, outcome
samples only) -> float32 round -> bit-descent polish under the TRUE
metric. RESULT 6/6 PASS: deg3 E=1.0632e-1 (floor 1.0628e-1, 0.04% above);
deg5 E=9.9582e-3 (floor 9.9495e-3, 0.09% above) — the from-outcome search
lands essentially ON the proven optimal floor. ~12 s total.

Gauntlet re-run after all engine touches: 19/19 + 9/9 PASS (103 s,
runs/proof_phase-1781298930).

Status: WORKS — tanh rung 1 done against our own proven floors. Next:
the actual HUNT (beat the polynomial frontier with bit/float hybrids at
equal op budget — the open question this domain exists for), gelu/sigmoid
down Joe's list, joint-constant optimizer, arm-A diversity machinery.
Files: engine/remez.py, domains/tanh.py, domains/tanh_shelf.py,
engine/molds_float.py (n_const), engine/registry.py, domains/rsqrt.py
(unclipped shaped loss, sample_shaped), scripts/run_tanh_calibration.py,
scripts/run_rsqrt_hunt.py (metric param), runs/tanh-calibration-*

## 2026-06-12 — the tanh HUNT: honest null — the polynomial frontier holds at ops 10/14, and the null ROUTES the portfolio

The question (predeclared, null declared valid): at op budgets 10 and 14
on [0.25, 8), can any program over our 32-bit op set beat the PROVEN
weighted-Remez polynomial floors? Search: per (budget, seed) two islands
(warm = seeded with the engine's own Lawson-calibrated polynomial; cold =
blank), unclipped shaped loss, memetic constants-descent every 50 gens,
migration every 20, ~580k evals per run, final true-metric polish +
exhaustive verification (runs/tanh-hunt-*/report.json).

HARNESS BUG CAUGHT BETWEEN BATCHES: the first batch finished ABOVE its
own warm seed's error (1.24e-2 vs the seed's 9.96e-3) — the shaped rank
drifts toward the mean-optimum and truncation discards the max-metric
baseline; the final polish could not return (nonsmooth descent). The
calibration had already taught this lesson (warm starts must not pass
through the shaped phase) and I reintroduced it in the hunt loop. Fix:
polish BOTH the search best AND the preserved warm seed, keep the better
— the hunt can no longer lose to its own starting point. First-batch
verdict on f32-gains was therefore void; beyond-polynomial null was
unaffected (no bit-op structure in either batch).

RESULT (fixed batch, 6/6 deterministic): ops<=10: E = 9.9582e-3 (proven
floor 9.9495e-3, shelf f32 9.9555e-3); ops<=14: E = 6.0927e-3 (floor
6.0768e-3, shelf 6.0908e-3). bits=False in every winner — across ~3.5M
total evaluations no bit-op-bearing candidate ever ranked into
contention. Verdict, scoped: no usable signal for beyond-polynomial
structure under this representation / search / budget on this scope. The
engine's best IS the minimax polynomial, as theory demands within the
float-only class; the bit ops bought nothing for tanh.

WHY — and where this routes the portfolio (the null's actual value):
rsqrt's magic trick works because rsqrt has EXPONENT-SCALING structure
(f(2^2k x) = 2^-k f(x)) that integer ops on the exponent field exploit
directly. tanh is a saturating knee on a bounded range — no such
symmetry. The real-world beyond-polynomial tools for the saturating
family are RATIONAL approximations (needs FDIV — currently excluded for
float64 double-rounding; single-rounding FDIV semantics would unlock the
class) and PIECEWISE evaluation (needs compare/select ops — absent from
the op set). Two cleanly-identified representational walls, each with a
concrete capability fix. Routing rule derived: hunt bit tricks on
exponent-structured functions (sqrt, exp2, log2 family); bring
FDIV+SELECT before hunting the saturating family (tanh, sigmoid, gelu,
erf, softplus) beyond polynomials.

Status: WORKS — null verdict, predeclared valid, doubly informative
(frontier confirmed + portfolio routing). Gauntlet extended (new module
sanities + tanh calibration stage); README portfolio section added.
Files: scripts/run_tanh_hunt.py, scripts/run_proof_phase.py, README.md,
runs/tanh-hunt-*

## 2026-06-12 — the exponent-family hunt: BEYOND-POLYNOMIAL WIN certified for log2 (129x / 83x past PROVEN floors); sqrt pareto points; the routing validated

Following the tanh-null routing: sqrt and log2, where exponent-scaling
symmetry lives (theory citation: Blinn 1997, "Floating-Point Tricks",
IEEE CG&A 17(4) — "the bit pattern, interpreted as an integer, gives a
piecewise linear approximation to the logarithm"). New capability, earned:
U2F / F2U value-conversion ops (single-rounding-safe: uint32->f64 exact,
one f32 round; F2U truncation semantics defined identically in both
paths) + AND32/OR32 completing the 32-bit family. Metric switch in the
pack base (err_kind rel/abs): log2 uses max ABSOLUTE error — the
shift-invariant choice, and relative blows up at the in-scope zero
log2(1)=0.

sqrt (scope all f32 in [2^-8,2^8), max rel, exhaustive certs):
- seed-only (2 ops): K = 0x1FBB4F19 found from outcome by the 1-D
  coarse-to-fine sweep; E = 3.4750e-2.
- via-rsqrt (8 ops): x *f rsqrt_A87(x), composing the engine's own rsqrt
  artifact: E = 1.7513e-3 — composition inherits the rsqrt accuracy, as
  theory predicts. Two certified pareto points.

log2 (scope all f32 in [2^-8,2^8), max ABS, exhaustive certs):
- PROVEN polynomial floors (weighted=None Remez at 60 dps, log-spaced
  extrema grid — a linear grid misses everything left of x=1 on a
  16-octave range, fixed in engine/remez.py): deg-1 (2 ops) floor 5.5431;
  deg-5 (10 ops) floor 3.5565. Polynomials are catastrophic at log2 over
  16 octaves, as expected.
- L3 trick (3 ops: U2F, FMUL c0, FADD c1; both constants genes): the
  engine found c0 = 0x34000000 = EXACTLY 2^-23 and c1 ~ -126.957 (the
  optimal bias). Exhaustive E = 4.3043e-2 -> 129x BELOW the proven
  deg-1 floor. Matches the folklore trick's known quality; the structure
  is Blinn-family (cited), the constants are from outcome, the
  certificates are ours. All 3 seeds identical (deterministic).
- L10 (trick + deg-2 mantissa correction, masks as genes): the correction
  NEVER ENGAGED (poly genes stayed zero; masks junk-inert) — the coupled-
  genes optimizer limitation again, OPEN. Still 83x below the deg-5
  floor via the base trick. The L10 structure's extra headroom is real
  and uncaptured: optimizer work item (CMA-style joint moves, or
  Lawson-on-the-residual once masks are engaged).
- PASS 6/6 (predeclared bar: every log2 run >10x below its floor).

Optimizer lessons (toolbox += 2): (a) the zero-output desert exists in
the ABS metric too (flat |truth|-level plateau) — plain and shaped
descent both trapped; (b) when a gene has a CLOSED-FORM optimum given the
others (the additive offset = Chebyshev center -(max+min)/2 of the
residual), solve it analytically and sweep only the rest — this killed
the desert. Also: NaN sort-keys silently corrupt coarse sweeps (guarded).

THE ARC, COMPLETE: tanh (saturating, no exponent symmetry) -> honest
null, polynomials hold; log2 (the symmetry itself) -> certified 129x
beyond-polynomial win. The routing rule is now empirical, both
directions, with certificates. Three sub-runs in runs/explog-hunt-*.

Status: WORKS (PASS 6/6 + 2 sqrt pareto points). Open: L10 correction
engagement; exp2 (needs F2U path exercised) next; FDIV+SELECT for the
saturating family still queued.
Files: engine/core_lang.py + engine/runner.py (U2F/F2U/AND32/OR32),
engine/molds_float.py (OPS_CVT, npfunc, pretty), engine/remez.py
(log-spaced grid), domains/rsqrt.py (err_kind switch),
domains/sqrt_log2.py, engine/registry.py, scripts/run_explog_hunt.py,
scripts/run_proof_phase.py (sanity add), runs/explog-hunt-*

## 2026-06-12 — exp2 (Schraudolph trick): F2U path exercised, certified 31x beyond-polynomial; bias correction REDISCOVERED from outcome

exp2 = log2's mirror (F2U writes the exponent field; canonical citation
Schraudolph 1999, Neural Computation 11(4), "On a Fast, Compact
Approximation of the Exponential Function"). First SIGNED-scope domain:
extended the pack base with a signed flag (mirrors the positive-magnitude
octave scope with its sign-flipped twin via XOR 0x80000000; verification
stays exhaustive over the doubled set; _signs() threaded through
verify/dense/cross-check). Scope: all float32 with |x| in [2^-8, 8), both
signs = EXACTLY 184,549,376 values (11 octaves x 2^23 x 2 signs),
max RELATIVE error vs float64 numpy.exp2, exhaustive.

Arm E3 (3 ops: FMUL c0, FADD c1, F2U): slope c0 = 2^23 GIVEN (forced by
the exponent field — derived, not searched, same status as rsqrt's Newton
coeffs); bias c1 found FROM OUTCOME (coarse-to-fine 1-D sweep + dense
fine-scan over the F2U sawtooth + exhaustive confirmation).

RESULT (runs/exp2-hunt-*/report.json): exhaustive max rel error
2.9827e-2. The slope was structurally FIXED at 0x4B000000 = 2^23 (GIVEN,
per the runspec slope_given field — NOT searched); the engine found the
BIAS from outcome = 0x4E7DE9A3 (= 1064986816.0), an implied correction of
+366400 = 0.0437 x 2^23 vs the zero-correction 127x2^23 — that IS
Schraudolph's published
correction class (~0.043 x 2^23), REDISCOVERED from outcome (his exact
constant not carried from memory; flagged matching-class). PROVEN
polynomial floors (weighted Remez, relative) are catastrophic as the
symmetry predicts — polynomials cannot track exp's relative error across
5 orders of magnitude: deg-1 9.99e-1, deg-3 9.30e-1, deg-5 4.50e-1. So
the 3-op trick has 31x LOWER MAX RELATIVE ERROR than the degree-3
polynomial floor (6-op Horner) and 15x lower than the degree-5 floor
(10-op), under the declared scope/metric. NOT a general "31x better than
polynomials" claim — it is against specific degrees/op-budgets on this
exact signed scope and relative metric. PASS (predeclared: E <
deg3_floor/2).

Cost note: the hunt took 6 min (dense fine-scan window x signed scope is
heavy); it is an on-demand experiment, NOT in the gauntlet (only its
module sanity is). Optimization of the sweep is a minor todo.

THE EXPONENT FAMILY, COMPLETE (3 certified members + the tanh contrast):
log2 129x, exp2 31x, sqrt 2 pareto points (incl. composing the engine's
own rsqrt artifact); tanh = honest null. The symmetry-based routing rule
("bit tricks win where exponent-scaling structure exists, polynomials
hold where it doesn't") is now empirical in BOTH directions with
exhaustive certificates on four functions.

[CORRECTION 2026-06-13, Joe caught it — claim hygiene]:
- SLOPE LANGUAGE: the exp2 entry above (and the pushed commit 504d3e0 msg,
  README, memory) originally said "the engine FOUND slope = 2^23". WRONG —
  the slope was structurally fixed/given (C0=fb(2^23) in the script, only
  c1 swept). The script's runspec logged slope_given correctly; the prose
  leaked a framing (exactly the parent-audit failure mode). Correct
  statement: "slope structurally fixed at 2^23; the BIAS was found from
  outcome." (Contrast: log2 DID search its slope and found 2^-23 — that
  claim stands.) Fixed in tracker/README/memory + the script's recognizer
  label.
- "31x BEYOND-POLYNOMIAL": tighten to "31x lower max relative error than
  the deg-3 polynomial floor under the declared scope/metric." Not "31x
  better than polynomials."
- SCOPE COUNT: exactly 184,549,376 (the "84M" was a confused leftover).
- FLOOR-DOMAIN VALIDITY (verified): the deg-k floors are real-arithmetic
  minimax over [-8,8] with RELATIVE weight; the trick is float32 over the
  signed octave scope. Both differences (real vs float32 eval; [-8,8]
  slightly wider than the octave scope) are CONSERVATIVE-for-the-polynomial
  (a real-arith floor <= any float32 poly impl; a wider interval only
  raises the floor), so 31x is a LOWER bound on the gap. Metric matches
  (both relative). The re-run (faster multi-resolution scan, 48s vs 6min)
  reproduced the result deterministically: bias 0x4E7DE9A3, E=2.9827e-2,
  31x — clean artifact in the latest runs/exp2-hunt-*.

Status: WORKS (PASS, predeclared; claim language corrected per above).
Next: FDIV+SELECT capability for the
saturating family (tanh/sigmoid/gelu/erf rationals + piecewise);
coupled-genes optimizer (rsqrt arm-B + log2 L10 uncaptured headroom);
gelu/sigmoid down Joe's list once FDIV lands.

## 2026-06-13 — FDIV capability + sigmoid: rational [2/2] CERTIFIED 2.9x below the proven deg-4 floor; [3/3] = exhaustive caught a false 15x "win"

Opened the saturating family (the tanh-null routing's other branch) with
FDIV. FDIV semantics (defined, stdlib-core-safe, bit-identical both
paths): the float64 quotient rounded once to float32 — NOT IEEE-correctly-
rounded f32 division (that needs a non-stdlib divide in the core runner),
but a single deterministic well-defined function (runner: Python f64
divide -> round f32; npfunc: f64 divide -> cast f32; x/0=+-inf, 0/0=nan).
Opt-in per mold (OPS_DIV), so rsqrt/log2/exp2 op sets are unchanged.
SigmoidPack (signed scope |x| in [2^-4, 8), ~150M values, max ABSOLUTE
error vs float64 1/(1+e^-x), exhaustive). engine/ratfit.py: the coupled-
coefficient optimizer the saturating family needs — linearized rational
least-squares + Lawson IRLS, coeffs FROM OUTCOME (deterministic, no RNG ->
no seed variance), then float32-rounded and EXHAUSTIVELY verified.

Proven polynomial floors (abs Remez over [-8,8], conservative-for-poly:
real-arith, slightly wider interval): deg-2 2.06e-1, deg-4 8.95e-2,
deg-6 4.09e-2.

RESULTS (runs/sigmoid-hunt-*/report.json):
- [1/1] (5 ops): exhaustive 2.07e-1 ~ deg-2 floor 2.06e-1 -> NULL (a
  [1/1] is essentially affine here; honest no-win).
- [2/2] (9 ops): exhaustive max abs 3.08e-2 -> 2.9x LOWER max abs error
  than the proven deg-4 (8-op) floor, AND below the deg-6 (12-op) floor
  4.09e-2. So a 9-op rational beats the proven floor of the 12-op
  polynomial class. CERTIFIED (exhaustive, both paths). The tanh-null
  routing ("saturating family needs rationals") CONFIRMED for sigmoid.
- [3/3] (13 ops): the real-coefficient fit is pole-free (min|Q|=1.0 over
  a dense scope, no sign change) and 15x below the deg-6 floor (2.65e-3
  on the sample) — BUT its coefficients span ~15 orders of magnitude (a
  denominator root at -6.1e14 => b3 ~ 1.6e-15 vs b1 ~ O(1)), which
  float32 CANNOT hold; rounding the coefficients produces an exhaustive
  max abs error of 8.0. EXHAUSTIVE VERIFICATION CAUGHT WHAT A SAMPLED
  BENCHMARK WOULD HAVE REPORTED AS A CLEAN 15x WIN. This is the project's
  ruthless-verification thesis in one data point.

Predeclared PASS bar was "[2/2] AND [3/3] beat their floors" -> FALSE (I
do NOT move the goalpost). But the meaningful result stands: [2/2] is a
clean certified rational win confirming the routing, and [3/3] is a
high-value verification catch. The fix for [3/3] is float32-AWARE
coefficient fitting (fpminimax-style: search coefficients in float32 /
control dynamic range), logged as next — naive real-fit + round is not
enough for higher-order rationals.

Status: WORKS ([2/2] certified; [3/3] honestly rejected by exhaustive).
Next: fpminimax-style f32-aware rational fit (rescue [3/3], unlock higher
orders); tanh/gelu/erf via the same FDIV machinery; SELECT/piecewise as
the second saturating tool.
Files: engine/core_lang.py + engine/runner.py (FDIV), engine/molds_float.py
(OPS_DIV, npfunc/pretty/cost), engine/ratfit.py, domains/sigmoid.py,
engine/registry.py, scripts/run_sigmoid_hunt.py,
scripts/run_proof_phase.py (sanities), runs/sigmoid-hunt-*
Files: domains/rsqrt.py (signed scope), domains/exp2.py,
engine/registry.py, scripts/run_exp2_hunt.py,
scripts/run_proof_phase.py (sanity), runs/exp2-hunt-*
