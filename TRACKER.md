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
