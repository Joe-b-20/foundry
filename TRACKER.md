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
