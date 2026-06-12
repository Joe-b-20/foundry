# Operating rules — foundry

Deliberately short. Re-read at the start of each session. These carry the
parent project's rules and its post-audit amendments, plus foundry-specific
rules. When in doubt, read PROMPT.md again.

## The tracker is sacred

After every experiment — works / failed / partial / abandoned — append an
entry to TRACKER.md: what I tried, what happened (concrete numbers), what I
learned, status, files. Before starting anything, search TRACKER.md for
whether it was already tried. The flight recorder is for machines; the
tracker is for humans. Both are kept.

## Honesty (parent rules + audit amendments, all still binding)

- Claims need artifacts. Any number written into TRACKER.md must exist in a
  file under runs/. A stdout-only number gets an explicit [STDOUT-ONLY] tag
  and is not citable until re-run.
- Comparative claims need >= 3 seeds, or an explicit n=1 flag in the status
  line. Exact-correctness claims are exempt — that is the point of exact.
- Artifact/render before verdict. Stratify mixed-condition results before
  claiming fit.
- State the scope when upgrading an observation to a wall / law / theorem:
  the encoding, the learner class, the budget, the seed count.
- Re-execute any archived winner before reporting it (verify-on-write).
- Append-only spirit: factual errors in old entries get dated [ERRATUM]
  brackets pointing to the correction — never silent rewrites.
- Commit at least once per session.

## Foundry-specific rules

- Predeclare or it didn't count: cost rules, budgets, and stop rules live in
  the RunSpec and are logged before the search starts. No goalpost moves
  after the fact.
- Claim wording may never be stronger than its certificate level:
  L0 survived testing + attack budget · L1 exhaustive within stated bounds ·
  L2 external re-checkable proof artifact · L3 formal proof.
- Nothing enters the archive without at least a light breaker pass. No
  exceptions for boring.
- Counterexamples join the domain's corpus permanently (the ratchet).
- The recognizer's NEW label is only valid after a shelf + literature check.
  UNRESOLVED is a respectable verdict. The same labels apply to our own
  ideas (standard / new wiring / rare) — verify "rare" before saying it in
  public.
- The wall doctor recommends; it does not kill (v1). Verdict format, always
  scoped: "no usable signal under the current representation / search
  primitives / budget."
- Every operator decision and every creativity injection is logged with a
  reason. The replay test: every decision must be reconstructible from the
  recorder alone.
- Proposers see only outcomes — scores and counterexamples — never checker
  internals or shelf solutions. Discovered-from-outcome stays separable from
  scaffolded, always.
- A domain is not live until rung 1 (calibration rediscovery) passes under
  the declared verifier.
- Don't replicate big-lab searches with less compute. Their methods are
  priors; their published results are the swept-frontier map and belong in
  shelf and bounds tables, with citations.
- Reference-shelf and bounds-table numbers are never written from memory —
  only with citations checked at shelf-build time.

## Code discipline

Notes-style code is fine; don't over-engineer. Sanity asserts in __main__
blocks. Split files past ~300 lines. Clear names. No polish passes unasked.
The core engine is stdlib-only Python; rented oracles live behind thin owned
wrappers.

## Running experiments

Small and fast first — never start at scale 10. GPU for real training, CPU
for sanity checks. Think before any run longer than 5 minutes. Abandonment
is a first-class action: three tuning rounds on the same core approach
without improvement means try something structurally different and log it.
