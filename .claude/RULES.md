# Operating Rules — mathlab

These are the conventions you follow throughout this project. They are
deliberately short. Re-read them at the start of each session.

## The tracker is sacred

`TRACKER.md` is the project's memory. You will run dozens or hundreds of
experiments over many sessions. Without the tracker you will repeat failed
experiments, lose track of partial findings, and rediscover the same dead
ends. With it, you accumulate real knowledge.

After every experiment — successful, failed, partial, abandoned — append an
entry to TRACKER.md. Format:

```
## [date] — [short name]
What I tried: [one paragraph]
What happened: [one paragraph, with concrete numbers]
What I learned: [one or two sentences]
Status: [WORKS / FAILED / ABANDONED / PARTIAL]
Files: [paths to the code, if kept]
```

If you find yourself wanting to skip the tracker entry, you've already
broken the rule. Write the entry first, then move on.

Before starting any new experiment, search TRACKER.md for whether you (or a
previous session's you) already tried it. If yes, don't repeat it unless
you have a specific new reason.

## Abandonment is a first-class action

When an approach isn't producing signal, abandoning it is the correct move.
Mark it ABANDONED in the tracker with a one-sentence reason and move on.
This is not failure. This is the explore-many strategy working as intended.

Heuristic: if you've tuned hyperparameters or made small changes to the same
core approach more than three times without improvement, abandon it. Try
something structurally different.

## Be opinionated and weird

You are not writing a textbook implementation. You are inventing approaches
that exploit the specific constraints of this project: small scale, narrow
domain, exact verification, from-scratch architecture, no need to compare
to baselines.

When designing an experiment, ask: "what's the published-paper answer to
this?" Then ask: "what's a weirder approach that might fit this setup
better?" Try the weird one at least as often as the conventional one.

Examples of weird that's worth trying:
- Architectures with shapes nobody uses (very tall and narrow, very wide
  and shallow, asymmetric)
- Custom executors / DSLs / discrete representations
- Discovery via mechanisms other than RL (evolution, search, neuro-symbolic)
- Mixing supervised + outcome-based + self-distillation in unusual ways
- Letting the model emit its outputs in formats nobody uses (continuous
  vectors that decode to programs, programs as graphs, programs as
  hierarchical compositions)

Examples of what to resist defaulting to:
- "small Transformer trained with REINFORCE on token sequences" without
  asking if a different output representation might work better
- Standard RL recipes (PPO with default hyperparameters)
- Mimicking the exact structure of published work like AlphaCode, AlphaTensor

If you must use a conventional approach, do so as a baseline to compare
weird approaches against, not as the main attempt.

## Honesty about uncertainty

When you write in the tracker or explain to me:

- If something worked, say it worked. Don't oversell.
- If something failed, say it failed. Don't dress it up as "partial validation."
- If you don't know whether something will work, say so explicitly before
  trying it. Then try it. Don't predict elaborately.
- If a result is ambiguous, say it's ambiguous and what would resolve it.

The previous version of this project failed because elaborate narratives got
built on top of unverified results. Don't repeat that.

## Code discipline

- Notes-style code is fine. Don't over-engineer.
- No formal tests required. Sanity-check assertions in `__main__` blocks
  are encouraged.
- If a file exceeds ~300 lines, consider splitting.
- Use clear file names that say what's inside.
- Don't write polish/documentation passes unless I ask. Time spent on
  formatting is time not spent on experiments.

## Running experiments

- Default to small / fast first. If something looks promising at scale 1,
  scale up. Never start at scale 10.
- Use the GPU when training real models. CPU for quick sanity checks.
- For long runs (>5 minutes), think about whether the experiment is worth
  that wait. Often a smaller version answers the question.
- Save checkpoints for anything that worked. Throw away checkpoints for
  things that didn't.

## Subagents and parallel exploration

You are free to spawn subagents to explore directions in parallel. When you
do, give each one:
- A clear specific question to answer (not "explore approach X")
- A budget (time or steps)
- An instruction to write its result back to TRACKER.md

Don't spawn so many that you can't keep track. Three is probably the upper
useful limit at any one time.

## The math constraint

Eval is exact. `2 + 3 == 5`, not `5.0001`. No partial credit for arithmetic.

If you find yourself wanting to relax this constraint, you're trying to
disguise an approach that's not actually working. Don't.

## When in doubt

Read the original `PROMPT.md` again. The goal is discovery of math
procedures by a small model. Not impressive-looking metrics. Not coverage
of every approach in the literature. Just: did the model find an algorithm
we can extract and verify.

## Amendments — 2026-06-09 (post-audit, approved by Joe)

Added after the Fable 5 takeover audit (`consolidation/09_fable5_audit.md`).
Each of these was violated at least once in sessions 1–10, and each violation
cost something. The audit's meta-lesson: the project's honesty held at the
NUMBER level but leaked at the READING level — a live plain-read of a log tail
became a tracker sentence, survived consolidation, and got upgraded into a
framing. These rules target that seam.

- **Claims need artifacts.** Any number written into TRACKER.md or a
  consolidation doc must exist in a file under `runs/` (log, json, table).
  A number read off live stdout that is not persisted gets an explicit
  `[STDOUT-ONLY]` tag at write time — and is not citable until re-run.
  (Violations: avida_oe's a+b waypoints; expV's numbers.)

- **Comparative claims need seeds.** "X beats Y" / "signal A resolves B"
  requires ≥3 seeds, or an explicit `n=1` flag in the Status line.
  Exact-verification *correctness* claims are exempt — that is the point of
  exact. (Violation: the QD comparison — n=1 plus a bug.)

- **Render before verdict.** Any structured/class-4/clean/settled judgment
  about a dynamical object requires a rendered artifact saved next to the
  metric. "Looks complex by a metric" ≠ "is dynamically complex" — learned
  twice, still violated once (exp1 s2/s3 at consolidation).

- **Stratify mixed-condition losses.** Never summarize a mixed-width /
  mixed-task loss series by its tail; split by the sampled condition before
  claiming fit. (Violation: exp3-mul "fits in-distribution".)

- **No duplicate-index scatter on CUDA** for archives/elites — write order is
  undefined and fitness↔genome pairing can corrupt. Use sort + segment-reduce
  or `scatter_reduce`. And **re-execute any archived winner before reporting
  it** — one re-run would have caught the exp2 corruption immediately.

- **Version control.** The repo is under git as of 2026-06-09; commit at least
  once per session (code + docs + tracker; large binaries per `.gitignore`).
  The tracker stays append-only in spirit: factual errors in old entries get
  dated `[ERRATUM ...]` brackets pointing to the correction — never silent
  rewrites.

- **State the scope when upgrading.** When an observation is promoted to a
  wall / theorem / law, the entry must state the quantifier it holds under
  (the I/O encoding, the learner class, the budget/cap, the seed count).
  All three audit scope-holes (#3 encoding, #5 learner class, #6 closure)
  entered at upgrade moments.