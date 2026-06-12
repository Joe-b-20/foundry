# foundry

Algorithm Foundry is a domain-pluggable search engine that proposes
candidate algorithms, executes them in a cost-metered language, verifies
correctness, attacks its own survivors, compares against known references,
diagnoses search walls, and records every action as training data for
future automation.

The goal: find better, faster, more efficient algorithms — in any and all
domains. Or, at minimum, be the project that honestly tries.

- [PROMPT.md](PROMPT.md) — the mission, the machine, the win ladder
- [RULES.md](RULES.md) — operating rules (claims need artifacts; claim
  wording never exceeds its certificate; predeclare budgets and cost rules)
- [TRACKER.md](TRACKER.md) — the experiment log, including the failures.
  Read this to see what is actually real.
- `reference/mathlab/` — the frozen parent project (public as
  [rediscovery-engine](https://github.com/Joe-b-20/rediscovery-engine.)),
  whose wall taxonomy and honesty rules this project inherits.

Status: three domains live through one generic engine. Sorting networks:
proven-optimal sizes found from outcome at n=3..8, optimality re-certified
by exhaustion at n<=4. Polynomial continued fractions: the parent's
flagship results replicated — seeded walks reach published family members
(3/3 seeds, controls perfect) and RM 8/(7 zeta3) rediscovered from outcome
in a predeclared sweep with an in-grid Apery control and an empty null arm.
Bilinear decompositions: Karatsuba rediscovered from outcome and NAMED
(exact canonical match, Karatsuba & Ofman 1962) in 3/3 seeds, R=3 proven
optimal by a flattening bound computed exactly in-repo. Bit-mixers: the
wall doctor passed its C1/C2 exam 6/6 — finds planted programs (sometimes
shorter than the plant, verified exhaustively) and correctly recommends
abandoning a keyed 8-round mixer, judging generalization on held-out data
rather than corpus fit. The calibration ladder (floor / middle / roof) is
complete. Full event logs under `runs/`. Next: the portfolio opens —
new domains chosen for headroom and neglect.
