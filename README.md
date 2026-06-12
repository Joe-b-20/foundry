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

Status: engine skeleton + first domain (sorting networks) calibrated —
proven-optimal-size networks found from outcome at n=3..8, optimality
independently re-certified by exhaustion at n=2..4, full event logs under
`runs/`. Next: the polynomial-continued-fraction domain pack (replicating
the parent's strongest results through the generic engine), then the
decomposition mold (Karatsuba), then the wall doctor's C1/C2 exam.
