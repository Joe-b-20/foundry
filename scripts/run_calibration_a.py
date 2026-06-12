"""Calibration A, v0 — the walking-skeleton check, end to end.

For n = 3..6: hill-climb sorting networks from scratch through the full
pipeline (mold -> proposer -> fast judge -> core-runner verify-on-write ->
recorder -> report). The goal of THIS script is rung 0 (the loop runs end to
end) plus first blood on rung 1 (correct networks found and L1-verified).
Naming found networks against the reference shelf (full rung 1) is build
step 2.

Run from repo root:  python3 -m scripts.run_calibration_a
"""

import sys

from engine.foreman import RunSpec, run


def main():
    overall_ok = True
    summaries = []
    for n in (3, 4, 5, 6):
        spec = RunSpec(
            domain="sorting_networks",
            domain_params={"n": n},
            proposer="hill-climb",
            proposer_params={"patience": 60},
            budget_candidates=6_000 if n <= 4 else 80_000,
            batch=32,
            init_length=max(3, n * (n - 1) // 2),
            seed=0,
            settle_batches=150,
        )
        report = run(spec)
        ok = bool(report.get("found_correct") and report.get("verified_canonical"))
        overall_ok &= ok
        b = report.get("best", {})
        summaries.append((n, ok, b.get("cost"), report["candidates_used"],
                          report["seconds"], b.get("pretty"), report["run_id"]))
        print(f"n={n}: {'OK' if ok else 'FAILED'} cost={b.get('cost')} "
              f"used={report['candidates_used']} t={report['seconds']}s")
        print(f"      {b.get('pretty')}")

    print()
    for n, ok, cost, used, secs, pretty, rid in summaries:
        if cost:
            bubble = n * (n - 1) // 2  # size of the naive bubble network
            print(f"n={n}: size {cost['comparators']} depth {cost['depth']} "
                  f"(bubble-network size {bubble}) run={rid}")
    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
