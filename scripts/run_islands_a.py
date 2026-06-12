"""Step-3 experiment: island ecology on the n=6 and n=8 headroom gaps.

Hill-climb (step 2) left n=6 at size 13 (proven optimal 12) and n=8 at 22
(proven optimal 19). Question: does the island ecology close those gaps —
and does it do so OUTCOME-ONLY (blank + pressure islands, no shelf
contamination), or only via the seeded island (improvement, not discovery)?

3 seeds per n (comparative claims need >= 3 seeds, RULES.md).

Run from repo root:  python3 -m scripts.run_islands_a
"""

import json
import sys
import time

from engine.islands import IslandsSpec, run_islands
from domains.sorting_networks_shelf import BOUNDS


def main():
    t0 = time.time()
    rows = []
    all_have_outcome = True
    for n in (6, 8):
        for seed in (0, 1, 2):
            spec = IslandsSpec(
                domain="sorting_networks",
                domain_params={"n": n},
                roles=("blank", "seeded", "pressure"),
                pop_size=64 if n == 6 else 96,
                generations=600 if n == 6 else 1200,
                migrate_every=25,
                init_length=15 if n == 6 else 28,
                seed=seed,
                target_outcome_size=BOUNDS[n]["size"],
            )
            r = run_islands(spec)
            oo = r.get("best_outcome-only")
            sd = r.get("best_seeded")
            all_have_outcome &= oo is not None
            rows.append({"n": n, "seed": seed, "run_id": r["run_id"],
                         "evals": r["evals_total"], "seconds": r["seconds"],
                         "stop": r["stop_reason"],
                         "outcome_only": oo and oo["cost"],
                         "oo_island": oo and oo["island"],
                         "seeded": sd and sd["cost"]})
            print(f"n={n} s{seed}: outcome-only="
                  f"{oo['cost'] if oo else None} ({oo['island'] if oo else '-'}) "
                  f"seeded={sd['cost'] if sd else None} "
                  f"evals={r['evals_total']} t={r['seconds']}s")
            print(f"        stop: {r['stop_reason']}")

    print()
    for n in (6, 8):
        b = BOUNDS[n]
        oo_sizes = [r["outcome_only"]["comparators"] for r in rows
                    if r["n"] == n and r["outcome_only"]]
        sd_sizes = [r["seeded"]["comparators"] for r in rows
                    if r["n"] == n and r["seeded"]]
        print(f"n={n}: proven optimal size {b['size']} ({b['size_source']}) | "
              f"outcome-only sizes {oo_sizes} | seeded sizes {sd_sizes}")
    out = {"seconds": round(time.time() - t0, 2), "rows": rows}
    path = f"runs/islands_summary-{int(t0)}.json"
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"summary -> {path}")
    return 0 if all_have_outcome else 1


if __name__ == "__main__":
    sys.exit(main())
