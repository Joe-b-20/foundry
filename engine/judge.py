"""Judge v0: gate 1 (correct on the pack's checker), cost ordering, and
verify-on-write (canonical re-execution through the core runner).

The full gates 2/3 (shelf comparison, breaker, recognizer) are build steps
2-4. v0 keeps their seams visible: the pack's fast checker is the
search-time path, and nothing is reported without the canonical core-runner
re-check — the fast path never gets the final word.
"""

from engine import runner as core_runner


def score(pack, mold, cand):
    """Returns (tidied_candidate, score_tuple, native_cost).

    score_tuple = (correct, n_sorted, -size, -depth); bigger is better.
    """
    tidy = mold.tidy(cand)
    n_ok, total = pack.fast_score(tidy)
    cost = mold.native_cost(tidy)
    sc = (int(n_ok == total), n_ok, -cost["comparators"], -cost["depth"])
    return tidy, sc, cost


def verify_canonical(pack, mold, cand):
    """Re-run the poured program through the core runner on the pack's full
    canonical input set. Returns (ok, details)."""
    program = mold.pour(cand)
    checked, last_cost = 0, None
    for inputs in pack.all_inputs():
        out, last_cost = core_runner.run(program, inputs)
        if not pack.is_correct(inputs, out):
            return False, {"checked": checked,
                           "failed_on": list(inputs), "got": list(out)}
        checked += 1
    return True, {"checked": checked,
                  "core_cost": last_cost.as_dict() if last_cost else None}
