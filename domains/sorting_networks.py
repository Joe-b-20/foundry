"""Domain pack A — sorting networks (the calibration floor).

Problem: a comparator network on n wires that sorts every input.

Verification is exact AND complete here: the 0/1 principle (Knuth, TAOCP
vol. 3, sec. 5.3.4) says a comparator network sorts all inputs if and only
if it sorts all 2^n binary inputs. The fast checker tests all 2^n binary
vectors at once with bitmasks (one Python int per wire, bit t = that wire's
value on test t; a comparator is then just AND/OR). The canonical trust path
re-runs winners instruction-by-instruction through the core runner on every
binary vector plus random integer vectors (verify-on-write).

Cost rules (predeclared): primary = comparator count (size),
secondary = depth (parallel layers).

Reference shelf + known-bounds table are build step 2 and get filled WITH
CITATIONS — published optimality numbers are never written from memory
(RULES.md).
"""

import random


class SortingNetworkPack:
    name = "sorting_networks"

    def __init__(self, n: int, extra_random_checks: int = 25, seed: int = 1234):
        assert 2 <= n <= 16, "fast checker builds 2^n-bit masks; keep n small"
        self.n = n
        self.total = 1 << n
        self.mask = (1 << self.total) - 1
        # column i = bitmask over all tests t of wire i's starting value,
        # where test t's input vector is the binary digits of t
        self._init_cols = []
        for i in range(n):
            col = 0
            for t in range(self.total):
                if (t >> i) & 1:
                    col |= 1 << t
            self._init_cols.append(col)
        rng = random.Random(seed)
        self._extra = [[rng.randrange(-1000, 1000) for _ in range(n)]
                       for _ in range(extra_random_checks)]

    # --- fast search-time checker: all 2^n binary tests in one pass --------
    def fast_score(self, cand):
        """Returns (number of binary tests sorted, total binary tests)."""
        cols = list(self._init_cols)
        for (i, j) in cand:
            lo = cols[i] & cols[j]
            hi = cols[i] | cols[j]
            cols[i], cols[j] = lo, hi
        viol = 0
        for i in range(self.n - 1):
            viol |= cols[i] & ~cols[i + 1]
        viol &= self.mask
        return self.total - bin(viol).count("1"), self.total

    # --- canonical trust path (judge.verify_canonical walks this) ----------
    def all_inputs(self):
        for t in range(self.total):
            yield [(t >> i) & 1 for i in range(self.n)]
        for v in self._extra:
            yield list(v)

    def is_correct(self, inputs, out):
        return out == sorted(inputs)

    cost_rules = {"primary": "comparators", "secondary": "depth"}

    bounds_note = ("published optimal sizes/depths for small n exist; the "
                   "bounds table is filled during shelf-building (step 2) "
                   "with citations, never from memory")

    # --- judge contract (engine/registry.py) ----------------------------
    def gate1(self, mold, tidy):
        n_ok, total = self.fast_score(tidy)
        cost = mold.native_cost(tidy)
        return ((int(n_ok == total), n_ok,
                 -cost["comparators"], -cost["depth"]), cost)

    def verify_trusted(self, mold, cand):
        """Re-run the poured program through the core runner on the full
        canonical input set (all 2^n binary vectors + random integers)."""
        from engine import runner as core_runner
        program = mold.pour(cand)
        checked, last_cost = 0, None
        for inputs in self.all_inputs():
            out, last_cost = core_runner.run(program, inputs)
            if not self.is_correct(inputs, out):
                return False, {"checked": checked,
                               "failed_on": list(inputs), "got": list(out)}
            checked += 1
        return True, {"checked": checked,
                      "core_cost": last_cost.as_dict() if last_cost else None}


if __name__ == "__main__":
    pack = SortingNetworkPack(4)
    # classic 5-comparator network for n=4: TESTED here, not trusted
    good = ((0, 1), (2, 3), (0, 2), (1, 3), (1, 2))
    assert pack.fast_score(good) == (16, 16)
    bad = ((0, 1), (2, 3))
    n_ok, _ = pack.fast_score(bad)
    assert n_ok < 16
    # identity (no comparators) sorts only the already-sorted binary inputs:
    # for n=4 the non-decreasing vectors are 0000,0001,0011,0111,1111 -> 5
    n_id, _ = pack.fast_score(())
    assert n_id == 5, n_id
    # cross-check the fast path against the canonical core-runner path
    from engine import judge
    from engine.molds import ComparatorMold
    ok, details = judge.verify_canonical(pack, ComparatorMold(4), good)
    assert ok, details
    assert details["checked"] == 16 + 25
    print("sorting_networks pack ok | canonical checks:", details["checked"])
