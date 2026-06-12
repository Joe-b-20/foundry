"""Domain pack: bilinear decompositions (calibration item Karatsuba; the
same pack takes Strassen's matmul tensor later — the target is data).

Problem: compute a target bilinear map with as few MULTIPLICATIONS as
possible. Verification is exact and complete: the candidate's induced
tensor must equal the target tensor entry-for-entry (integers). By
bilinearity, tensor identity implies correctness for ALL inputs — so the
certificate is stronger than any test set: "L1-exact-tensor-identity".
The trust path additionally pours to core and cross-checks the runner on
random integer inputs (consistency of the compiled program).

Cost rules (predeclared): primary = number of multiplications R;
secondary = description length (sum of |entries|).
"""

TARGETS = {
    # degree-1 polynomial multiplication (2 coeffs x 2 coeffs -> 3 coeffs):
    # T[i][j][k] = 1 if i+j == k else 0, flattened in (i, j, k) order
    "polymul2": (1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1),
}


class BilinearPack:
    name = "bilinear"

    def __init__(self, target="polymul2"):
        self.target_name = target
        self.target = TARGETS[target]
        self.cost_rules = ("primary: multiplications (R); "
                           "secondary: description length")

    # --- judge contract ---------------------------------------------------
    def gate1(self, mold, tidy):
        t = mold.tensor(tidy)
        l1 = sum(abs(a - b) for a, b in zip(t, self.target))
        cost = mold.native_cost(tidy)
        exact = int(l1 == 0 and len(tidy) > 0)
        return ((exact, -l1, -cost["mults"], -cost["dl"]), cost)

    def verify_trusted(self, mold, cand):
        import random
        from engine import runner as core_runner
        tidy = mold.tidy(cand)
        t = mold.tensor(tidy)
        if t != self.target:
            return False, {"reason": "tensor mismatch",
                           "l1": sum(abs(a - b)
                                     for a, b in zip(t, self.target))}
        prog = mold.pour(tidy)
        rng = random.Random(12345)
        for _ in range(50):
            a0, a1, b0, b1 = (rng.randint(-999, 999) for _ in range(4))
            out, cost = core_runner.run(prog, [a0, a1, b0, b1])
            if out[:3] != [a0 * b0, a0 * b1 + a1 * b0, a1 * b1]:
                return False, {"reason": "core-runner cross-check FAILED "
                                         "(pour bug?)",
                               "inputs": [a0, a1, b0, b1], "got": out[:3]}
        return True, {"certificate": {
            "level": "L1-exact-tensor-identity",
            "claim": f"computes {self.target_name} exactly for ALL inputs",
            "evidence": "induced tensor equals target entry-for-entry "
                        "(integer identity; bilinearity extends to all "
                        "inputs) + 50 random-input core-runner cross-checks"},
            "core_mults": cost.native.get("mult")}


if __name__ == "__main__":
    from engine.molds_bilinear import BilinearMold
    mold, pack = BilinearMold(), BilinearPack()
    NAIVE = (((1, 0), (1, 0), (1, 0, 0)), ((1, 0), (0, 1), (0, 1, 0)),
             ((0, 1), (1, 0), (0, 1, 0)), ((0, 1), (0, 1), (0, 0, 1)))
    KARATSUBA = (((1, 0), (1, 0), (1, -1, 0)),
                 ((1, 1), (1, 1), (0, 1, 0)),
                 ((0, 1), (0, 1), (0, -1, 1)))
    for cand, r in ((NAIVE, 4), (KARATSUBA, 3)):
        tidy = mold.tidy(cand)
        sc, cost = pack.gate1(mold, tidy)
        assert sc[0] == 1 and cost["mults"] == r, (sc, cost)
        ok, det = pack.verify_trusted(mold, tidy)
        assert ok and det["core_mults"] == r, det
    # a broken variant must fail the tensor identity
    BAD = KARATSUBA[:2]
    sc, _ = pack.gate1(mold, mold.tidy(BAD))
    assert sc[0] == 0
    print("bilinear pack ok: naive R=4 and karatsuba R=3 verified exactly")
