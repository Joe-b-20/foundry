"""The exponent-family hunt: sqrt and log2 — where the tanh-null routing
said the bit tricks live. The log2 arm is the mirror experiment to the
tanh null: there, polynomials held at equal ops; here, the prediction
(predeclared) is that the Blinn-family integer-aliasing trick BEATS the
proven polynomial floors by an order of magnitude or more at equal ops.

sqrt arms (scope: all float32 in [2^-8, 2^8), max RELATIVE error):
  S1 seed-only (2 ops): out = K +i (x >> 1) — the magic constant K found
     from outcome by coarse-to-fine 1-D sweep + exhaustive fine scan.
  S2 via-rsqrt (8 ops): x *f rsqrtA87(x) — composes the engine's OWN
     rsqrt artifact (constant 0x5F375A87 + derived Newton). Measured, not
     searched.

log2 arms (scope: all float32 in [2^-8, 2^8), max ABSOLUTE error —
shift-invariant; relative blows up at log2(1)=0):
  L3 (3 ops): out = U2F(x_bits) *f c0 +f c1 — both constants genes, from
     outcome (two-phase bit-descent, 12 restarts).
  L10 (10 ops): L3 plus a degree-2 mantissa correction whose MASK
     CONSTANTS are genes too (the search may rediscover 0x007FFFFF /
     0x3F800000 from outcome). 7 genes, same optimizer, 10 restarts.
Floors: weighted=None (absolute) Remez brackets for the comparable
polynomial budgets — deg-1 (2 ops) vs L3, deg-5 (10 ops) vs L10 — proven
for the exact-arithmetic polynomial class on the interval, log-spaced
extrema grid.

Verdicts (predeclared): WIN = exhaustive E < floor/10 at equal-or-fewer
ops; honest scope note: the floor covers POLYNOMIALS only (rational and
piecewise classes are not floored here), and the trick STRUCTURE is
folklore (Blinn 1997) — what is ours is the from-outcome constants, the
exhaustive certificates, and the measured pareto points.

Run from repo root:  python3 -m scripts.run_explog_hunt
"""

import json
import random
import struct
import sys
import time
from pathlib import Path

import mpmath as mp

from domains.rsqrt_shelf import trick_newton
from engine import registry
from engine.recorder import Recorder
from engine.remez import remez
from scripts.run_rsqrt_hunt import ConstOnlyMold, bit_descent

SEEDS = (0, 1, 2)


def fb(v):
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def sweep_1d(pack, mold, skeleton, base_after, gene_index=0):
    """Coarse-to-fine global sweep of one 32-bit gene (the B1 recipe),
    finished with an exhaustive +-8 fine scan."""
    def cand_of(c):
        consts = list(base_after)
        consts[gene_index] = c & 0xFFFFFFFF
        return (skeleton, tuple(consts))

    def E_of(c):
        return pack.sample_max_rel(mold, cand_of(c))
    coarse = sorted((E_of(k << 19), k << 19) for k in range(1 << 13))[:4]
    best_c, best_e = None, float("inf")
    for _, center in coarse:
        c, w = center, 1 << 19
        while w >= 8:
            grid = list(range(max(0, c - w), min(1 << 32, c + w),
                              max(1, w // 16)))
            c = min(grid, key=E_of)
            w //= 8
        for f in range(max(0, c - 64), c + 65):
            e = E_of(f)
            if e < best_e:
                best_c, best_e = f, e
    exh = {}
    for c in range(best_c - 8, best_c + 9):
        ok, det = pack.verify_trusted(mold, cand_of(c))
        if ok:
            exh[c] = det["max_rel_err"]
    best_c = min(exh, key=exh.get)
    return cand_of(best_c), exh[best_c]


def descend(pack, cmold, rng, n_genes, restarts, passes=10):
    best, best_e = None, float("inf")
    for _ in range(restarts):
        start = cmold.random_candidate(rng)
        cand, _e, _n = bit_descent(pack, cmold, start, passes,
                                   const_indices=range(n_genes),
                                   metric=pack.sample_shaped)
        cand, e, _n = bit_descent(pack, cmold, cand, 6,
                                  const_indices=range(n_genes))
        if e < best_e:
            best, best_e = cand, e
    return best, best_e


def main():
    t0 = time.time()
    run_id = f"explog-hunt-{int(t0)}"
    rec = Recorder(Path("runs") / run_id, run_id)
    rows = []

    # ---------------- sqrt -------------------------------------------------
    pack, mold = registry.build("sqrt", {"n_const": 4})
    T = 1 + mold.N_CONST
    s1_skel = (("SHR32", T, 0, 2), ("ADD32", 0, 1, T))
    cand, E = sweep_1d(pack, mold, s1_skel, (0, 1, 0, 0), gene_index=0)
    rec.event("judge", "sqrt_seed_only",
              payload={"K": hex(cand[1][0]), "E": E, "ops": 2})
    rows.append({"fn": "sqrt", "arm": "seed-only", "ops": 2, "E": E,
                 "consts": [hex(c) for c in cand[1]]})
    print(f"sqrt seed-only (2 ops): K={hex(cand[1][0])} E={E:.4e}")

    rs = trick_newton(0x5F375A87)
    s2 = (rs[0] + (("FMUL", 0, 0, 5),), rs[1])
    # rs leaves rsqrt estimate in slot 0; we need x * estimate. The rsqrt
    # skeleton overwrote slot 0, so recompute: estimate ends in slot 0 via
    # final FMUL 0,6,7 — change to write t0 (5), then multiply by x.
    s2_skel = tuple(list(rs[0][:-1]) + [("FMUL", 5, 6, 7),
                                        ("FMUL", 0, 0, 5)])
    s2 = (s2_skel, rs[1])
    ok, det = pack.verify_trusted(mold, mold.tidy(s2))
    assert ok, det
    E2 = det["max_rel_err"]
    rows.append({"fn": "sqrt", "arm": "via-rsqrt-A87", "ops": 8, "E": E2,
                 "consts": [hex(c) for c in s2[1]]})
    rec.event("judge", "sqrt_via_rsqrt", payload={"E": E2, "ops": 8})
    print(f"sqrt via rsqrt-A87 (8 ops): E={E2:.4e}")

    # ---------------- log2 -------------------------------------------------
    floors = {}
    for deg, ops in ((1, 2), (5, 10)):
        r = remez(lambda x: mp.log(x) / mp.log(2), "0.00390625", "256",
                  deg, dps=60, grid_n=8192, grid_kind="log")
        floors[ops] = r["bound_low"]
        rec.event("shelf", "log2_poly_floor",
                  payload={"deg": deg, "ops": ops,
                           "bracket": [r["bound_low"], r["bound_high"]],
                           "alternation": r["alternation_points"]})
        print(f"log2 poly floor deg{deg} ({ops} ops): "
              f"[{r['bound_low']:.4e}, {r['bound_high']:.4e}]")

    lpack, lmold = registry.build("log2", {"n_const": 7})
    LT = 1 + lmold.N_CONST
    l3_skel = (("U2F", LT, 0, 0), ("FMUL", LT, LT, 1), ("FADD", 0, LT, 2))
    l10_skel = (("U2F", LT, 0, 0), ("FMUL", LT, LT, 1), ("FADD", LT, LT, 2),
                ("AND32", LT + 1, 0, 3), ("OR32", LT + 1, LT + 1, 4),
                ("FMUL", LT + 2, 5, LT + 1), ("FADD", LT + 2, LT + 2, 6),
                ("FMUL", LT + 2, LT + 2, LT + 1),
                ("FADD", LT + 2, LT + 2, 7), ("FADD", 0, LT, LT + 2))

    def l3_fit(seed):
        """Slope gene by 1-D coarse-to-fine sweep; for each slope the
        optimal OFFSET has a closed form (Chebyshev center of the
        residual: -(max+min)/2) — kills the zero-output desert that
        trapped plain descent (measured: first batch, E=8 plateaus)."""
        import numpy as np
        u = lpack.sample_bits.astype(np.float64)   # U2F values (f64 model)
        t = lpack.sample_truth

        def E_center(c0_bits):
            with np.errstate(all="ignore"):
                c0 = np.uint32(c0_bits).view(np.float32).astype(np.float64)
                r = u * c0 - t
                hi, lo = r.max(), r.min()
                if not (np.isfinite(hi) and np.isfinite(lo)):
                    return float("inf"), 0.0   # NaN keys corrupt sorted()
                return float((hi - lo) / 2), float(-(hi + lo) / 2)
        coarse = sorted((E_center(k << 19)[0], k << 19)
                        for k in range(1 << 13))[:4]
        best = None
        for _, center in coarse:
            c, w = center, 1 << 19
            while w >= 2:
                grid = range(max(0, c - w), min(1 << 32, c + w),
                             max(1, w // 16))
                c = min(grid, key=lambda b: E_center(b)[0])
                w //= 8
            e, off = E_center(c)
            if best is None or e < best[0]:
                best = (e, c, off)
        _, c0_bits, off = best
        cand = (l3_skel, (c0_bits, fb(off), 0, 0, 0, 0, 0))
        # float32-execution polish of both genes on the true metric
        cand, e, _ = bit_descent(lpack, ConstOnlyMold(lmold, l3_skel),
                                 cand, 6, const_indices=(0, 1))
        return cand, e

    l3_results = {}
    for seed in SEEDS:
        l3_results[seed] = l3_fit(seed)

    for name, skel, n_genes, restarts, ops in (
            ("L3", l3_skel, 2, 0, 3), ("L10", l10_skel, 7, 8, 10)):
        for seed in SEEDS:
            rng = random.Random(seed * 131 + ops)
            cmold = ConstOnlyMold(lmold, skel)
            if name == "L3":
                best, _se = l3_results[seed]
            else:
                # warm start: the L3 solution with the correction OFF
                # (poly genes zero) — descent can only improve on it;
                # masks and poly genes are searched from outcome
                l3c = l3_results[seed][0][1]
                warm = (skel, (l3c[0], l3c[1],
                               rng.getrandbits(32), rng.getrandbits(32),
                               0, 0, 0))
                best, best_e = bit_descent(
                    lpack, cmold, warm, 10,
                    const_indices=range(n_genes))[:2]
                for _ in range(restarts):
                    start = cmold.random_candidate(rng)
                    cand, _e1, _n = bit_descent(
                        lpack, cmold, start, 8,
                        const_indices=range(n_genes),
                        metric=lpack.sample_shaped)
                    cand, e, _n = bit_descent(lpack, cmold, cand, 5,
                                              const_indices=range(n_genes))
                    if e < best_e:
                        best, best_e = cand, e
            tidy = lmold.tidy(best)
            ok, det = lpack.verify_trusted(lmold, tidy)
            E = det["max_rel_err"] if ok else float("inf")
            floor = floors[2 if ops == 3 else 10]
            win = ok and E < floor / 10
            row = {"fn": "log2", "arm": name, "seed": seed, "ops": ops,
                   "E": E, "poly_floor": floor,
                   "factor": floor / E if ok and E > 0 else None,
                   "win": win, "consts": [hex(c) for c in tidy[1]]}
            rows.append(row)
            rec.event("judge", "log2_result", payload=row)
            print(f"log2 {name} s{seed} ({ops} ops): E={E:.4e} "
                  f"floor={floor:.4e} factor={row['factor']:.0f}x "
                  f"{'WIN' if win else 'no-win'}")
            print(f"   consts: {row['consts']}")

    wins = sum(1 for r in rows if r.get("win"))
    report = {"run_id": run_id, "seconds": round(time.time() - t0, 1),
              "rows": rows, "log2_wins": wins,
              "PASS": wins == 6}     # predeclared: all 6 log2 runs WIN
    rec.event("foreman", "report", payload=report)
    (rec.run_dir / "report.json").write_text(json.dumps(report, indent=2))
    rec.close()
    print(f"\nlog2 beyond-polynomial WINs: {wins}/6 | PASS: "
          f"{report['PASS']} ({report['seconds']}s)")
    return 0 if report["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
