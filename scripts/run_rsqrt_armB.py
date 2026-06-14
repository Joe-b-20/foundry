"""rsqrt arm-B, done right — JOINT optimization of the magic constant AND
the Newton-step coefficients from outcome (the coupled optimizer the
earlier bit-descent arm B lacked, leaving it 0/3).

Structure: y0 = magic - (x>>1); y = y0 * (k1 - k2*x*y0*y0). Standard Newton
is k1=1.5, k2=0.5 (the exact Newton step). But y0 from the magic trick has
a BIASED error, so a MODIFIED step (k1,k2 tuned off 1.5,0.5) corrects it
better at the same op count — this is the Moroz/Walczyk/Cieslinski insight
(Computation 2019; Entropy 2021). Their exact constants are NOT carried
from memory; if our from-outcome (magic,k1,k2) lands in their improvement
class (~2x better than standard Newton) that is a rediscovery, flagged
pending a citation check.

Optimizer: alternate {1-D coarse-to-fine sweep of magic} with {Nelder-Mead
on (k1,k2)} (the two have very different scales — magic ~1.6e9, k ~O(1) —
so optimizing them separately is far better conditioned than a joint
bit-descent), then a final carry-aware bit-polish of all three constants
on the true metric, then EXHAUSTIVE verification. Scope and metric match
the A87 result (all float32 in [2^-8, 2^8), max rel error vs float64).

PASS (predeclared): exhaustive max rel error < A87's 1.751288e-3 (i.e.
the joint search beats the standard-Newton optimum). A ~2x improvement
would match the Moroz-class.

Run from repo root:  python3 -m scripts.run_rsqrt_armB
"""

import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from engine import registry
from engine.recorder import Recorder
from domains.rsqrt_shelf import trick_newton
from scripts.run_rsqrt_hunt import bit_descent

A87_SCOPE_ERR = 1.751288e-3


def fb(v):
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def main():
    t0 = time.time()
    run_id = f"rsqrt-armB-{int(t0)}"
    rec = Recorder(Path("runs") / run_id, run_id)
    pack, mold = registry.build("rsqrt", {"lo_exp": -8, "hi_exp": 8})
    rec.event("foreman", "runspec",
              payload={"structure": "y0=magic-(x>>1); y=y0*(k1-k2*x*y0*y0)",
                       "search": "alt magic-sweep + NelderMead(k1,k2) + "
                                 "bit-polish; exhaustive verify",
                       "A87_baseline": A87_SCOPE_ERR,
                       "pass": "exhaustive max rel < A87 (1.751288e-3)"},
              reason="predeclaration: joint magic+Newton from outcome; "
                     "Moroz-class constants NOT carried from memory")

    def err(magic, k1, k2):
        return pack.sample_max_rel(
            mold, mold.tidy(trick_newton(int(magic) & 0xFFFFFFFF,
                                         half=k2, three_halves=k1)))

    def sweep_magic(k1, k2):
        coarse = sorted((err(k << 18, k1, k2), k << 18)
                        for k in range(1 << 14))[:4]
        best = None
        for _, center in coarse:
            c, w = center, 1 << 18
            while w >= 2:
                grid = range(max(0, c - w), min(1 << 32, c + w),
                             max(1, w // 16))
                c = min(grid, key=lambda m: err(m, k1, k2))
                w //= 8
            e = err(c, k1, k2)
            if best is None or e < best[0]:
                best = (e, c)
        return best[1]

    magic, k1, k2 = 0x5F375A87, 1.5, 0.5
    for rnd in range(4):
        magic = sweep_magic(k1, k2)
        res = minimize(lambda kk: err(magic, kk[0], kk[1]), [k1, k2],
                       method="Nelder-Mead",
                       options={"xatol": 1e-7, "fatol": 1e-9, "maxiter": 400})
        k1, k2 = float(res.x[0]), float(res.x[1])
        rec.event("foreman", "alt_round",
                  payload={"round": rnd, "magic": hex(magic), "k1": k1,
                           "k2": k2, "sample_err": float(res.fun)})

    # final carry-aware bit-polish of all three constants on the true metric
    cand = mold.tidy(trick_newton(magic, half=k2, three_halves=k1))
    cand, _e, _n = bit_descent(pack, mold, cand, 10, const_indices=(0, 2, 3))
    ok, det = pack.verify_trusted(mold, cand)
    E = det["max_rel_err"] if ok else float("inf")

    a87 = mold.tidy(trick_newton(0x5F375A87))
    _, det87 = pack.verify_trusted(mold, a87)
    e87 = det87["max_rel_err"]

    improved = ok and E < e87
    moroz_class = ok and E < e87 * 0.6        # ~2x territory
    consts = cand[1]
    k1_final = struct.unpack("<f", struct.pack("<I", consts[3]))[0]
    k2_final = struct.unpack("<f", struct.pack("<I", consts[2]))[0]
    report = {"run_id": run_id, "seconds": round(time.time() - t0, 1),
              "magic": hex(consts[0]), "k1_three_halves": k1_final,
              "k2_half": k2_final, "exhaustive_max_rel": E,
              "A87_standard_newton": e87,
              "improvement_factor": e87 / E if ok and E > 0 else None,
              "beats_standard_newton": improved,
              "moroz_class_~2x": moroz_class, "PASS": improved,
              "pretty": mold.pretty(cand),
              "certificate": det.get("certificate") if ok else det,
              "note": "structure standard; magic+Newton coeffs from outcome "
                      "(joint). If ~2x, matches the Moroz/Walczyk/Cieslinski "
                      "modified-Newton class (their exact constants not "
                      "carried) -> UNRESOLVED pending citation check. Scope "
                      "all float32 [2^-8,2^8), max rel vs float64."}
    rec.event("judge", "armB_result", payload=report)
    rec.event("recognizer", "verdict", payload={
        "label": ("modified Newton (k1=%.5f,k2=%.5f) + magic %s from "
                  "outcome: exhaustive %.4e = %.2fx better than standard-"
                  "Newton A87 (%.4e). %s" %
                  (k1_final, k2_final, hex(consts[0]), E, e87 / E, e87,
                   "Moroz-class ~2x — UNRESOLVED pending citation"
                   if moroz_class else "improvement beyond standard Newton"))
                 if improved else "did not beat standard-Newton A87"})
    rec.event("foreman", "report", payload=report)
    (rec.run_dir / "report.json").write_text(json.dumps(report, indent=2))
    rec.close()
    print(f"magic={hex(consts[0])} k1={k1_final:.6f} k2={k2_final:.6f}")
    print(f"exhaustive max rel = {E:.6e}  (A87 standard-Newton {e87:.6e})")
    print(f"improvement = {e87/E:.2f}x  beats_standard={improved}  "
          f"moroz_class~2x={moroz_class}")
    print(f"PASS: {report['PASS']} ({report['seconds']}s)")
    return 0 if report["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
