"""sin hunt — the first PERIODIC target, and a demonstration of ARGUMENT
REDUCTION (the technique periodic functions demand). A fixed low-degree
polynomial cannot track sin's ~5 oscillations over [-16,16]; reduce x to
one period first, then a small polynomial suffices.

Program (13 ops): magic-add argument reduction xr = x - round(x/2pi)*2pi
(FMUL/FADD/FSUB, the round trick — no new op), then the odd polynomial
sin(xr) ~ xr * P(xr^2). The reduction constants (1/2pi, 1.5*2^23, 2pi) are
mathematical / GIVEN; the polynomial P is found FROM OUTCOME (minimax via
Remez on the reduced-range function g(u)=sin(sqrt u)/sqrt u, deg 3 in u =
degree 7 in xr). Everything is verified EXHAUSTIVELY over all ~167M signed
float32 in scope.

Comparison (the honest, equal-degree statement): the SAME degree-7
polynomial WITHOUT reduction cannot approximate sin over [-16,16] — best
unreduced deg-7 max error is ~O(1). So argument reduction cuts the error
from ~O(1) to the reduced-poly level at ~equal op count. Where Remez
equioscillates over [-16,16], its bound is a PROVEN floor; where it does
not (low degree vs many oscillations), the best unreduced poly is reported
as a measured baseline, labeled as such.

PASS (predeclared): the range-reduced sin verifies exhaustively at max abs
error < 1e-3, AND beats the best unreduced deg-7 polynomial over [-16,16]
by >= 100x.

Run from repo root:  python3 -m scripts.run_sin_hunt
"""

import json
import struct
import sys
import time
from pathlib import Path

import mpmath as mp
import numpy as np

from engine import registry
from engine.recorder import Recorder
from engine.remez import remez, polyval


def fb(v):
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def build_sin(mold, pcoef):
    """pcoef = [c1, c3, c5, c7] (coeffs of P(u), u=xr^2). 13 ops."""
    INV2PI, MAGIC, TWO_PI = 1.0 / (2 * np.pi), 1.5 * 2 ** 23, 2 * np.pi
    Tk, Tu, Tp = 9, 10, 11                  # temps (N_CONST=8 -> temps 9..12)
    c1s, c3s, c5s, c7s = 4, 5, 6, 7         # const slots for the poly
    skel = (("FMUL", Tk, 0, 1), ("FADD", Tk, Tk, 2), ("FSUB", Tk, Tk, 2),
            ("FMUL", Tk, Tk, 3), ("FSUB", 0, 0, Tk),          # xr in slot 0
            ("FMUL", Tu, 0, 0),                               # u = xr^2
            ("FMUL", Tp, c7s, Tu), ("FADD", Tp, Tp, c5s),     # Horner P(u)
            ("FMUL", Tp, Tp, Tu), ("FADD", Tp, Tp, c3s),
            ("FMUL", Tp, Tp, Tu), ("FADD", Tp, Tp, c1s),
            ("FMUL", 0, 0, Tp))                               # xr * P(u)
    consts = (fb(INV2PI), fb(MAGIC), fb(TWO_PI),
              fb(pcoef[0]), fb(pcoef[1]), fb(pcoef[2]), fb(pcoef[3]))
    return mold.tidy((skel, consts))


def main():
    t0 = time.time()
    run_id = f"sin-hunt-{int(t0)}"
    rec = Recorder(Path("runs") / run_id, run_id)
    pack, mold = registry.build("sin", {"n_const": 8})

    # fit P(u) = sin(sqrt u)/sqrt u on u in [0, pi^2] (deg 3 -> deg-7 odd)
    def g(u):
        u = mp.mpf(u)
        return mp.sin(mp.sqrt(u)) / mp.sqrt(u) if u > 0 else mp.mpf(1)
    r = remez(g, mp.mpf("1e-12"), mp.pi ** 2, 3, dps=40, grid_n=8192)
    pcoef = [float(c) for c in r["coeffs"]]
    rec.event("shelf", "reduced_poly",
              payload={"P_coeffs": pcoef, "bracket": [r["bound_low"],
                       r["bound_high"]], "alternation": r["alternation_points"]})

    cand = build_sin(mold, pcoef)
    ok, det = pack.verify_trusted(mold, cand)
    E = det["max_rel_err"] if ok else float("inf")
    print(f"range-reduced sin (13 ops): exhaustive max abs = {E:.4e}  ok={ok}")

    # baselines: best UNREDUCED polynomial over [-16,16] (same/up degrees)
    base = {}
    for deg in (7, 11, 15):
        rb = remez(mp.sin, mp.mpf(-16), mp.mpf(16), deg, dps=40, grid_n=16384)
        proven = rb["alternation_points"] == deg + 2
        base[deg] = {"ops": 2 * deg, "best_max_err": rb["bound_high"],
                     "floor_low": rb["bound_low"], "equioscillates": proven,
                     "status": "PROVEN floor" if proven else
                               "measured best (Remez did not equioscillate "
                               "-> low degree vs many oscillations)"}
        print(f"unreduced poly deg{deg} ({2*deg} ops) over [-16,16]: "
              f"best max err {rb['bound_high']:.4e} [{base[deg]['status']}]")
    rec.event("shelf", "unreduced_baselines", payload=base)

    factor7 = base[7]["best_max_err"] / E if ok and E > 0 else None
    PASS = ok and E < 1e-3 and factor7 is not None and factor7 >= 100
    row = {"run_id": run_id, "seconds": round(time.time() - t0, 1),
           "reduced_sin_ops": 13, "exhaustive_max_abs": E,
           "P_coeffs": pcoef,
           "vs_unreduced_deg7": {"best_max_err": base[7]["best_max_err"],
                                 "factor": factor7,
                                 "status": base[7]["status"]},
           "unreduced_baselines": base,
           "consts": [hex(c) for c in cand[1]], "pretty": mold.pretty(cand),
           "certificate": det.get("certificate") if ok else det, "PASS": PASS,
           "note": "argument reduction (math constants given) + odd poly from "
                   "outcome; exhaustive; ABS metric. The win is reduction vs "
                   "NO-reduction at equal poly degree, scoped to |x|<16 with "
                   "single-2pi reduction (~1e-6 reduction error)."}
    rec.event("judge", "sin_result", payload=row)
    rec.event("recognizer", "verdict", payload={
        "label": ("argument reduction makes a deg-7 polynomial track sin "
                  "over ~5 periods: exhaustive max abs %.3e vs ~%.2e for the "
                  "best UNREDUCED deg-7 poly = %.0fx, at equal degree"
                  % (E, base[7]["best_max_err"], factor7)) if PASS else
                 "did not clear the predeclared bar"})
    rec.event("foreman", "report", payload=row)
    (rec.run_dir / "report.json").write_text(json.dumps(row, indent=2))
    rec.close()
    print(f"\n{cand and mold.pretty(cand)}")
    print(f"sin hunt PASS: {PASS} ({row['seconds']}s)")
    return 0 if PASS else 1


if __name__ == "__main__":
    sys.exit(main())
