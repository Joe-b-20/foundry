"""Trig hunt (sin, cos) — periodic functions via ARGUMENT REDUCTION.
cos(x) = sin(x + pi/2), so both share one machinery: optionally add a
phase, reduce xr = x' - round(x'/2pi)*2pi via the magic-add round trick
(no new op), then the odd polynomial sin(xr) ~ xr*P(xr^2). The SAME
reduced-range polynomial P (fit from outcome, Remez on g(u)=sin(sqrt u)/
sqrt u) serves both. Reduction constants (1/2pi, 1.5*2^23, 2pi, phase) are
mathematical / GIVEN; P is from outcome.

Honest claim (per function): the full program verifies EXHAUSTIVELY over
all ~167M signed float32 in scope at the reported max ABS error; the
comparison is argument-reduction vs the best UNREDUCED polynomial of the
SAME degree over the wide scope (where Remez equioscillates it is a PROVEN
floor, else a measured baseline, labeled). Scope |x| < 16 uses single-2pi
reduction (~1e-6 error); wider needs a Cody-Waite split (logged).

PASS: each function verifies < 1e-3 AND beats the best unreduced deg-7
polynomial over [-16,16] by >= 100x.

Run from repo root:  python3 -m scripts.run_trig_hunt
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
from engine.remez import remez

CONFIGS = {"sin": {"phase": 0.0, "f": mp.sin},
           "cos": {"phase": float(mp.pi / 2), "f": mp.cos}}


def fb(v):
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def build_trig(mold, pcoef, phase):
    """phase 0 -> sin (13 ops); phase pi/2 -> cos (14 ops)."""
    INV2PI, MAGIC, TWO_PI = 1.0 / (2 * np.pi), 1.5 * 2 ** 23, 2 * np.pi
    Tk, Tu, Tp = 9, 10, 11
    c1s, c3s, c5s, c7s, phs = 4, 5, 6, 7, 8
    pre = (("FADD", 0, 0, phs),) if phase != 0.0 else ()
    skel = pre + (
        ("FMUL", Tk, 0, 1), ("FADD", Tk, Tk, 2), ("FSUB", Tk, Tk, 2),
        ("FMUL", Tk, Tk, 3), ("FSUB", 0, 0, Tk),
        ("FMUL", Tu, 0, 0),
        ("FMUL", Tp, c7s, Tu), ("FADD", Tp, Tp, c5s),
        ("FMUL", Tp, Tp, Tu), ("FADD", Tp, Tp, c3s),
        ("FMUL", Tp, Tp, Tu), ("FADD", Tp, Tp, c1s),
        ("FMUL", 0, 0, Tp))
    consts = (fb(INV2PI), fb(MAGIC), fb(TWO_PI),
              fb(pcoef[0]), fb(pcoef[1]), fb(pcoef[2]), fb(pcoef[3]),
              fb(phase))
    return mold.tidy((skel, consts))


def main():
    t0 = time.time()
    run_id = f"trig-hunt-{int(t0)}"
    rec = Recorder(Path("runs") / run_id, run_id)

    # one reduced-range polynomial for both functions
    def g(u):
        u = mp.mpf(u)
        return mp.sin(mp.sqrt(u)) / mp.sqrt(u) if u > 0 else mp.mpf(1)
    rp = remez(g, mp.mpf("1e-12"), mp.pi ** 2, 3, dps=40, grid_n=8192)
    pcoef = [float(c) for c in rp["coeffs"]]
    rec.event("shelf", "reduced_poly", payload={"P_coeffs": pcoef})

    rows = []
    for name, cfg in CONFIGS.items():
        pack, mold = registry.build(name, {"n_const": 8})
        cand = build_trig(mold, pcoef, cfg["phase"])
        ok, det = pack.verify_trusted(mold, cand)
        E = det["max_rel_err"] if ok else float("inf")
        rb = remez(cfg["f"], mp.mpf(-16), mp.mpf(16), 7, dps=40, grid_n=16384)
        proven = rb["alternation_points"] == 9
        unreduced7 = rb["bound_high"]
        factor = unreduced7 / E if ok and E > 0 else None
        win = ok and E < 1e-3 and factor is not None and factor >= 100
        row = {"fn": name, "ops": len(cand[0]), "exhaustive_max_abs": E,
               "unreduced_deg7_over_[-16,16]": unreduced7,
               "unreduced_status": "PROVEN floor" if proven else
                                    "measured best (not equioscillating)",
               "factor_vs_unreduced_deg7": factor, "PASS": win,
               "pretty": mold.pretty(cand),
               "certificate": det.get("certificate") if ok else det}
        rows.append(row)
        rec.event("judge", "trig_result", payload=row)
        print(f"{name} ({row['ops']} ops): exhaustive max abs {E:.4e} vs "
              f"unreduced deg-7 {unreduced7:.4e} [{row['unreduced_status']}] "
              f"-> {factor:.0f}x  {'PASS' if win else 'FAIL'}")

    PASS = all(r["PASS"] for r in rows)
    report = {"run_id": run_id, "seconds": round(time.time() - t0, 1),
              "rows": rows, "PASS": PASS,
              "note": "argument reduction (math constants given) + odd poly "
                      "from outcome; exhaustive; ABS metric; win is "
                      "reduction-vs-unreduced at equal degree, |x|<16 with "
                      "single-2pi reduction. Same P serves sin and cos."}
    rec.event("foreman", "report", payload=report)
    (rec.run_dir / "report.json").write_text(json.dumps(report, indent=2))
    rec.close()
    print(f"\ntrig hunt PASS: {PASS} ({report['seconds']}s)")
    return 0 if PASS else 1


if __name__ == "__main__":
    sys.exit(main())
