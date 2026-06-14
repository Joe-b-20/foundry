"""gelu hunt — the most ML-load-bearing activation, and the first
saturating target needing a COMPOSED structure: gelu(x) = x * Phi(x), so
the program is x * R(x) with R a rational approximating Phi (a pure
rational can't match gelu's asymmetric tails -> x right, -> 0 left).

R is fit from outcome to Phi (linearized least-squares + Lawson IRLS,
engine/ratfit.py), float32-rounded via mold.build_gelu, then x*R is
EXHAUSTIVELY verified against gelu. The gelu error is |x|*|R-Phi|, so a
|x|-weighted fit is also tried (prioritizing accuracy where |x| is large).

Floors: weighted=None (absolute) Remez of gelu over [-8,8], proven
polynomial floor for each degree class. gelu[p/q] costs 2p+2q+2 ops (the
extra *x); compared to the equal-op polynomial: [2/2]=10 ops vs deg-5
(10), [3/3]=14 ops vs deg-7 (14). FDIV is Foundry-defined (f64 quot->f32).
Floors are real-arith minimax over a slightly wider interval than the
pack scope -> conservative for the polynomial -> factors are lower bounds.

PASS (predeclared): gelu[2/2] and gelu[3/3] each beat the proven floor of
their equal-op polynomial class.

Run from repo root:  python3 -m scripts.run_gelu_hunt
"""

import json
import sys
import time
from pathlib import Path

import mpmath as mp
import numpy as np

from engine import registry
from engine.ratfit import rational_fit
from engine.recorder import Recorder
from engine.remez import remez


def main():
    t0 = time.time()
    run_id = f"gelu-hunt-{int(t0)}"
    rec = Recorder(Path("runs") / run_id, run_id)
    pack, mold = registry.build("gelu", {"n_const": 8})
    xs = pack.sample_bits.view(np.float32).astype(np.float64)
    phi = pack.phi(xs)

    floors = {}
    gelu_mp = lambda x: mp.mpf("0.5") * x * (1 + mp.erf(x / mp.sqrt(2)))
    for deg, ops in ((5, 10), (7, 14)):
        r = remez(gelu_mp, "-8", "8", deg, dps=40, grid_n=8192)
        floors[ops] = r["bound_low"]
        rec.event("shelf", "gelu_poly_floor",
                  payload={"deg": deg, "ops": ops,
                           "bracket": [r["bound_low"], r["bound_high"]],
                           "alternation": r["alternation_points"]})
        print(f"gelu poly deg{deg} ({ops} ops) proven abs floor: "
              f"{r['bound_low']:.4e}")
    rec.event("foreman", "runspec",
              payload={"floors": floors, "scope_size": pack.scope_size,
                       "structure": "x * rational(Phi)",
                       "pass": "gelu[2/2]<deg5_floor and gelu[3/3]<deg7_floor"},
              reason="predeclaration: R fit from outcome (deterministic), "
                     "exhaustive verify of x*R vs gelu, floors proven "
                     "(conservative-for-poly); FDIV Foundry-defined")

    rows = []
    for p, q, ops, fl_ops in ((2, 2, 10, 10), (3, 3, 14, 14)):
        best = None
        for tag, pw in (("unweighted", None), ("x-weighted", np.abs(xs))):
            a, b = rational_fit(xs, phi, p, q,
                                weight0=(pw if pw is None else pw ** 2))
            cand = mold.build_gelu(p, q, a, b)
            ok, det = pack.verify_trusted(mold, cand)
            E = det["max_rel_err"] if ok else float("inf")
            if best is None or E < best[0]:
                best = (E, cand, tag, det)
        E, cand, tag, det = best
        floor = floors[fl_ops]
        deg = fl_ops // 2
        win = np.isfinite(E) and E < floor
        row = {"gelu_rational": f"x*[{p}/{q}]", "ops": ops, "fit": tag,
               "exhaustive_max_abs": E, "vs_poly_deg": deg,
               "poly_floor": floor,
               "factor_lower": floor / E if np.isfinite(E) and E > 0 else None,
               "beats_equal_op_poly": win,
               "consts": [hex(c) for c in cand[1]], "pretty": mold.pretty(cand),
               "certificate": det.get("certificate") if win else det}
        rows.append(row)
        rec.event("judge", "gelu_rational", payload=row)
        tagw = ("%.1fx lower max abs error than the proven deg-%d (%d-op) "
                "poly floor, under the declared scope/metric"
                % (row["factor_lower"], deg, fl_ops) if win else "NO-WIN")
        print(f"x*[{p}/{q}] ({ops} ops, {tag} fit): exhaustive max abs "
              f"{E:.4e} vs deg-{deg} floor {floor:.4e} -> {tagw}")

    PASS = all(r["beats_equal_op_poly"] for r in rows)
    report = {"run_id": run_id, "seconds": round(time.time() - t0, 1),
              "rows": rows, "PASS": PASS,
              "note": "gelu = x*Phi; program is x*rational(Phi); coeffs from "
                      "outcome; exhaustive; factors lower bounds "
                      "(conservative-for-poly); FDIV Foundry-defined "
                      "f64-quot->f32. ABSOLUTE-error metric (not comparable "
                      "to relative-metric domains like tanh)."}
    rec.event("foreman", "report", payload=report)
    (rec.run_dir / "report.json").write_text(json.dumps(report, indent=2))
    rec.close()
    print(f"\ngelu hunt PASS: {PASS} ({report['seconds']}s)")
    return 0 if PASS else 1


if __name__ == "__main__":
    sys.exit(main())
