"""Generic rational hunt for the saturating family (sigmoid was the first
demand, tanh the second -> generalized). For a domain: fit P/Q from outcome
(linearized least-squares + Lawson IRLS, engine/ratfit.py), float32-round
via mold.build_rational, EXHAUSTIVELY verify, and compare to the PROVEN
minimax-polynomial floor of the SAME total degree (a [p/q] rational at
2p+2q+1 ops vs the deg-(p+q) polynomial floor at 2(p+q) ops).

FDIV is Foundry-defined (f64 quotient -> f32), not IEEE f32 division — the
label stays visible. Coefficient fit is deterministic (no RNG -> no seed
variance). Floors are real-arithmetic minimax over the interval, which is
>= any float32 polynomial and (for sigmoid) over a slightly wider interval
than the pack scope — both conservative FOR the polynomial, so each factor
is a lower bound on the gap. Metric matches the pack (sigmoid abs, tanh
relative via weight).

PASS (predeclared): every rational with total degree >= 4 ([2/2], [3/3])
beats the proven floor of its equal-total-degree polynomial class. [1/1]
reported for context.

Run from repo root:  python3 -m scripts.run_rational_hunt [sigmoid|tanh]
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

CONFIGS = {
    "sigmoid": {"interval": ("-8", "8"), "metric": "abs", "weight": None,
                "f": lambda x: 1 / (1 + mp.e ** (-x)),
                "rats": [(1, 1), (2, 2), (3, 3)]},
    "tanh": {"interval": ("0.25", "8"), "metric": "rel",
             "weight": (lambda x: mp.tanh(x)),
             "f": (lambda x: mp.tanh(x)),
             "rats": [(2, 2), (3, 3)]},
}


def main():
    domain = sys.argv[1] if len(sys.argv) > 1 else "sigmoid"
    cfg = CONFIGS[domain]
    t0 = time.time()
    run_id = f"{domain}-rational-{int(t0)}"
    rec = Recorder(Path("runs") / run_id, run_id)
    pack, mold = registry.build(domain, {"n_const": 8})
    assert pack.err_kind == cfg["metric"], (pack.err_kind, cfg["metric"])

    degs = sorted({p + q for (p, q) in cfg["rats"]} | {2})
    floors = {}
    for d in degs:
        r = remez(cfg["f"], cfg["interval"][0], cfg["interval"][1], d,
                  dps=40, grid_n=8192, weight=cfg["weight"])
        floors[d] = r["bound_low"]
        rec.event("shelf", "poly_floor",
                  payload={"deg": d, "ops": 2 * d,
                           "bracket": [r["bound_low"], r["bound_high"]],
                           "alternation": r["alternation_points"]})
        print(f"{domain} poly deg{d} ({2*d} ops) proven {cfg['metric']} "
              f"floor: {r['bound_low']:.4e}")
    rec.event("foreman", "runspec",
              payload={"domain": domain, "metric": cfg["metric"],
                       "interval": cfg["interval"], "floors": floors,
                       "scope_size": pack.scope_size,
                       "pass": "every rational total-deg>=4 beats its "
                               "equal-deg poly floor"},
              reason="predeclaration: rational coeffs from outcome "
                     "(deterministic), exhaustive verify, floors proven "
                     "(conservative-for-poly); FDIV Foundry-defined")

    xs = pack.sample_bits.view(np.float32).astype(np.float64)
    fs = pack.sample_truth
    rows = []
    for p, q in cfg["rats"]:
        a, b = rational_fit(xs, fs, p, q)
        cand = mold.build_rational(p, q, a, b)
        ok, det = pack.verify_trusted(mold, cand)
        E = det["max_rel_err"] if ok else float("inf")
        d = p + q
        floor = floors[d]
        win = ok and E < floor and d >= 4
        row = {"rational": f"[{p}/{q}]", "ops": 2 * p + 2 * q + 1,
               "exhaustive_max_err": E, "metric": cfg["metric"],
               "vs_poly_deg": d, "poly_floor": floor,
               "factor_lower": floor / E if ok and E > 0 else None,
               "beats_equal_deg_poly": win,
               "consts": [hex(c) for c in cand[1]], "pretty": mold.pretty(cand),
               "certificate": det.get("certificate") if ok else det}
        rows.append(row)
        rec.event("judge", "rational", payload=row)
        if d < 4:
            tag = "context (total deg < 4; expected ~equal to poly)"
        elif win:
            tag = ("%.1fx lower max %s error than the proven deg-%d (%d-op) "
                   "poly floor, under the declared scope/metric"
                   % (row["factor_lower"], cfg["metric"], d, 2 * d))
        else:
            tag = "NO-WIN vs the equal-degree polynomial floor"
        print(f"[{p}/{q}] ({row['ops']} ops): exhaustive max {cfg['metric']} "
              f"{E:.4e} vs deg-{d} floor {floor:.4e} -> {tag}")

    PASS = all(r["beats_equal_deg_poly"] for r in rows
               if r["vs_poly_deg"] >= 4)
    report = {"run_id": run_id, "domain": domain,
              "seconds": round(time.time() - t0, 1), "rows": rows,
              "PASS": PASS,
              "note": "rational (FDIV, Foundry-defined f64-quot->f32) vs "
                      "PROVEN poly floor; coeffs from outcome (deterministic "
                      "IRLS); exhaustive; factors are lower bounds "
                      "(conservative-for-poly)"}
    rec.event("foreman", "report", payload=report)
    (rec.run_dir / "report.json").write_text(json.dumps(report, indent=2))
    rec.close()
    print(f"\n{domain} rational hunt PASS: {PASS} ({report['seconds']}s)")
    return 0 if PASS else 1


if __name__ == "__main__":
    sys.exit(main())
