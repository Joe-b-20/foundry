"""The tanh HUNT (first true headroom hunt of the portfolio).

Question, predeclared: at op budget B over our 32-bit op set (float
add/sub/mul + integer bit ops + constant genes), can ANY program achieve
lower exhaustive max-rel-error on [0.25, 8) than the PROVEN floor of the
degree-(B/2) polynomial class (weighted-Remez bracket)? Programs strictly
contain float-only straight-line code (which IS polynomial evaluation),
so the only candidate power beyond the floor is the bit ops.

Search: two islands per (budget, seed) — "warm" seeded with our own
calibrated polynomial baseline (Lawson-from-outcome, the engine's rung-1
artifact; no external implementation injected), "cold" blank. Both run
the unclipped shaped loss; every 50 generations the island best gets a
short constants-only descent (memetic); exact bests migrate both ways
every 20 generations. Final best per run: full constants polish on the
true metric, exhaustive verification, then verdicts:

  E < floor*0.98 AND uses bit ops -> BEYOND-POLYNOMIAL find (re-verified)
  E < shelf f32 entry             -> improved float32 implementation at
                                     equal ops (FPminimax-class gain)
  otherwise                       -> null: "no usable signal under the
                                     current representation / search
                                     primitives / budget" at this scope

A null is a valid, predeclared outcome. PASS for the SCRIPT = all runs
complete with verified comparisons (the hunt's value is the verdict
either way, not a guaranteed find).

Run from repo root:  python3 -m scripts.run_tanh_hunt
"""

import json
import random
import struct
import sys
import time
from pathlib import Path

import numpy as np

from domains.tanh import TanhPack
from domains.tanh_shelf import build_shelf, horner_skeleton
from engine import judge
from engine.molds_float import OPS_I, FloatProgMold
from engine.proposers import EvolutionProposer
from engine.recorder import Recorder
from scripts.run_rsqrt_hunt import bit_descent

BUDGETS = (10, 14)          # compare vs deg-5 / deg-7 floors
SEEDS = (0, 1, 2)
POP = 96
GENS = 1500
MIGRATE = 20
MEMETIC = 50
N_CONST = 8


def fb(v):
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def lawson_seed(pack, mold, deg):
    """The engine's own calibrated baseline (outcome data only)."""
    xs = pack.sample_bits.view(np.float32).astype(np.float64)
    V = np.vander(xs, deg + 1, increasing=True)
    A = V / pack.sample_truth[:, None]
    b = np.ones(len(xs))
    W = np.ones(len(xs))
    for _ in range(150):
        sw = np.sqrt(W)[:, None]
        c, *_ = np.linalg.lstsq(A * sw, b * np.sqrt(W), rcond=None)
        e = np.abs(A @ c - b)
        W = W * (e + 1e-300)
        W /= W.sum()
    consts = tuple(fb(ci) for ci in c) + tuple(
        fb(0.0) for _ in range(N_CONST - deg - 1))
    cand = mold.tidy((horner_skeleton(deg, mold), consts))
    cand, _e, _n = bit_descent(pack, mold, cand, 6,
                               const_indices=range(N_CONST))
    return cand


def run_hunt(budget, seed, pack, shelf_by_deg, rec):
    t0 = time.time()
    deg = budget // 2
    mold = FloatProgMold(n_const=N_CONST, max_len=budget)
    warm_seed = lawson_seed(pack, mold, deg)
    islands = {
        "warm": EvolutionProposer(pop_size=POP, seeds=[warm_seed]),
        "cold": EvolutionProposer(pop_size=POP),
    }
    rngs = {nm: random.Random((seed << 8) + i)
            for i, nm in enumerate(islands)}
    best = None
    evals = 0
    for gen in range(GENS):
        for nm, prop in islands.items():
            ctx = {"rng": rngs[nm], "mold": mold, "batch": POP,
                   "init_length": budget - 2}
            results = []
            for cand in prop.propose(ctx):
                tidy, sc, _cost = judge.score(pack, mold, cand)
                results.append((tidy, sc))
                evals += 1
                if best is None or sc > best[0]:
                    best = (sc, tidy)
            prop.feedback(results)
        if gen % MIGRATE == 0:
            bs = [p.best() for p in islands.values() if p.best()]
            for p in islands.values():
                for b in bs:
                    p.absorb([b])
        if gen and gen % MEMETIC == 0:
            for nm, prop in islands.items():
                pb = prop.best()
                if pb:
                    cand, _e, n = bit_descent(
                        pack, mold, pb[0], 1,
                        const_indices=range(N_CONST),
                        metric=pack.sample_shaped)
                    evals += n
                    tidy, sc, _ = judge.score(pack, mold, cand)
                    evals += 1
                    prop.absorb([(tidy, sc)])
                    if sc > best[0]:
                        best = (sc, tidy)

    # final: polish constants on the TRUE metric — for BOTH the search's
    # shaped-best AND the preserved warm seed, keeping the better. The
    # shaped rank drifts toward the mean-optimum (max-metric worse), and
    # without this the hunt loses its own baseline (measured: first hunt
    # batch finished ABOVE the seed's error — wrong-metric drift bug).
    finalists = []
    for c in (best[1], warm_seed):
        cand, e, n = bit_descent(pack, mold, c, 8,
                                 const_indices=range(N_CONST))
        evals += n
        finalists.append((e, cand))
    tidy = mold.tidy(min(finalists)[1])
    ok, det = pack.verify_trusted(mold, tidy)
    E = det["max_rel_err"] if ok else float("inf")
    floor = shelf_by_deg[deg]["rel_bracket_real_model"][0]
    e_shelf = shelf_by_deg[deg]["max_rel_err_f32"]
    uses_bits = any(op in OPS_I for (op, *_r) in tidy[0])
    if ok and E < floor * 0.98 and uses_bits:
        ok2, det2 = pack.verify_trusted(mold, tidy)   # re-verify hard
        verdict = ("BEYOND-POLYNOMIAL FIND (re-verified)" if ok2
                   and det2["max_rel_err"] < floor * 0.98 else
                   "sub-floor reading did not re-verify — suspect us")
    elif ok and E < e_shelf:
        verdict = ("improved float32 implementation at equal ops "
                   "(FPminimax-class gain over the rounded-Remez entry)")
    else:
        verdict = ("null: no usable signal under the current "
                   "representation / search primitives / budget "
                   "(beyond-polynomial structure not found at this scope)")
    row = {"budget": budget, "seed": seed, "evals": evals,
           "seconds": round(time.time() - t0, 1), "verified": ok,
           "E": E, "poly_floor_proven": floor, "shelf_f32": e_shelf,
           "uses_bit_ops": uses_bits, "ops": len(tidy[0]),
           "pretty": mold.pretty(tidy),
           "found_cand": {"instrs": [list(i) for i in tidy[0]],
                          "consts": [hex(c) for c in tidy[1]]},
           "verdict": verdict}
    rec.event("judge", "hunt_result", payload=row)
    return row


def main():
    t0 = time.time()
    run_id = f"tanh-hunt-{int(t0)}"
    rec = Recorder(Path("runs") / run_id, run_id)
    pack = TanhPack()
    shelf = build_shelf(pack, lambda n: FloatProgMold(n_const=n, max_len=16))
    shelf_by_deg = {e["deg"]: e for e in shelf}
    rec.event("foreman", "runspec",
              payload={"budgets": BUDGETS, "seeds": SEEDS, "pop": POP,
                       "gens": GENS, "question": "beat the proven "
                       "polynomial floor at equal ops with bit/float "
                       "hybrids?", "null_is_valid": True},
              reason="predeclaration: verdict rules and comparisons fixed "
                     "before the hunt; a null is a first-class outcome")
    rows = []
    for budget in BUDGETS:
        for seed in SEEDS:
            r = run_hunt(budget, seed, pack, shelf_by_deg, rec)
            rows.append(r)
            print(f"ops<={budget} s{seed}: E={r['E']:.4e} "
                  f"floor={r['poly_floor_proven']:.4e} "
                  f"shelf={r['shelf_f32']:.4e} bits={r['uses_bit_ops']} "
                  f"t={r['seconds']}s")
            print(f"   {r['pretty'][:110]}")
            print(f"   -> {r['verdict']}")
    completed = all(r["verified"] for r in rows)
    beyond = [r for r in rows if r["verdict"].startswith("BEYOND")]
    gains = [r for r in rows if r["verdict"].startswith("improved")]
    report = {"run_id": run_id, "seconds": round(time.time() - t0, 1),
              "rows": rows, "beyond_polynomial": len(beyond),
              "f32_gains": len(gains), "PASS_script": completed}
    rec.event("foreman", "report", payload=report)
    (rec.run_dir / "report.json").write_text(json.dumps(report, indent=2))
    rec.close()
    print(f"\nbeyond-polynomial finds: {len(beyond)} | f32 gains: "
          f"{len(gains)} | script PASS: {completed} "
          f"({report['seconds']}s)")
    return 0 if completed else 1


if __name__ == "__main__":
    sys.exit(main())
