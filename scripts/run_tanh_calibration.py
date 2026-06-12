"""tanh rung-1 calibration: Horner skeleton of degree d given,
COEFFICIENTS from outcome — random float32 genes, cyclic carry-aware
bit-descent under the sample metric — measured against OUR weighted-Remez
relative-error floor (proven for the exact-arithmetic polynomial class)
and the float32-rounded Remez shelf entry.

PASS (predeclared): exhaustive E_found <= 1.10 * E_shelf_f32 for degrees
3 and 5, all 3 seeds. If E_found < E_shelf_f32 the label notes the
FPminimax effect (coefficients fitted in float32 arithmetic can beat
rounded real-optimal coefficients — cf. Sollya's fpminimax; expected, not
novel). E_found below the real-model floor would be CONTRADICTS-PROVEN-
BOUND for exact polynomials — but the executed float32 Horner is a
rounded evaluation, so tiny dips below are rounding artifacts, flagged
not celebrated.

Run from repo root:  python3 -m scripts.run_tanh_calibration
"""

import json
import random
import sys
import time
from pathlib import Path

from domains.tanh import TanhPack
from domains.tanh_shelf import build_shelf, horner_skeleton
from engine.molds_float import FloatProgMold
from engine.recorder import Recorder
from scripts.run_rsqrt_hunt import ConstOnlyMold, bit_descent

DEGREES = (3, 5)
SEEDS = (0, 1, 2)
RESTARTS = 6
PASSES = 8


def main():
    t0 = time.time()
    run_id = f"tanh-calibration-{int(t0)}"
    rec = Recorder(Path("runs") / run_id, run_id)
    pack = TanhPack()
    shelf = build_shelf(pack, lambda n: FloatProgMold(n_const=n, max_len=16))
    by_deg = {e["deg"]: e for e in shelf}
    rec.event("foreman", "runspec",
              payload={"degrees": DEGREES, "seeds": SEEDS,
                       "restarts": RESTARTS, "passes": PASSES,
                       "shelf": [{k: e[k] for k in
                                  ("name", "ops", "max_rel_err_f32",
                                   "rel_bracket_real_model")}
                                 for e in shelf]},
              reason="predeclaration: PASS = exhaustive E <= 1.10x shelf "
                     "f32 entry for deg 3 and 5, all seeds")
    for e in shelf:
        lo, hi = e["rel_bracket_real_model"]
        print(f"shelf {e['name']}: E_f32={e['max_rel_err_f32']:.4e} "
              f"floor=[{lo:.4e},{hi:.4e}]")

    rows, ok_all = [], True
    for deg in DEGREES:
        mold = FloatProgMold(n_const=deg + 1, max_len=2 * deg + 2)
        skel = horner_skeleton(deg, mold)
        cmold = ConstOnlyMold(mold, skel)
        floor = by_deg[deg]["rel_bracket_real_model"][0]
        e_shelf = by_deg[deg]["max_rel_err_f32"]
        # from-outcome minimax initializer for a linear-in-parameters
        # class: Lawson's iteratively-reweighted least squares (Lawson
        # 1961) on the SAMPLE — converges to the (relative-error) minimax
        # coefficients using outcome data only. Plain L2 lands ~1.8x off
        # minimax (measured); the weight iteration closes that gap.
        import struct as _st
        import numpy as np
        xs = pack.sample_bits.view(np.float32).astype(np.float64)
        V = np.vander(xs, deg + 1, increasing=True)
        A = V / pack.sample_truth[:, None]      # rows scaled: relative err
        b = np.ones(len(xs))
        W = np.ones(len(xs))
        for _ in range(150):            # Lawson converges linearly: be patient
            sw = np.sqrt(W)[:, None]
            c, *_ = np.linalg.lstsq(A * sw, b * np.sqrt(W), rcond=None)
            e = np.abs(A @ c - b)
            W = W * (e + 1e-300)
            W /= W.sum()
        lstsq_init = (skel, tuple(
            _st.unpack("<I", _st.pack("<f", float(ci)))[0] for ci in c))

        for seed in SEEDS:
            t1 = time.time()
            rng = random.Random(seed * 97 + deg)
            best, best_e, evals = None, float("inf"), 0
            for r in range(RESTARTS):
                if r == 0:
                    # Lawson warm start is already near-minimax: polishing
                    # on the shaped loss would pull it AWAY toward the
                    # mean-optimum — go straight to the true metric
                    cand, e, n2 = bit_descent(pack, cmold, lstsq_init, 10,
                                              const_indices=range(deg + 1))
                    evals += n2
                else:
                    # cold restarts: smooth shaped loss first (coordinate
                    # descent provably stalls on the nonsmooth max metric
                    # from cold — measured, calibration v1), then polish
                    start = cmold.random_candidate(rng)
                    cand, _e1, n1 = bit_descent(pack, cmold, start, PASSES,
                                                const_indices=range(deg + 1),
                                                metric=pack.sample_shaped)
                    cand, e, n2 = bit_descent(pack, cmold, cand, 4,
                                              const_indices=range(deg + 1))
                    evals += n1 + n2
                if e < best_e:
                    best, best_e = cand, e
            tidy = mold.tidy(best)
            ok, det = pack.verify_trusted(mold, tidy)
            E = det["max_rel_err"] if ok else float("inf")
            passed = ok and E <= 1.10 * e_shelf
            ok_all &= passed
            labels = []
            if ok and E < e_shelf:
                labels.append("beats the rounded-Remez shelf entry "
                              "(FPminimax effect — expected, not novel)")
            if ok and E < floor:
                labels.append("below the exact-poly floor: rounding "
                              "artifact of float32 evaluation — flagged")
            row = {"deg": deg, "seed": seed, "evals": evals,
                   "seconds": round(time.time() - t1, 2),
                   "E_found": E, "E_shelf_f32": e_shelf,
                   "floor_real_model": floor, "PASS": passed,
                   "consts": [hex(c) for c in tidy[1]], "labels": labels}
            rows.append(row)
            rec.event("judge", "result", payload=row)
            print(f"deg{deg} s{seed}: E={E:.4e} shelf={e_shelf:.4e} "
                  f"floor={floor:.4e} {'PASS' if passed else 'FAIL'} "
                  f"{'; '.join(labels)}")

    report = {"run_id": run_id, "seconds": round(time.time() - t0, 1),
              "rows": rows, "PASS": ok_all}
    rec.event("foreman", "report", payload=report)
    (rec.run_dir / "report.json").write_text(json.dumps(report, indent=2))
    rec.close()
    print(f"\ntanh calibration PASS: {ok_all} ({report['seconds']}s)")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
