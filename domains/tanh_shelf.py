"""Reference shelf for tanh(x) on [0.25, 8): Remez minimax polynomials.

Built by OUR Remez implementation (engine/remez.py) at 40 dps, each with
its de la Vallee Poussin bracket — a PROVEN floor for the whole degree
class (numeric-at-40-dps grade). Coefficients are then rounded to float32
and the resulting Horner program is exhaustively measured through the
pack (float32 rounding inflates the error above the real-model bracket;
fitting coefficients IN float32 directly — cf. Sollya's fpminimax — is
the stronger craft, noted as future work, not performed here).

Citation: Remez exchange algorithm (E. Remez 1934; standard reference:
Trefethen, "Approximation Theory and Approximation Practice"). The
implementation and brackets are self-verified by equioscillation.
"""

import struct

import mpmath as mp

from engine.remez import remez

DEGREES = (3, 5, 7)
_A, _B = "0.25", "8"


def _fb(v):
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def horner_skeleton(deg, mold):
    """((c_d x + c_{d-1})x + ...) + c0 over the mold's slot layout:
    consts c0..c_d in slots 1..deg+1, accumulator in the first temp,
    final FADD writes slot 0. 2*deg ops, no MOV needed."""
    T = 1 + mold.N_CONST              # first temp slot
    ins = [("FMUL", T, 1 + deg, 0)]   # acc = c_d * x
    for k in range(deg - 1, 0, -1):
        ins.append(("FADD", T, T, 1 + k))
        ins.append(("FMUL", T, T, 0))
    ins.append(("FADD", 0, T, 1))     # out = acc + c0
    return tuple(ins)


def build_shelf(pack, make_mold):
    """make_mold(n_const) -> FloatProgMold sized for the degree."""
    entries = []
    for deg in DEGREES:
        # WEIGHTED Remez (weight = tanh): minimizes RELATIVE error, the
        # pack's metric, so the bracket is the floor for the whole
        # degree-d polynomial class under exact arithmetic. (The executed
        # float32 Horner is a rounded evaluation, not an exact polynomial;
        # its measured error sits above the floor in practice — asserted
        # as sanity, not theorem.)
        r = remez(mp.tanh, _A, _B, deg, weight=mp.tanh)
        assert r["alternation_points"] == deg + 2, r
        mold = make_mold(deg + 1)
        cand = mold.tidy((horner_skeleton(deg, mold),
                          tuple(_fb(c) for c in r["coeffs"])))
        ok, det = pack.verify_trusted(mold, cand)
        assert ok, det
        e32 = det["max_rel_err"]
        assert e32 >= r["bound_low"] * 0.98, (e32, r)
        entries.append({
            "name": f"remez-deg{deg}-f32", "deg": deg, "cand": cand,
            "ops": 2 * deg, "max_rel_err_f32": e32,
            "rel_bracket_real_model": [r["bound_low"], r["bound_high"]],
            "citation": "weighted Remez exchange (our impl, equioscillation-"
                        f"verified at {r['dps']} dps); float32-rounded "
                        "coefficients",
        })
    assert (entries[0]["max_rel_err_f32"] > entries[1]["max_rel_err_f32"]
            > entries[2]["max_rel_err_f32"])
    return entries


if __name__ == "__main__":
    from domains.tanh import TanhPack
    from engine.molds_float import FloatProgMold
    pack = TanhPack()
    shelf = build_shelf(pack, lambda n: FloatProgMold(n_const=n, max_len=16))
    for e in shelf:
        lo, hi = e["rel_bracket_real_model"]
        print(f"  {e['name']}: ops={e['ops']} E_f32={e['max_rel_err_f32']:.4e} "
              f"rel-bracket=[{lo:.4e},{hi:.4e}]")
    print("tanh shelf ok: Remez floors computed and float32 entries verified")
