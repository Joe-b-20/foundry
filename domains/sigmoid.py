"""Domain pack: sigmoid(x) = 1/(1+e^-x), float32 — first member of the
SATURATING family (tanh/sigmoid/gelu/erf/softplus), the family the
tanh-null routing said needs RATIONALS (FDIV) or PIECEWISE, not just
polynomials. This pack exercises the new FDIV op (rationals).

Scope: SIGNED — all float32 with |x| in [2^-4, 2^3) = magnitude in
[0.0625, 8), both signs (~150M values). Exhaustive verification, max
ABSOLUTE error vs the float64 reference 1/(1+exp(-x)) (sigmoid in (0,1);
absolute error is the standard metric — relative error blows up as
sigmoid -> 0 on the left tail).

Why rationals should beat polynomials here: sigmoid saturates to a
horizontal asymptote (1 as x->+inf, 0 as x->-inf). A rational P/Q tends
to the ratio of leading coefficients = a constant, matching saturation
naturally; a polynomial diverges and must wiggle to stay flat across the
saturated tail. The hunt measures whether that intuition cashes out at
equal op budget, exhaustively, against the PROVEN polynomial floor.
"""

import numpy as np

from domains.rsqrt import RsqrtPack


class SigmoidPack(RsqrtPack):
    name = "sigmoid"
    err_kind = "abs"

    def __init__(self, lo_exp=-4, hi_exp=3, **kw):
        super().__init__(lo_exp=lo_exp, hi_exp=hi_exp, signed=True, **kw)
        self.cost_rules = ("primary: max ABSOLUTE error vs float64 "
                           "1/(1+exp(-x)) over the signed declared scope "
                           "(exhaustive); secondary: op count, dl")

    @staticmethod
    def _truth(bits_u32):
        x = bits_u32.view(np.float32).astype(np.float64)
        return 1.0 / (1.0 + np.exp(-x))


if __name__ == "__main__":
    import struct as _s
    from engine.molds_float import OPS_DIV, OPS_F, FloatProgMold
    from engine import runner as core_runner

    def fb(v):
        return _s.unpack("<I", _s.pack("<f", v))[0]

    # a crude [1/1] rational sanity: (0.25x + 0.5)/(0.16|.|... ) — just
    # checks FDIV runs and both paths agree; quality irrelevant here
    mold = FloatProgMold(n_const=4, max_len=6, ops=OPS_F + OPS_DIV)
    T = 1 + mold.N_CONST
    cand = ((("FMUL", T, 2, 0), ("FADD", T, T, 1),       # P = a1 x + a0
             ("FMUL", T + 1, 4, 0), ("FADD", T + 1, T + 1, 3),  # Q = b1 x + 1
             ("FDIV", 0, T, T + 1)),
            (fb(0.5), fb(0.25), fb(1.0), fb(0.1)))
    pack = SigmoidPack(lo_exp=-2, hi_exp=2)              # small sanity scope
    ok, det = pack.verify_trusted(mold, mold.tidy(cand))
    assert ok, det                       # FDIV path + cross-check must pass
    # explicit FDIV bit-exactness, both signs, incl. a near-zero denominator
    xs = np.array([fb(v) for v in (-3.0, -0.5, 0.5, 3.0, -0.0625)],
                  dtype=np.uint32)
    vec = mold.npfunc(mold.tidy(cand))(xs).view(np.uint32)
    prog = mold.pour(mold.tidy(cand))
    for xbi, vb in zip(xs, vec):
        out, _ = core_runner.run(prog, [int(xbi)])
        assert (out[0] & 0xFFFFFFFF) == int(vb), (hex(int(xbi)),)
    print(f"sigmoid pack ok: FDIV runs + bit-exact both paths; crude "
          f"[1/1] rational exhaustive max abs err = {det['max_rel_err']:.3f}")
