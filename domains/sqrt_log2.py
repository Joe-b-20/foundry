"""Domain packs: sqrt(x) and log2(x), float32 — the exponent-structured
family, where the tanh-null routing said the bit tricks live.

- sqrt has the same scaling symmetry as rsqrt (f(4x) = 2 f(x)); metric =
  max RELATIVE error, scope all float32 in [2^-8, 2^8), exhaustive.
- log2 is the symmetry itself: the float bit pattern, read as an integer,
  is a piecewise-linear approximation of log2 (Blinn 1997, "Floating-
  Point Tricks", IEEE CG&A 17(4)). Metric = max ABSOLUTE error — the
  shift-invariant choice (log2(2x) = log2(x) + 1, and relative error
  blows up at the in-scope zero crossing log2(1) = 0).

Both reuse the rsqrt pack machinery (rule of three: a shared base class
is now justified — RsqrtPack IS that base; renaming it is cosmetic debt,
noted, not urgent).
"""

import numpy as np

from domains.rsqrt import RsqrtPack


class SqrtPack(RsqrtPack):
    name = "sqrt"
    err_kind = "rel"

    @staticmethod
    def _truth(bits_u32):
        x = bits_u32.view(np.float32).astype(np.float64)
        return np.sqrt(x)


class Log2Pack(RsqrtPack):
    name = "log2"
    err_kind = "abs"

    def __init__(self, **kw):
        super().__init__(**kw)
        self.cost_rules = ("primary: max ABSOLUTE error vs float64 "
                           "numpy.log2 over the declared scope "
                           "(exhaustive); secondary: op count, dl")

    @staticmethod
    def _truth(bits_u32):
        x = bits_u32.view(np.float32).astype(np.float64)
        return np.log2(x)


if __name__ == "__main__":
    import struct as _s
    from engine.molds_float import OPS_CVT, OPS_F, OPS_I, FloatProgMold
    from engine import runner as core_runner

    def fb(v):
        return _s.unpack("<I", _s.pack("<f", v))[0]

    # the Blinn-family affine log2 trick: U2F(bits) * 2^-23 - B
    mold = FloatProgMold(n_const=2, max_len=4, ops=OPS_F + OPS_I + OPS_CVT)
    T = 1 + mold.N_CONST
    trick = ((("U2F", T, 0, 0), ("FMUL", T, T, 1), ("FADD", 0, T, 2)),
             (fb(2.0 ** -23), fb(-126.94)))
    pack = Log2Pack()
    ok, det = pack.verify_trusted(mold, mold.tidy(trick))
    assert ok and det["max_rel_err"] < 0.1, det     # field name is legacy
    # bit-exactness of the new CVT ops across both paths
    xs = np.array([fb(v) for v in (0.011, 0.25, 1.0, 3.7, 200.0)],
                  dtype=np.uint32)
    vec = mold.npfunc(mold.tidy(trick))(xs).view(np.uint32)
    prog = mold.pour(mold.tidy(trick))
    for xb, vb in zip(xs, vec):
        out, _ = core_runner.run(prog, [int(xb)])
        assert (out[0] & 0xFFFFFFFF) == int(vb)
    sq = SqrtPack(lo_exp=-2, hi_exp=2)
    assert abs(float(sq._truth(np.array([fb(4.0)], dtype=np.uint32))[0])
               - 2.0) < 1e-12
    print(f"sqrt+log2 packs ok: affine log2 trick exhaustive max ABS err "
          f"= {det['max_rel_err']:.4f} (3 ops, folklore constants)")
