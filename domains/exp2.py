"""Domain pack: exp2(x) = 2^x, float32 — log2's mirror in the exponent-
structured family. log2 READS the exponent field (integer-from-float, the
U2F trick); exp2 WRITES it (float-from-integer, the F2U trick). This is
the canonical Schraudolph 1999 construction ("On a Fast, Compact
Approximation of the Exponential Function", Neural Computation 11(4)).

Scope: SIGNED — all float32 with |x| in [2^-8, 2^3) = magnitude in
[0.0039, 8), both signs (~185M values). Exhaustive verification (this
domain's 0/1 principle), max RELATIVE error vs the float64 numpy.exp2
reference (multiplicative error is the natural choice for an exponential;
2^x > 0 everywhere so no zero-crossing).

Precision note (measured, not assumed): the trick's multiply-add lands
near 2^30, where float32 keeps ~24 of ~30 bits — the lost low bits are
the OUTPUT's low mantissa bits, a ~1e-5 relative quantization, negligible
against the trick's intrinsic ~3%. verify_trusted confirms exhaustively.
"""

import numpy as np

from domains.rsqrt import RsqrtPack


class Exp2Pack(RsqrtPack):
    name = "exp2"
    err_kind = "rel"

    def __init__(self, lo_exp=-8, hi_exp=3, **kw):
        super().__init__(lo_exp=lo_exp, hi_exp=hi_exp, signed=True, **kw)
        self.cost_rules = ("primary: max RELATIVE error vs float64 "
                           "numpy.exp2 over the signed declared scope "
                           "(exhaustive); secondary: op count, dl")

    @staticmethod
    def _truth(bits_u32):
        x = bits_u32.view(np.float32).astype(np.float64)
        return np.exp2(x)


if __name__ == "__main__":
    import struct as _s
    from engine.molds_float import OPS_CVT, OPS_F, OPS_I, FloatProgMold
    from engine import runner as core_runner

    def fb(v):
        return _s.unpack("<I", _s.pack("<f", v))[0]

    # Schraudolph-family exp2: t = x*2^23 + 127*2^23 (float), B = (uint)t,
    # output = reinterpret(B) as float32. 3 ops, F2U exercised.
    mold = FloatProgMold(n_const=2, max_len=4, ops=OPS_F + OPS_I + OPS_CVT)
    T = 1 + mold.N_CONST
    trick = ((("FMUL", T, 0, 1), ("FADD", T, T, 2), ("F2U", 0, T, 0)),
             (fb(2.0 ** 23), fb(127.0 * 2 ** 23)))
    pack = Exp2Pack(lo_exp=-2, hi_exp=2)            # small scope for sanity
    ok, det = pack.verify_trusted(mold, mold.tidy(trick))
    assert ok and det["max_rel_err"] < 0.10, det    # zero-correction ~6%
    # bit-exactness of F2U across both execution paths, both signs
    xs = np.array([fb(v) for v in (-3.7, -1.0, -0.3, 0.3, 1.0, 3.7)],
                  dtype=np.uint32)
    vec = mold.npfunc(mold.tidy(trick))(xs).view(np.uint32)
    prog = mold.pour(mold.tidy(trick))
    for xb, vb in zip(xs, vec):
        out, _ = core_runner.run(prog, [int(xb)])
        assert (out[0] & 0xFFFFFFFF) == int(vb), (hex(int(xb)),)
    print(f"exp2 pack ok: Schraudolph trick exhaustive max REL err "
          f"= {det['max_rel_err']:.4f} (3 ops, zero-correction bias; "
          f"F2U bit-exact both signs)")
