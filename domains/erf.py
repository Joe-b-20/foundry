"""Domain pack: erf(x), float32 — third saturating-family member (after
sigmoid, tanh). erf is the Gaussian error function, odd, saturating to
+-1; the building block for gelu (gelu(x) = 0.5 x (1 + erf(x/sqrt2))).

Scope: SIGNED — all float32 with |x| in [2^-4, 2^3) = magnitude in
[0.0625, 8), both signs (~185M values). erf(8) = 1 to float64. Exhaustive
verification, max ABSOLUTE error vs the float64 reference (erf crosses 0
at x=0 and is small near it -> absolute, not relative, is the metric).

Reference oracle: scipy.special.erf (vectorized, ~1e-15 accurate) behind
this pack wrapper — a rented oracle, like numpy, used only on the search/
verify side (never in the stdlib-only core runner). math.erf (stdlib) is
the scalar fallback.
"""

import numpy as np

try:
    from scipy.special import erf as _erf_vec
except ImportError:                                    # stdlib fallback
    import math
    _erf_vec = np.vectorize(math.erf)

from domains.rsqrt import RsqrtPack


class ErfPack(RsqrtPack):
    name = "erf"
    err_kind = "abs"

    def __init__(self, lo_exp=-4, hi_exp=3, **kw):
        super().__init__(lo_exp=lo_exp, hi_exp=hi_exp, signed=True, **kw)
        self.cost_rules = ("primary: max ABSOLUTE error vs float64 "
                           "scipy.special.erf over the signed declared "
                           "scope (exhaustive); secondary: op count, dl")

    @staticmethod
    def _truth(bits_u32):
        x = bits_u32.view(np.float32).astype(np.float64)
        return _erf_vec(x)


if __name__ == "__main__":
    import struct as _s
    from engine.molds_float import OPS_DIV, OPS_F, FloatProgMold

    def fb(v):
        return _s.unpack("<I", _s.pack("<f", v))[0]
    pack = ErfPack(lo_exp=-2, hi_exp=2)               # small sanity scope
    mold = FloatProgMold(n_const=8, max_len=20, ops=OPS_F + OPS_DIV)
    # crude [2/2] just to exercise the pipeline (quality irrelevant here)
    cand = mold.build_rational(2, 2, [0.0, 1.0, 0.0], [0.0, 0.2])
    ok, det = pack.verify_trusted(mold, cand)
    assert ok and np.isfinite(det["max_rel_err"]), det
    assert abs(float(pack._truth(np.array([fb(1.0)], dtype=np.uint32))[0])
               - 0.8427007929) < 1e-9
    print(f"erf pack ok: scope {pack.scope_size:,} signed float32; crude "
          f"[2/2] exhaustive max abs err = {det['max_rel_err']:.3f}")
