"""Domain pack: gelu(x) = x * Phi(x), float32 — the most ML-load-bearing
activation (every modern transformer). Phi is the standard-normal CDF,
gelu(x) = 0.5 x (1 + erf(x/sqrt2)). gelu -> x as x->+inf and -> 0 as
x->-inf (with a small negative dip near x=-0.75), so a PURE rational
(equal tail slopes) cannot match it; the right structure is x * rational
approximating Phi (built by FloatProgMold.build_gelu).

Scope: SIGNED — all float32 with |x| in [2^-4, 2^3) = [0.0625, 8), both
signs (~185M values). Exhaustive verification, max ABSOLUTE error vs the
float64 reference (gelu crosses 0 and is small/negative near it ->
absolute metric). Reference oracle: scipy.special.erf behind this wrapper.
"""

import numpy as np

try:
    from scipy.special import erf as _erf_vec
except ImportError:
    import math
    _erf_vec = np.vectorize(math.erf)

from domains.rsqrt import RsqrtPack

_SQRT2 = np.sqrt(2.0)


class GeluPack(RsqrtPack):
    name = "gelu"
    err_kind = "abs"

    def __init__(self, lo_exp=-4, hi_exp=3, **kw):
        super().__init__(lo_exp=lo_exp, hi_exp=hi_exp, signed=True, **kw)
        self.cost_rules = ("primary: max ABSOLUTE error vs float64 "
                           "0.5 x (1+erf(x/sqrt2)) over the signed declared "
                           "scope (exhaustive); secondary: op count, dl")

    @staticmethod
    def _truth(bits_u32):
        x = bits_u32.view(np.float32).astype(np.float64)
        return 0.5 * x * (1.0 + _erf_vec(x / _SQRT2))

    @staticmethod
    def phi(x):
        """Phi(x) = gelu(x)/x for x!=0 (and 0.5 at 0): the gating function a
        rational approximates; the program computes x * R(x)."""
        return 0.5 * (1.0 + _erf_vec(np.asarray(x, float) / _SQRT2))


if __name__ == "__main__":
    import struct as _s
    from engine.molds_float import OPS_DIV, OPS_F, FloatProgMold

    def fb(v):
        return _s.unpack("<I", _s.pack("<f", v))[0]
    pack = GeluPack(lo_exp=-2, hi_exp=2)
    mold = FloatProgMold(n_const=8, max_len=20, ops=OPS_F + OPS_DIV)
    # crude x*[2/2] just to exercise the gelu structure + verify
    cand = mold.build_gelu(2, 2, [0.5, 0.4, 0.0], [0.0, 0.3])
    ok, det = pack.verify_trusted(mold, cand)
    assert ok and np.isfinite(det["max_rel_err"]), det
    # reference spot checks: gelu(0)=0, gelu(1)=0.8413, gelu(-1)=-0.1587
    g = pack._truth(np.array([fb(0.0), fb(1.0), fb(-1.0)], dtype=np.uint32))
    assert abs(g[0]) < 1e-9 and abs(g[1] - 0.84134) < 1e-3 \
        and abs(g[2] + 0.15866) < 1e-3, g
    print(f"gelu pack ok: scope {pack.scope_size:,} signed; x*[2/2] builds "
          f"and verifies; gelu(1)={g[1]:.5f}")
