"""Domain pack: sin(x), float32 — the first PERIODIC / oscillating target
(every prior domain was monotonic or saturating). Periodic functions are
where ARGUMENT REDUCTION lives: no fixed-degree polynomial tracks many
oscillations over a wide range, but reduce x to one period first and a low
-degree polynomial suffices. This is the technique, not a rational.

Scope: SIGNED — all float32 with |x| in [2^-6, 2^4) = magnitude in
[0.0156, 16), both signs (~167M values). 16 > 4*pi, so the scope spans ~5
periods — a single polynomial is hopeless, reduction is essential.
Exhaustive verification, max ABSOLUTE error vs float64 numpy.sin (sin in
[-1,1], crosses 0 -> absolute metric).

Reduction note: x' = x - round(x/2pi)*2pi via the magic-add round trick
((t+1.5*2^23)-1.5*2^23, exact rounding for |t|<2^22; here |x/2pi|<2.6).
Single-constant 2pi reduction loses ~|k|*ulp(2pi) ~ 1e-6 over this scope —
negligible vs a low-degree poly's error. Wider scopes would need a
Cody-Waite split 2pi = hi+lo (logged, not needed here).
"""

import numpy as np

from domains.rsqrt import RsqrtPack


class SinPack(RsqrtPack):
    name = "sin"
    err_kind = "abs"

    def __init__(self, lo_exp=-6, hi_exp=4, **kw):
        super().__init__(lo_exp=lo_exp, hi_exp=hi_exp, signed=True, **kw)
        self.cost_rules = ("primary: max ABSOLUTE error vs float64 numpy.sin "
                           "over the signed declared scope (exhaustive); "
                           "secondary: op count, dl")

    @staticmethod
    def _truth(bits_u32):
        x = bits_u32.view(np.float32).astype(np.float64)
        return np.sin(x)


if __name__ == "__main__":
    import struct as _s
    from engine.molds_float import OPS_F, FloatProgMold

    def fb(v):
        return _s.unpack("<I", _s.pack("<f", v))[0]
    mold = FloatProgMold(n_const=8, max_len=28, ops=OPS_F)
    # magic-add argument reduction proof of concept: xr = x - round(x/2pi)*2pi
    INV2PI, MAGIC, TWO_PI = 1.0 / (2 * np.pi), 1.5 * 2 ** 23, 2 * np.pi
    T = 1 + mold.N_CONST                       # temps from here
    # c0=inv2pi c1=magic c2=two_pi ; then a small odd poly c3=c1coef...
    # here just verify reduction puts xr in [-pi,pi] then sin~xr (deg1) is
    # crude but finite — quality irrelevant, this only checks the pipeline
    skel = ((("FMUL", T, 0, 1),               # t = x*inv2pi
             ("FADD", T, T, 2),               # t += magic
             ("FSUB", T, T, 2),               # t -= magic  => round(x/2pi)
             ("FMUL", T, T, 3),               # t = k*2pi
             ("FSUB", 0, 0, T)),              # x' = x - k*2pi  (output=xr)
            (fb(INV2PI), fb(MAGIC), fb(TWO_PI), 0, 0, 0, 0, 0))
    cand = mold.tidy(skel)
    xs = np.array([fb(v) for v in (0.5, 3.0, 7.0, -7.0, 10.0)], dtype=np.uint32)
    xr = mold.npfunc(cand)(xs).astype(np.float64)
    assert np.all(np.abs(xr) <= np.pi + 1e-3), xr     # reduced to [-pi,pi]
    truex = xs.view(np.float32).astype(np.float64)
    assert np.allclose(np.sin(xr), np.sin(truex), atol=1e-5)  # sin(xr)=sin(x)
    print(f"sin pack ok: scope {SinPack().scope_size:,}; magic-add reduction "
          f"maps to [-pi,pi] and preserves sin (max|xr|={np.max(np.abs(xr)):.3f})")
