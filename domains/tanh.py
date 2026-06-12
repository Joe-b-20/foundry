"""Domain pack: tanh(x), float32 (portfolio domain #1, function #2 —
the polynomial/rational-shaped counterpart to rsqrt's bit-trick shape).

Reuses the rsqrt pack machinery wholesale (same scope/exhaustive/cross-
check pattern; generalization waits for the third function per the
no-abstraction-before-demand rule). Declared scope: all float32 in
[2^-2, 2^3) = [0.25, 8) — the knee region where tanh is neither ~x nor
~1. Metric: max relative error vs the float64 numpy.tanh reference
(reference noise ~1e-16, negligible at approximation error levels).

The polynomial corner of this domain is THEORY-SOLVED: engine/remez.py
computes the minimax polynomial and its de la Vallee Poussin bracket —
a PROVEN floor for every polynomial of the given degree. The shelf
carries those floors; the recognizer's CONTRADICTS-PROVEN-BOUND logic
applies beneath them (for polynomial-class candidates). Headroom for the
hunt lives OUTSIDE the polynomial class: bit/float hybrids.
"""

import numpy as np

from domains.rsqrt import RsqrtPack


class TanhPack(RsqrtPack):
    name = "tanh"

    def __init__(self, lo_exp=-2, hi_exp=3, **kw):
        super().__init__(lo_exp=lo_exp, hi_exp=hi_exp, **kw)
        self.cost_rules = ("primary: max relative error vs float64 "
                           "numpy.tanh over the declared scope "
                           "(exhaustive); secondary: op count, dl; "
                           "op budget <= 14; no FDIV")

    @staticmethod
    def _truth(bits_u32):
        x = bits_u32.view(np.float32).astype(np.float64)
        return np.tanh(x)


if __name__ == "__main__":
    from engine.molds_float import FloatProgMold
    import struct as _s

    def fb(v):
        return _s.unpack("<I", _s.pack("<f", v))[0]
    # a crude degree-1 sanity candidate: 0.07x + 0.65 — terrible but finite
    mold = FloatProgMold(n_const=2, max_len=4)
    cand = ((("FMUL", mold.N_SLOTS - 1, 0, 1),
             ("FADD", 0, mold.N_SLOTS - 1, 2)),
            (fb(0.07), fb(0.65)))
    pack = TanhPack()
    sc, cost = pack.gate1(mold, mold.tidy(cand))
    assert np.isfinite(sc[0]) and cost["ops"] == 2
    ok, det = pack.verify_trusted(mold, cand)
    assert ok and 0.01 < det["max_rel_err"] < 2.0, det
    print(f"tanh pack ok: scope {pack.scope_size:,} float32, crude line "
          f"E={det['max_rel_err']:.3f} (terrible, as expected)")
