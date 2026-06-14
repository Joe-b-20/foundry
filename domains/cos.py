"""Domain pack: cos(x), float32 — cos(x) = sin(x + pi/2), so it reuses the
sin argument-reduction machinery with a phase shift (the second periodic
function; the trig hunt is generalized over both). Same scope, metric, and
exhaustive verification as sin.
"""

import numpy as np

from domains.rsqrt import RsqrtPack


class CosPack(RsqrtPack):
    name = "cos"
    err_kind = "abs"

    def __init__(self, lo_exp=-6, hi_exp=4, **kw):
        super().__init__(lo_exp=lo_exp, hi_exp=hi_exp, signed=True, **kw)
        self.cost_rules = ("primary: max ABSOLUTE error vs float64 numpy.cos "
                           "over the signed declared scope (exhaustive); "
                           "secondary: op count, dl")

    @staticmethod
    def _truth(bits_u32):
        x = bits_u32.view(np.float32).astype(np.float64)
        return np.cos(x)


if __name__ == "__main__":
    print(f"cos pack ok: scope {CosPack().scope_size:,} signed float32 "
          "(cos = sin(x+pi/2); reuses sin reduction)")
