"""Rational minimax fitting from outcome data — the coupled-coefficient
optimizer the saturating family needs (coordinate descent can't fit
P/Q jointly; this is the linearized rational least-squares + Lawson IRLS
that the tanh calibration foreshadowed for polynomials).

numpy is the search-side oracle (allowed; not core). Fits R(x)=P(x)/Q(x),
P degree p, Q degree q with Q(0)=1 (fixes the shared scale), to minimize
ABSOLUTE error on the given (xs, fs) samples. Returns real coefficients;
the caller float32-rounds and EXHAUSTIVELY verifies — this is a search
heuristic, never the claim.
"""

import numpy as np


def rational_fit(xs, fs, p, q, iters=80, weight0=None):
    """(A ridge-regularized variant was tried 2026-06-13 to "rescue" a
    [3/3] sigmoid fit that exhaustive verification flagged at 8.0 — but that
    failure was a max_len truncation BUG, not a float32-coefficient problem;
    the naive fit float32-rounds fine. The ridge machinery was removed as
    speculative — re-add WITH A REAL TEST if a genuinely ill-conditioned
    case appears.)"""
    xs = np.asarray(xs, dtype=np.float64)
    fs = np.asarray(fs, dtype=np.float64)
    Xp = np.vander(xs, p + 1, increasing=True)         # [1, x, ..., x^p]
    Xq = np.vander(xs, q + 1, increasing=True)[:, 1:]  # [x, ..., x^q]
    W = np.ones(len(xs)) if weight0 is None else np.asarray(weight0, float)
    a = b = None
    best = None
    for _ in range(iters):
        # minimize W * (P - f*Q),  Q = 1 + Xq@b  ->  Xp@a - f*(Xq@b) = f
        A = np.concatenate([Xp, -(fs[:, None] * Xq)], axis=1)
        sw = np.sqrt(W)[:, None]
        sol, *_ = np.linalg.lstsq(A * sw, fs * np.sqrt(W), rcond=None)
        a, b = sol[: p + 1], sol[p + 1:]
        Q = 1.0 + Xq @ b
        with np.errstate(all="ignore"):
            R = (Xp @ a) / Q
        e = np.abs(R - fs)
        if np.all(np.isfinite(e)):
            m = float(e.max())
            if best is None or m < best[0]:
                best = (m, a.copy(), b.copy())
        # Lawson reweight toward the minimax (abs) solution
        W = W * (e + 1e-12)
        s = W.sum()
        if not np.isfinite(s) or s == 0:
            break
        W /= s
    _, a, b = best if best is not None else (None, a, b)
    return a, b


if __name__ == "__main__":
    xs = np.linspace(-8, 8, 4000)
    fs = 1.0 / (1.0 + np.exp(-xs))
    a, b = rational_fit(xs, fs, 2, 2)
    Q = 1.0 + sum(b[k] * xs ** (k + 1) for k in range(len(b)))
    P = sum(a[j] * xs ** j for j in range(len(a)))
    err = np.max(np.abs(P / Q - fs))
    # a [2/2] rational should fit sigmoid well below a deg-2 polynomial
    assert err < 0.05, err
    print(f"ratfit ok: [2/2] sigmoid real-coeff max abs err = {err:.4e}")
