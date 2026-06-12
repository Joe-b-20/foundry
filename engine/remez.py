"""Remez exchange: minimax polynomial approximation with the
de la Vallee Poussin bracket as the certificate. mpmath stays behind this
owned wrapper (same policy as engine/numeric.py).

For continuous f on [a,b] and degree d, the minimax polynomial
equioscillates at d+2 points (Chebyshev). The exchange iterates: solve
p(x_i) + (-1)^i E = f(x_i) for coefficients and E, move the nodes to the
error's alternating extrema, repeat. On convergence the bracket
    bound_low  = min_i |err(x_i)|   (de la Vallee Poussin THEOREM:
                  every degree-<=d polynomial has max error >= this)
    bound_high = max |err| on a dense grid (what THIS polynomial achieves)
pins the true minimax error between two nearly-equal numbers. The
certificate is numeric at the stated working precision — bracket honesty,
not formal proof.
"""

import mpmath as mp


def polyval(coeffs, x):
    acc = mp.mpf(0)
    for c in reversed(coeffs):
        acc = acc * x + c
    return acc


def remez(f, a, b, deg, dps=40, iters=40, grid_n=4096, weight=None,
          grid_kind="lin"):
    """weight=None minimizes ABSOLUTE error; weight=f (positive on [a,b])
    minimizes RELATIVE error — the generalized equioscillation theorem
    covers positive-weighted Chebyshev systems, so the bracket remains a
    proven floor for the weighted error. grid_kind="log" places the
    extrema-search grid geometrically — needed on wide ranges (e.g. log2
    on [2^-8, 256]) where the error's features cluster near the left edge
    and a linear grid misses them entirely."""
    with mp.workdps(dps):
        a, b = mp.mpf(a), mp.mpf(b)
        w = weight or (lambda x: mp.mpf(1))
        n = deg + 2
        xs = [(a + b) / 2 - (b - a) / 2 * mp.cos(mp.pi * k / (n - 1))
              for k in range(n)]
        if grid_kind == "log":
            ratio = b / a
            grid = [a * ratio ** (mp.mpf(k) / grid_n)
                    for k in range(grid_n + 1)]
        else:
            grid = [a + (b - a) * k / grid_n for k in range(grid_n + 1)]
        coeffs, E, low, high = None, None, None, None
        for it in range(iters):
            rows = [[x ** j for j in range(deg + 1)]
                    + [mp.mpf((-1) ** i) * w(x)]
                    for i, x in enumerate(xs)]
            rhs = [f(x) for x in xs]
            sol = mp.lu_solve(mp.matrix(rows), mp.matrix(rhs))
            coeffs = [sol[j] for j in range(deg + 1)]
            E = abs(sol[deg + 1])
            errs = [(polyval(coeffs, x) - f(x)) / w(x) for x in grid]
            # alternating extrema: the max-|err| point of each sign run
            ext, i = [], 0
            while i <= grid_n:
                s = mp.sign(errs[i]) or 1
                best = i
                while i <= grid_n and (mp.sign(errs[i]) or s) == s:
                    if abs(errs[i]) > abs(errs[best]):
                        best = i
                    i += 1
                ext.append(best)
            while len(ext) > n:
                if len(ext) == n + 1:
                    ext.pop(0 if abs(errs[ext[0]]) < abs(errs[ext[-1]])
                            else -1)
                    continue
                k = min(range(len(ext)), key=lambda t: abs(errs[ext[t]]))
                if 0 < k < len(ext) - 1:
                    left, right = ext[k - 1], ext[k + 1]
                    d2 = k - 1 if abs(errs[left]) < abs(errs[right]) else k + 1
                    for idx in sorted((k, d2), reverse=True):
                        ext.pop(idx)
                else:
                    ext.pop(k)
            if len(ext) < n:
                break                      # degenerate run; keep last state
            xs = [grid[e] for e in ext]
            low = min(abs(errs[e]) for e in ext)
            high = max(abs(e) for e in errs)
            if high - low < mp.mpf("1e-10") * high:
                break
        return {"coeffs": coeffs, "E_solved": float(E),
                "bound_low": float(low), "bound_high": float(high),
                "alternation_points": len(ext), "iterations": it + 1,
                "dps": dps}


if __name__ == "__main__":
    r5 = remez(mp.tanh, "0.25", "8", 5)
    assert r5["alternation_points"] == 7, r5
    assert r5["bound_high"] / r5["bound_low"] < 1.001, \
        (r5["bound_low"], r5["bound_high"])
    r3 = remez(mp.tanh, "0.25", "8", 3)
    r7 = remez(mp.tanh, "0.25", "8", 7)
    assert r3["bound_low"] > r5["bound_low"] > r7["bound_low"]
    # weighted (RELATIVE-error) variant: floors for the pack's metric
    w5 = remez(mp.tanh, "0.25", "8", 5, weight=mp.tanh)
    assert w5["alternation_points"] == 7
    assert w5["bound_high"] / w5["bound_low"] < 1.001
    assert w5["bound_low"] > r5["bound_low"]      # relative floor is larger
    print("remez ok: tanh [0.25,8] abs brackets "
          f"deg3={r3['bound_high']:.3e} deg5={r5['bound_high']:.3e} "
          f"deg7={r7['bound_high']:.3e}; REL deg5={w5['bound_high']:.3e} "
          "(equioscillation verified)")
