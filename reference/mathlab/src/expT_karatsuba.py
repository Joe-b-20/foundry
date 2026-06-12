"""
expT_karatsuba.py — discover KARATSUBA (sub-quadratic integer multiplication) from OUTCOME ALONE.

The project discovered schoolbook long-multiplication (O(n^2), digit-serial) many times, and STRASSEN for
MATRIX multiplication (7 mults, via the bilinear tensor-rank framing). The missing piece is KARATSUBA --
fast INTEGER multiplication (1960; it refuted Kolmogorov's conjectured O(n^2) lower bound). Karatsuba is the
INTEGER-mult analog of Strassen: multiplying two 2-limb numbers (x1*B+x0)(y1*B+y0) has product coefficients
  p0=x0y0, p1=x0y1+x1y0, p2=x1y1
-- a BILINEAR map = the polynomial-multiplication tensor T[k,i,j]=[i+j==k], shape (2n-1) x n x n. The number
of scalar MULTIPLICATIONS = the tensor RANK = the EFFICIENCY BUDGET. Naive = n^2 (=4 for n=2); Karatsuba = 3
(=2n-1). So: apply the SAME method that found Strassen (expN: differentiable CP decomposition + annealed
integer-lattice penalty -> round -> verify the EXACT bilinear identity) to the POLYNOMIAL-mult tensor, sweep R,
and see whether a multiplication-count budget discovers Karatsuba (R=3) and that R=2 is impossible.

Then GROUND it: build a RECURSIVE integer multiplier from the discovered decomposition and verify it multiplies
large random integers EXACTLY (length-gen), counting multiplications to confirm the O(n^log2(3))=O(n^1.585) scaling.

Run: python expT_karatsuba.py
"""
from __future__ import annotations
import argparse, math, time
import torch

import expN_matmul as XN                          # reuse CP+lattice-anneal core (sets float64 on import)


def poly_tensor(n):
    """Polynomial-multiplication tensor: p_k = sum_{i+j=k} x_i y_j. Shape (2n-1) x n x n."""
    T = torch.zeros(2 * n - 1, n, n)
    for i in range(n):
        for j in range(n):
            T[i + j, i, j] = 1.0
    return T


def search(T, R, restarts, steps, lam, lr, device, seed0=0):
    """Batched restarts at fixed rank R; round to integers and check the EXACT tensor identity (reuse expN)."""
    fit, U, V, W = XN.fit_batched(T, R, restarts, steps, lam, lr, seed0, device)
    mask = XN.exact_mask(U, V, W, T)
    best = fit.min().item()
    if mask.any():
        idx = int(mask.nonzero()[0])
        return best, (U[idx].round().long(), V[idx].round().long(), W[idx].round().long())
    keep = min(48, restarts); order = torch.argsort(fit)[:keep]
    fit2, U2, V2, W2 = XN.fit_batched(T, R, keep, steps, lam * 2.5, lr * 0.5, seed0 + 1, device, warm=(U[order], V[order], W[order]))
    mask2 = XN.exact_mask(U2, V2, W2, T)
    best = min(best, fit2.min().item())
    if mask2.any():
        idx = int(mask2.nonzero()[0])
        return best, (U2[idx].round().long(), V2[idx].round().long(), W2[idx].round().long())
    return best, None


def naive_decomp(n):
    pairs = [(a, b) for a in range(n) for b in range(n)]
    R = len(pairs)
    U = torch.zeros(R, n, dtype=torch.long); V = torch.zeros(R, n, dtype=torch.long); W = torch.zeros(R, 2 * n - 1, dtype=torch.long)
    for r, (a, b) in enumerate(pairs):
        U[r, a] = 1; V[r, b] = 1; W[r, a + b] = 1
    return U, V, W


def verify_limbs(U, V, W, n, trials=3000, hi=10 ** 6, seed=0):
    """Check the bilinear identity on random integer limb-vectors: W^T((Ux)*(Vy)) == true poly-product coefs."""
    g = torch.Generator().manual_seed(seed); bad = 0
    for _ in range(trials):
        x = torch.randint(-hi, hi + 1, (n,), generator=g).double()
        y = torch.randint(-hi, hi + 1, (n,), generator=g).double()
        p = W.double().t() @ ((U.double() @ x) * (V.double() @ y))
        ptrue = torch.zeros(2 * n - 1, dtype=torch.double)
        for i in range(n):
            for j in range(n):
                ptrue[i + j] += x[i] * y[j]
        if not torch.equal(p.round().long(), ptrue.long()):
            bad += 1
    return trials - bad, trials


def describe(U, V, W, n):
    def comb(coefs, sym):
        ts = []
        for k, c in enumerate(coefs.tolist()):
            if c == 0:
                continue
            ts.append(("+" if c > 0 else "-") + (f"{abs(c)}*" if abs(c) != 1 else "") + f"{sym}{k}")
        s = "".join(ts); return s[1:] if s.startswith("+") else s
    R = U.shape[0]
    print(f"    {R} products:")
    for r in range(R):
        print(f"      m{r} = ({comb(U[r], 'x')}) * ({comb(V[r], 'y')})")
    print("    recombination:")
    for c in range(2 * n - 1):
        ts = []
        for r in range(R):
            w = int(W[r, c])
            if w:
                ts.append(("+" if w > 0 else "-") + (f"{abs(w)}*" if abs(w) != 1 else "") + f"m{r}")
        s = "".join(ts); s = s[1:] if s.startswith("+") else s
        print(f"      p{c} = {s}")


# ---- recursive integer multiplier built FROM the discovered decomposition (U,V,W) ----
_MULS = [0]


def rec_mul(x, y, U, V, W, n, base=10, threshold=10):
    """Multiply integers x,y using the bilinear decomposition recursively. Counts base-case mults in _MULS."""
    if abs(x) < threshold or abs(y) < threshold:
        _MULS[0] += 1
        return x * y
    sx, sy = (1 if x >= 0 else -1), (1 if y >= 0 else -1)
    ax, ay = abs(x), abs(y)
    D = max(len(str(ax)), len(str(ay)))
    w = (D + n - 1) // n
    Bb = base ** w
    X = [(ax // Bb ** i) % Bb for i in range(n)]
    Y = [(ay // Bb ** i) % Bb for i in range(n)]
    R = U.shape[0]
    P = [0] * (2 * n - 1)
    for r in range(R):
        Ar = sum(int(U[r, i]) * X[i] for i in range(n))
        Br = sum(int(V[r, j]) * Y[j] for j in range(n))
        Mr = rec_mul(Ar, Br, U, V, W, n, base, threshold)
        for c in range(2 * n - 1):
            wc = int(W[r, c])
            if wc:
                P[c] += wc * Mr
    res = sum(P[c] * Bb ** c for c in range(2 * n - 1))
    return sx * sy * res


def verify_recursive(U, V, W, n, trials=400, seed=0):
    import random
    rng = random.Random(seed); bad = 0
    for _ in range(trials):
        dx = rng.randint(1, 40); dy = rng.randint(1, 40)
        x = rng.randint(10 ** (dx - 1), 10 ** dx - 1)
        y = rng.randint(10 ** (dy - 1), 10 ** dy - 1)
        if rec_mul(x, y, U, V, W, n) != x * y:
            bad += 1
    return trials - bad, trials


def mult_scaling(U, V, W, n, digit_sizes):
    print("    multiplication count vs operand size (single-digit base-case mults):")
    prev = None
    for D in digit_sizes:
        import random
        rng = random.Random(D)
        x = rng.randint(10 ** (D - 1), 10 ** D - 1); y = rng.randint(10 ** (D - 1), 10 ** D - 1)
        _MULS[0] = 0
        got = rec_mul(x, y, U, V, W, n)
        ok = (got == x * y)
        exp = ""
        if prev is not None:
            exp = f"  exponent~{math.log(_MULS[0]/prev[1]) / math.log(D/prev[0]):.3f}"
        print(f"      D={D:4d} digits: {_MULS[0]:8d} mults  ok={ok}{exp}")
        prev = (D, _MULS[0])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2, help="limbs per operand (2=Karatsuba, 3=Toom-3)")
    ap.add_argument("--restarts", type=int, default=256)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lam", type=float, default=0.4)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    n = args.n
    T = poly_tensor(n); naive = n * n; opt = 2 * n - 1
    print(f"device={args.device}  {n}-limb integer/polynomial multiplication  T{tuple(T.shape)}  "
          f"naive #mults={naive}  known optimum (Toom/Karatsuba)={opt}")

    # sanity: naive decomposition reconstructs T exactly and multiplies recursively
    Un, Vn, Wn = naive_decomp(n)
    assert torch.equal(torch.einsum('ra,rb,rc->cab', Un.double(), Vn.double(), Wn.double()).round().long(), T.long())
    print("naive decomposition reconstructs the tensor exactly (sanity).")

    print(f"\n  SWEEP rank R from naive={naive} down past the optimum (multiplication budget):")
    found = {}; t0 = time.time()
    for R in range(naive, max(1, opt - 2) - 1, -1):
        best, exact = search(T, R, args.restarts, args.steps, args.lam, args.lr, args.device)
        found[R] = exact
        tag = "EXACT integer decomposition FOUND" if exact else "no exact (this few mults not achieved)"
        print(f"    R={R}: best residual {best:.2e}  -> {tag}   [{time.time()-t0:.0f}s]")
        if exact:
            okl, tot = verify_limbs(*exact, n)
            coeffs = sorted(set(exact[0].unique().tolist() + exact[1].unique().tolist() + exact[2].unique().tolist()))
            print(f"        bilinear identity verified on {okl}/{tot} random integer limb-vectors; coeffs {coeffs}")

    exrs = [R for R, ex in found.items() if ex is not None]
    print(f"\n  === RESULT for {n}-limb multiplication ===")
    if exrs:
        mn = min(exrs)
        print(f"  fewest multiplications with an EXACT verified algorithm: R={mn}  (naive={naive}, known optimum={opt})")
        if mn < naive:
            print(f"  => SUB-NAIVE by {naive - mn}  " + ("== KARATSUBA" if n == 2 and mn == 3 else f"(= {opt}-mult scheme)" if mn == opt else ""))
        print()
        describe(*found[mn], n)
        print("\n  GROUNDING: recursive integer multiplier built from the discovered decomposition:")
        okr, totr = verify_recursive(*found[mn], n, trials=400)
        print(f"    multiplies {okr}/{totr} random integer pairs (up to ~40 digits each) EXACTLY")
        mult_scaling(*found[mn], n, [4, 8, 16, 32, 64, 128])
        print("\n  CONTRAST: the naive n^2-mult schoolbook decomposition (same recursive multiplier):")
        mult_scaling(Un, Vn, Wn, n, [4, 8, 16, 32, 64, 128])
        print(f"    => discovered R={mn} gives exponent ~log({mn})/log({n})={math.log(mn)/math.log(n):.3f} "
              f"vs naive log({naive})/log({n})={math.log(naive)/math.log(n):.3f} (=2).")
    else:
        print(f"  no exact sub-naive decomposition found.")
