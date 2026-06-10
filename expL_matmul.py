"""
expL_matmul.py — discover matrix multiplication from OUTCOME ALONE; WHICH algorithm (naive vs Strassen)?

For matmul the meaningful "algorithm" axis is the SCALAR-MULTIPLICATION COUNT. 2x2 naive uses 8 products;
Strassen (1969) uses 7. We frame discovery as a bilinear RANK-R decomposition of the 2x2 matmul tensor T:
  c_k = sum_r W[r,k] * (sum_i U[r,i] a_i) * (sum_j V[r,j] b_j)
computes A*B EXACTLY iff sum_r U[r,i] V[r,j] W[r,k] == T[k,i,j] (an algebraic identity -> holds for ALL
matrices). R = the number of scalar multiplications (the EFFICIENCY BUDGET, the analog of the GCD step
budget that selected Euclid). The "outcome" is the tensor reconstruction error = "does it compute A*B".

Method (NOT AlphaTensor's RL/transformer): differentiable CP decomposition by gradient, with an annealed
INTEGER-LATTICE penalty (x^3 - x)^2 that pulls coefficients onto {-1,0,1} (Strassen's coefficients), so the
converged solution ROUNDS to an EXACT integer decomposition -> verified by the exact tensor identity + on
random integer matrices. Sweep R in {6,7,8}.

Run: python expL_matmul.py
"""
from __future__ import annotations
import argparse, itertools
import torch

import expA_mealy as E
torch.set_default_dtype(torch.float64)       # need precision to round cleanly to integers
# This tensor is tiny (4x4x4); fp64 on a consumer GPU is far SLOWER than CPU here, so default to CPU.
DEVICE = "cpu"


def matmul_tensor(n=2):
    """T[k,i,j] for n x n matmul with row-major flattening of A,B,C into length n*n vectors."""
    d = n * n
    T = torch.zeros(d, d, d)
    for i0 in range(n):
        for j0 in range(n):
            for k0 in range(n):
                c = i0 * n + j0            # C[i0,j0]
                a = i0 * n + k0            # A[i0,k0]
                b = k0 * n + j0            # B[k0,j0]
                T[c, a, b] = 1.0
    return T


def reconstruct(U, V, W):
    # U,V,W: (R, d). T_hat[k,i,j] = sum_r U[r,i] V[r,j] W[r,k]
    return torch.einsum('ri,rj,rk->kij', U, V, W)


def lattice_pen(X):
    return ((X ** 3 - X) ** 2).sum()         # 0 exactly on {-1,0,1}


def fit_rank(T, R, steps=6000, lam_max=0.4, seed=0, lr=0.03):
    d = T.shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed)
    U = (torch.randn(R, d, generator=g) * 0.5).to(DEVICE).requires_grad_(True)
    V = (torch.randn(R, d, generator=g) * 0.5).to(DEVICE).requires_grad_(True)
    W = (torch.randn(R, d, generator=g) * 0.5).to(DEVICE).requires_grad_(True)
    T = T.to(DEVICE)
    opt = torch.optim.Adam([U, V, W], lr=lr)
    for s in range(steps):
        lam = lam_max * (s / steps) ** 2                 # anneal integer-lattice pressure up
        That = reconstruct(U, V, W)
        fit = ((That - T) ** 2).sum()
        pen = lattice_pen(U) + lattice_pen(V) + lattice_pen(W)
        loss = fit + lam * pen
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        fit = ((reconstruct(U, V, W) - T) ** 2).sum().item()
    return fit, U.detach(), V.detach(), W.detach()


def exact_check_integer(U, V, W, T):
    """Round to nearest integer; if the tensor identity holds EXACTLY (integer arithmetic), it's an exact
    rank-R algorithm. Returns (ok, Ui,Vi,Wi) on CPU."""
    Ui, Vi, Wi = U.round().long().cpu(), V.round().long().cpu(), W.round().long().cpu()
    Tc = T.long().cpu()
    That = torch.einsum('ri,rj,rk->kij', Ui.double(), Vi.double(), Wi.double())
    ok = torch.equal(That.round().long(), Tc)
    return ok, Ui, Vi, Wi


def verify_on_matrices(Ui, Vi, Wi, n=2, trials=2000, hi=10 ** 6, seed=0):
    """Exact check that the decomposition computes A*B for random integer matrices (any magnitude)."""
    g = torch.Generator().manual_seed(seed)
    d = n * n
    bad = 0
    for _ in range(trials):
        A = torch.randint(-hi, hi + 1, (d,), generator=g)
        B = torch.randint(-hi, hi + 1, (d,), generator=g)
        m = (Ui.double() @ A.double()) * (Vi.double() @ B.double())     # R products
        c = (Wi.double().t() @ m)                                       # combine -> C (length d)
        # true C
        Am = A.reshape(n, n); Bm = B.reshape(n, n); Ctrue = (Am @ Bm).reshape(-1)
        if not torch.equal(c.round().long(), Ctrue.long()):
            bad += 1
    return trials - bad, trials


def describe(Ui, Vi, Wi, n=2):
    """Print the discovered products as combinations of A,B entries + the C recombination."""
    names_a = [f"a{i//n+1}{i%n+1}" for i in range(n * n)]
    names_b = [f"b{i//n+1}{i%n+1}" for i in range(n * n)]
    def comb(coefs, names):
        ts = []
        for c, nm in zip(coefs.tolist(), names):
            if c == 0: continue
            s = ("+" if c > 0 else "-") + (f"{abs(c)}*" if abs(c) != 1 else "") + nm
            ts.append(s)
        out = "".join(ts) if ts else "0"
        return out[1:] if out.startswith("+") else out
    R = Ui.shape[0]
    print(f"    {R} products:")
    for r in range(R):
        print(f"      m{r+1} = ({comb(Ui[r], names_a)}) * ({comb(Vi[r], names_b)})")
    print("    recombination:")
    for k in range(n * n):
        ts = []
        for r in range(R):
            c = int(Wi[r, k])
            if c == 0: continue
            s = ("+" if c > 0 else "-") + (f"{abs(c)}*" if abs(c) != 1 else "") + f"m{r+1}"
            ts.append(s)
        out = "".join(ts); out = out[1:] if out.startswith("+") else out
        print(f"      c{k//n+1}{k%n+1} = {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--restarts", type=int, default=40)
    ap.add_argument("--steps", type=int, default=6000)
    args = ap.parse_args()
    T = matmul_tensor(args.n)
    naive = args.n ** 3
    print(f"device={DEVICE}  {args.n}x{args.n} matmul tensor: shape {tuple(T.shape)}, naive #mults = {naive}")
    print("Discovering low-rank (= few-multiplication) algorithms from the OUTCOME (compute A*B) alone.\n")

    found = {}
    for R in range(naive - 2, naive + 1):       # e.g. 2x2 -> R in {6,7,8}
        best_fit = 1e9; best = None; exact = None
        for s in range(args.restarts):
            fit, U, V, W = fit_rank(T, R, steps=args.steps, seed=s)
            if fit < best_fit:
                best_fit = fit; best = (U, V, W)
            ok, Ui, Vi, Wi = exact_check_integer(U, V, W, T)
            if ok:
                exact = (Ui, Vi, Wi)
                break
        found[R] = (best_fit, exact)
        tag = "EXACT integer decomposition FOUND" if exact else "no exact integer decomposition"
        print(f"  R={R} (#mults={R}): best tensor-fit residual {best_fit:.2e}   -> {tag}")
        if exact:
            nz = int((exact[0] != 0).sum() + (exact[1] != 0).sum() + (exact[2] != 0).sum())
            okm, tot = verify_on_matrices(*exact, n=args.n)
            print(f"      exact on {okm}/{tot} random integer matrices (entries up to 1e6); coeff set "
                  f"{sorted(set(exact[0].unique().tolist()+exact[1].unique().tolist()+exact[2].unique().tolist()))}")

    print("\n  === WHICH ALGORITHM ===")
    min_exact_R = min([R for R, (_, ex) in found.items() if ex is not None], default=None)
    if min_exact_R is not None:
        print(f"  Fewest multiplications with an EXACT algorithm: R = {min_exact_R}  (naive = {naive}).")
        if min_exact_R < naive:
            print(f"  => discovered a SUB-NAIVE algorithm: {min_exact_R} multiplications (Strassen's bound for 2x2 is 7).")
        Ui, Vi, Wi = found[min_exact_R][1]
        describe(Ui, Vi, Wi, n=args.n)
    # show the rank boundary: which R admits an exact algorithm
    print("\n  rank boundary (best residual per R; exact = an integer decomposition was verified):")
    for R in sorted(found):
        resid, ex = found[R]
        achievable = (ex is not None) or (resid < 1e-8)
        rstr = "0 (exact integer)" if ex is not None else f"{resid:.2e}"
        print(f"    R={R}: residual {rstr}" + ("  (ACHIEVABLE)" if achievable else "  (NOT achievable -> this few mults is IMPOSSIBLE)"))
