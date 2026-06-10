"""
expN_matmul.py — SCALE the matmul-discovery method (expL) past 2x2: rectangular <m,k,p>, bigger squares,
push the rank R down to the boundary. Same core as expL (differentiable CP decomposition of the matmul
tensor + annealed integer-lattice penalty -> round -> verify the EXACT bilinear identity), but:
  * rectangular tensor <m,k,p> (C = A[m,k] @ B[k,p]);
  * BATCHED restarts (B parallel random inits in ONE vectorized einsum+Adam step) so hundreds of restarts
    are cheap -> the optimizer gets many shots at the hard high-dim landscape;
  * a POLISH phase (re-anneal the best near-integer candidates at higher lattice pressure);
  * push R from naive down to find the MINIMUM rank with an EXACT integer decomposition = the scaling boundary.

rank R = scalar-multiplication count = the EFFICIENCY BUDGET. An exact integer (U,V,W) with
sum_r U[r,a] V[r,b] W[r,c] == T[c,a,b] computes A*B for ALL matrices (length-gen automatic), verified exactly.

Run: python expN_matmul.py --m 3 --k 3 --p 3 --Rmax 27 --Rmin 21 --restarts 256
"""
from __future__ import annotations
import argparse, time
import torch

torch.set_default_dtype(torch.float64)       # precision to round cleanly to integers


def matmul_tensor(m, k, p):
    """T[c,a,b] for C=A@B with A:m x k, B:k x p, C:m x p, row-major flattening.
       a = i*k+l (A[i,l]), b = l*p+j (B[l,j]), c = i*p+j (C[i,j])."""
    dA, dB, dC = m * k, k * p, m * p
    T = torch.zeros(dC, dA, dB)
    for i in range(m):
        for l in range(k):
            for j in range(p):
                T[i * p + j, i * k + l, l * p + j] = 1.0
    return T


def reconstruct(U, V, W):
    # batched: U(B,R,dA) V(B,R,dB) W(B,R,dC) -> That(B,dC,dA,dB)
    return torch.einsum('nra,nrb,nrc->ncab', U, V, W)


def lattice_pen(X):
    return ((X ** 3 - X) ** 2).sum(dim=(1, 2))     # per-batch; 0 exactly on {-1,0,1}


def fit_batched(T, R, B, steps, lam_max, lr, seed, device, init=0.5, warm=None):
    dC, dA, dB = T.shape
    g = torch.Generator(device="cpu").manual_seed(seed)
    if warm is None:
        U = (torch.randn(B, R, dA, generator=g) * init).to(device).requires_grad_(True)
        V = (torch.randn(B, R, dB, generator=g) * init).to(device).requires_grad_(True)
        W = (torch.randn(B, R, dC, generator=g) * init).to(device).requires_grad_(True)
    else:
        U = warm[0].clone().to(device).requires_grad_(True)
        V = warm[1].clone().to(device).requires_grad_(True)
        W = warm[2].clone().to(device).requires_grad_(True)
    Td = T.to(device).unsqueeze(0)
    opt = torch.optim.Adam([U, V, W], lr=lr)
    for s in range(steps):
        lam = lam_max * (s / steps) ** 2
        That = reconstruct(U, V, W)
        fit = ((That - Td) ** 2).sum(dim=(1, 2, 3))
        pen = lattice_pen(U) + lattice_pen(V) + lattice_pen(W)
        loss = (fit + lam * pen).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        fit = ((reconstruct(U, V, W) - Td) ** 2).sum(dim=(1, 2, 3))
    return fit.detach().cpu(), U.detach().cpu(), V.detach().cpu(), W.detach().cpu()


def exact_mask(U, V, W, T):
    """Per-restart: round to int, check the tensor identity EXACTLY. Returns bool mask (B,)."""
    Ui, Vi, Wi = U.round(), V.round(), W.round()
    That = torch.einsum('nra,nrb,nrc->ncab', Ui, Vi, Wi)
    return (That.round().long() == T.long().unsqueeze(0)).all(dim=(1, 2, 3))


def verify_on_matrices(Ui, Vi, Wi, m, k, p, trials=2000, hi=10 ** 6, seed=0):
    g = torch.Generator().manual_seed(seed)
    dA, dB, dC = m * k, k * p, m * p
    bad = 0
    for _ in range(trials):
        A = torch.randint(-hi, hi + 1, (dA,), generator=g).double()
        B = torch.randint(-hi, hi + 1, (dB,), generator=g).double()
        prod = (Ui.double() @ A) * (Vi.double() @ B)          # R products
        c = Wi.double().t() @ prod                             # combine -> C (length dC)
        Ctrue = (A.reshape(m, k) @ B.reshape(k, p)).reshape(-1)
        if not torch.equal(c.round().long(), Ctrue.long()):
            bad += 1
    return trials - bad, trials


def describe(Ui, Vi, Wi, m, k, p):
    na = [f"a{i//k+1}{i%k+1}" for i in range(m * k)]
    nb = [f"b{i//p+1}{i%p+1}" for i in range(k * p)]
    def comb(coefs, names):
        ts = []
        for c, nm in zip(coefs.tolist(), names):
            if c == 0: continue
            ts.append(("+" if c > 0 else "-") + (f"{abs(c)}*" if abs(c) != 1 else "") + nm)
        out = "".join(ts) or "0"
        return out[1:] if out.startswith("+") else out
    R = Ui.shape[0]
    print(f"    {R} products:")
    for r in range(R):
        print(f"      m{r+1} = ({comb(Ui[r], na)}) * ({comb(Vi[r], nb)})")
    print("    recombination:")
    for c in range(m * p):
        ts = []
        for r in range(R):
            w = int(Wi[r, c])
            if w == 0: continue
            ts.append(("+" if w > 0 else "-") + (f"{abs(w)}*" if abs(w) != 1 else "") + f"m{r+1}")
        out = "".join(ts); out = out[1:] if out.startswith("+") else out
        print(f"      c{c//p+1}{c%p+1} = {out}")


def search_rank(T, R, m, k, p, restarts, steps, lam_max, lr, device, seed0=0, polish=True, verbose=True):
    """Batched restarts at fixed R; return (best_fit, exact_or_None)."""
    fit, U, V, W = fit_batched(T, R, restarts, steps, lam_max, lr, seed0, device)
    mask = exact_mask(U, V, W, T)
    best_fit = fit.min().item()
    if mask.any():
        idx = int(mask.nonzero()[0])
        return best_fit, (U[idx].round().long(), V[idx].round().long(), W[idx].round().long())
    if polish:
        # re-anneal the best `keep` near-integer candidates at higher lattice pressure
        keep = min(32, restarts)
        order = torch.argsort(fit)[:keep]
        warm = (U[order], V[order], W[order])
        fit2, U2, V2, W2 = fit_batched(T, R, keep, steps, lam_max * 2.5, lr * 0.5, seed0 + 1, device, warm=warm)
        mask2 = exact_mask(U2, V2, W2, T)
        best_fit = min(best_fit, fit2.min().item())
        if mask2.any():
            idx = int(mask2.nonzero()[0])
            return best_fit, (U2[idx].round().long(), V2[idx].round().long(), W2[idx].round().long())
    return best_fit, None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=2); ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--Rmax", type=int, default=-1); ap.add_argument("--Rmin", type=int, default=-1)
    ap.add_argument("--restarts", type=int, default=128)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--lam", type=float, default=0.4)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    m, k, p = args.m, args.k, args.p
    naive = m * k * p
    Rmax = args.Rmax if args.Rmax > 0 else naive
    Rmin = args.Rmin if args.Rmin > 0 else max(1, naive - 2)
    T = matmul_tensor(m, k, p)
    print(f"device={args.device}  <{m},{k},{p}> matmul  T{tuple(T.shape)}  naive #mults={naive}  "
          f"sweeping R={Rmax}..{Rmin}  restarts={args.restarts} steps={args.steps}")

    found = {}
    t0 = time.time()
    for R in range(Rmax, Rmin - 1, -1):
        bf, exact = search_rank(T, R, m, k, p, args.restarts, args.steps, args.lam, args.lr, args.device)
        found[R] = (bf, exact)
        tag = "EXACT integer decomposition FOUND" if exact else "no exact (this few mults not achieved)"
        print(f"  R={R:3d}: best residual {bf:.2e}  -> {tag}   [{time.time()-t0:.0f}s]")
        if exact:
            okm, tot = verify_on_matrices(*exact, m, k, p)
            coeffs = sorted(set(exact[0].unique().tolist() + exact[1].unique().tolist() + exact[2].unique().tolist()))
            nz = int((exact[0] != 0).sum() + (exact[1] != 0).sum() + (exact[2] != 0).sum())
            print(f"        verified {okm}/{tot} random integer matrices; coeffs {coeffs}; nnz={nz}")

    exrs = [R for R, (_, ex) in found.items() if ex is not None]
    print(f"\n  === SCALING RESULT for <{m},{k},{p}> ===")
    if exrs:
        mn = min(exrs)
        print(f"  fewest multiplications with an EXACT verified algorithm: R={mn}  (naive={naive})")
        if mn < naive:
            print(f"  => SUB-NAIVE by {naive - mn}")
        describe(*found[mn][1], m, k, p)
    else:
        print(f"  no exact sub-naive decomposition found in R={Rmax}..{Rmin}")
