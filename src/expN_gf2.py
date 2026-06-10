"""
expN_gf2.py — matmul-rank discovery over GF(2) (mod-2 arithmetic), AlphaTensor's regime, from outcome alone.

Over GF(2) the matmul tensor identity is sum_r U[r,a] V[r,b] W[r,c] == T[c,a,b]  (mod 2), with coeffs in
{0,1}. Sub-Strassen ranks exist here that don't over the reals (famously 4x4 = 47 < 49). We discover them
with a WEIRD differentiable relaxation (NOT AlphaTensor's RL): a SOFT-XOR (noisy-XOR) fold.
  - coeffs p = sigmoid(theta) in (0,1);
  - per-rank contribution t_r = U_ra * V_rb * W_rc in [0,1];
  - parity of the R contributions via the differentiable XOR fold  x <- x + t - 2*x*t  (exact XOR on {0,1});
  - fit loss = (xor_fold - T)^2  (the mod-2 tensor identity), + a binarization penalty p*(1-p).
Anneal the binarization pressure; round to {0,1}; VERIFY the exact GF(2) identity + on random binary matrices.

Run: python expN_gf2.py --m 4 --k 4 --p 4 --Rmax 49 --Rmin 46 --restarts 96 --steps 9000
"""
from __future__ import annotations
import argparse, time
import torch

torch.set_default_dtype(torch.float64)


def matmul_tensor(m, k, p):
    dA, dB, dC = m * k, k * p, m * p
    T = torch.zeros(dC, dA, dB)
    for i in range(m):
        for l in range(k):
            for j in range(p):
                T[i * p + j, i * k + l, l * p + j] = 1.0
    return T


def xor_fold(P):
    """P:(B,R,dC,dA,dB) of soft bits -> parity over R via differentiable XOR. Returns (B,dC,dA,dB)."""
    B, R = P.shape[0], P.shape[1]
    x = torch.zeros(B, *P.shape[2:], device=P.device)
    for r in range(R):
        t = P[:, r]
        x = x + t - 2.0 * x * t
    return x


def reconstruct_soft(U, V, W):
    # soft bits in [0,1]; per-rank product then XOR-fold (parity) = mod-2 reconstruction surrogate
    P = torch.einsum('nra,nrb,nrc->nrcab', U, V, W)
    return xor_fold(P)


def fit_batched(T, R, B, steps, lam_max, lr, seed, device):
    dC, dA, dB = T.shape
    g = torch.Generator(device="cpu").manual_seed(seed)
    tU = (torch.randn(B, R, dA, generator=g) * 1.0).to(device).requires_grad_(True)
    tV = (torch.randn(B, R, dB, generator=g) * 1.0).to(device).requires_grad_(True)
    tW = (torch.randn(B, R, dC, generator=g) * 1.0).to(device).requires_grad_(True)
    Td = T.to(device).unsqueeze(0)
    opt = torch.optim.Adam([tU, tV, tW], lr=lr)
    for s in range(steps):
        lam = lam_max * (s / steps) ** 2
        U, V, W = torch.sigmoid(tU), torch.sigmoid(tV), torch.sigmoid(tW)
        rec = reconstruct_soft(U, V, W)
        fit = ((rec - Td) ** 2).sum(dim=(1, 2, 3))
        binp = (U * (1 - U)).sum(dim=(1, 2)) + (V * (1 - V)).sum(dim=(1, 2)) + (W * (1 - W)).sum(dim=(1, 2))
        loss = (fit + lam * binp).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        U, V, W = torch.sigmoid(tU), torch.sigmoid(tV), torch.sigmoid(tW)
        fit = ((reconstruct_soft(U, V, W) - Td) ** 2).sum(dim=(1, 2, 3))
    return fit.detach().cpu(), U.detach().cpu(), V.detach().cpu(), W.detach().cpu()


def exact_mask(U, V, W, T):
    """Round to {0,1}; check the GF(2) identity sum_r U V W == T (mod 2) exactly. mask (B,)."""
    Ui, Vi, Wi = U.round(), V.round(), W.round()
    rec = torch.einsum('nra,nrb,nrc->ncab', Ui, Vi, Wi)            # integer counts
    return ((rec.long() - T.long().unsqueeze(0)) % 2 == 0).all(dim=(1, 2, 3))


def verify_gf2(Ui, Vi, Wi, m, k, p, trials=3000, seed=0):
    g = torch.Generator().manual_seed(seed)
    dA, dB = m * k, k * p
    bad = 0
    for _ in range(trials):
        A = torch.randint(0, 2, (dA,), generator=g)
        B = torch.randint(0, 2, (dB,), generator=g)
        prod = ((Ui.long() @ A) % 2) * ((Vi.long() @ B) % 2)       # R products, in GF(2)
        c = (Wi.long().t() @ prod) % 2                              # combine mod 2
        Ctrue = (A.reshape(m, k).long() @ B.reshape(k, p).long()).reshape(-1) % 2
        if not torch.equal(c % 2, Ctrue):
            bad += 1
    return trials - bad, trials


def search_rank(T, R, m, k, p, restarts, steps, lam, lr, device):
    fit, U, V, W = fit_batched(T, R, restarts, steps, lam, lr, 0, device)
    mask = exact_mask(U, V, W, T)
    if mask.any():
        idx = int(mask.nonzero()[0])
        return fit.min().item(), (U[idx].round().long(), V[idx].round().long(), W[idx].round().long())
    return fit.min().item(), None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=2); ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--Rmax", type=int, default=-1); ap.add_argument("--Rmin", type=int, default=-1)
    ap.add_argument("--restarts", type=int, default=128); ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--lam", type=float, default=0.5); ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    m, k, p = args.m, args.k, args.p
    naive = m * k * p
    Rmax = args.Rmax if args.Rmax > 0 else naive
    Rmin = args.Rmin if args.Rmin > 0 else max(1, naive - 2)
    T = matmul_tensor(m, k, p)
    print(f"device={args.device}  GF(2) <{m},{k},{p}>  T{tuple(T.shape)}  naive={naive}  R={Rmax}..{Rmin}  "
          f"restarts={args.restarts} steps={args.steps}")
    found = {}
    t0 = time.time()
    for R in range(Rmax, Rmin - 1, -1):
        bf, exact = search_rank(T, R, m, k, p, args.restarts, args.steps, args.lam, args.lr, args.device)
        found[R] = exact
        tag = "EXACT GF(2) decomposition FOUND" if exact else "no exact"
        print(f"  R={R:3d}: best soft residual {bf:.3e}  -> {tag}   [{time.time()-t0:.0f}s]")
        if exact:
            okm, tot = verify_gf2(*exact, m, k, p)
            print(f"        verified {okm}/{tot} random binary matrices (mod 2)")
    exrs = [R for R, ex in found.items() if ex is not None]
    if exrs:
        print(f"\n  GF(2) <{m},{k},{p}>: fewest mults with EXACT mod-2 algorithm: R={min(exrs)} (naive={naive})")
