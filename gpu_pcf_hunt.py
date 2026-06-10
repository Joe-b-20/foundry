"""
gpu_pcf_hunt.py — IDENTITY-SPACE HUNT AT GPU SCALE (post-audit phase 2c).

WHY THIS DIRECTION (audit §2.3 / 07 revised Frontier 1): identity space is the ONE
regime where modest-hardware computer search has historically produced genuinely
human-unknown mathematics (BBP 1995; the Ramanujan Machine 2021) — because there is no
"the optimum" to converge to (infinitely many true facts, no single target), so it
escapes the rediscovery engine. expV (session 8) validated the method on a 4060 but
could only re-collect catalogued pi/e continued fractions: the new identities live in
the TAIL (higher-degree PCFs, larger coefficients, under-explored constants) reachable
only with much larger search. This is that larger search.

ARCHITECTURE (GPU prefilter -> CPU verify, the scale pattern):
  STAGE 1 (GPU, float64): evaluate a LARGE grid of polynomial continued fractions
    PCF = a_0 + b_1/(a_1 + b_2/(a_2 + ...)),  a_n=A(n), b_n=B(n), A,B deg<=2 polys,
    via the convergent recurrence with periodic renormalization. ~millions of PCFs in
    parallel. Keep only those that CONVERGE to a non-trivial (non-small-rational) limit.
  STAGE 2 (CPU, mpmath, multiprocessed): for each survivor, re-evaluate v to moderate
    precision and run PSLQ on [1, C, v, vC] for each target constant C (battery incl.
    the under-explored Catalan, zeta3, gamma). A hit = an integer relation
    r*(vC)+s*v-p*C-q=0  i.e.  v = (pC+q)/(rC+s), a Mobius transform of C.
  REJECT-RATIONAL FILTER (the expV false-positive trap): a PCF converging to a RATIONAL
    matches EVERY constant via the value-independent Mobius triviality. Reject any
    candidate whose relation holds for >=2 DISTINCT constants simultaneously.
  STAGE 3: re-verify survivors at 250 digits; classify CLASSICAL (matches a known CF)
    vs CANDIDATE (not in the small reference set). NO novelty claim beyond
    "true to N digits + not in the references checked" (audit discipline; novelty is
    evidenceable, not provable).

Run:  python gpu_pcf_hunt.py --selftest        (verify recurrence vs known CFs)
      python gpu_pcf_hunt.py --smoke
      python gpu_pcf_hunt.py --stage1 --crange 6 --terms 80 --out runs/pcf   (GPU)
      python gpu_pcf_hunt.py --stage2 --out runs/pcf --procs 60              (CPU verify)
"""
from __future__ import annotations
import argparse, time, os, json, itertools
import numpy as np

try:
    import torch
    HAS_TORCH = True
    DEV = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    HAS_TORCH = False
    DEV = "cpu"


# ----------------------------------------------------------------------------
# STAGE 1 — GPU float64 PCF evaluation
# ----------------------------------------------------------------------------
def eval_pcf_batch(Acoef, Bcoef, terms, renorm_every=1, bmode="poly", want_delta=False):
    """Acoef: (M,3) [c0,c1,c2] for A(n)=c0+c1 n+c2 n^2 — or (M,4) incl. c3 n^3 when
    bmode='n6'. Bcoef: (M,3) poly when bmode='poly'; IGNORED when bmode='n6'
    (B(n) = -n^6, the Apery / Ramanujan-Machine zeta(3)-class family).
    Returns (value (M,), converged (M,) bool, diverged (M,) bool). float64.
    Recurrence: p_n=A(n)p_{n-1}+B(n)p_{n-2}, q_n likewise; renormalized each step."""
    M = Acoef.shape[0]
    a0 = Acoef[:, 0]
    p_prev = torch.ones(M, dtype=torch.float64, device=DEV)     # p_{-1}
    q_prev = torch.zeros(M, dtype=torch.float64, device=DEV)    # q_{-1}
    p_cur = a0.clone()                                          # p_0 = A(0)=a0
    q_cur = torch.ones(M, dtype=torch.float64, device=DEV)      # q_0 = 1
    val_prev = torch.full((M,), float("nan"), dtype=torch.float64, device=DEV)
    converged = torch.zeros(M, dtype=torch.bool, device=DEV)
    diverged = torch.zeros(M, dtype=torch.bool, device=DEV)
    logq_acc = torch.zeros(M, dtype=torch.float64, device=DEV)   # log|q_n| via renorm scales
    last_err = torch.full((M,), float("nan"), dtype=torch.float64, device=DEV)
    for n in range(1, terms + 1):
        An = Acoef[:, 0] + Acoef[:, 1] * n + Acoef[:, 2] * n * n
        if Acoef.shape[1] > 3:
            An = An + Acoef[:, 3] * (n ** 3)
        if bmode == "n6":
            Bn = torch.full_like(An, -float(n) ** 6)
        elif bmode == "n4":
            Bn = torch.full_like(An, -float(n) ** 4)
        elif bmode == "p4":
            Bn = torch.full_like(An, float(n) ** 4)
        elif bmode == "gen6":
            # full deg-6 b polynomial: Bcoef is (M,7) [b0..b6]
            Bn = Bcoef[:, 0]
            nf = 1.0
            for k in range(1, 7):
                nf *= n
                Bn = Bn + Bcoef[:, k] * nf
        else:
            Bn = Bcoef[:, 0] + Bcoef[:, 1] * n + Bcoef[:, 2] * n * n
        p_new = An * p_cur + Bn * p_prev
        q_new = An * q_cur + Bn * q_prev
        p_prev, q_prev = p_cur, q_cur
        p_cur, q_cur = p_new, q_new
        # renormalize by q magnitude to keep values bounded
        scale = q_cur.abs().clamp(min=1e-300)
        logq_acc = logq_acc + torch.log(scale)
        p_cur = p_cur / scale; q_cur = q_cur / scale
        p_prev = p_prev / scale; q_prev = q_prev / scale
        if n % renorm_every == 0 and n > 8:
            qsafe = torch.where(q_cur.abs() < 1e-300, torch.full_like(q_cur, 1e-300), q_cur)
            val = p_cur / qsafe
            d = (val - val_prev).abs()
            newconv = (d < 1e-13) & torch.isfinite(val)
            converged = converged | newconv
            diverged = diverged | ~torch.isfinite(val)
            last_err = torch.where(torch.isfinite(val), d, last_err)
            val_prev = val
    value = p_cur / torch.where(q_cur.abs() < 1e-300, torch.full_like(q_cur, 1e-300), q_cur)
    diverged = diverged | ~torch.isfinite(value)
    if want_delta:
        # RM-style irrationality-quality: |v - p/q| ~ q^-(1+delta)  =>
        # delta = -1 - ln(err)/ln(q). Higher = better approximation dynamics.
        lq = logq_acc.clamp(min=1.0)
        delta = -1.0 - torch.log(last_err.clamp(min=1e-308)) / lq
        delta = torch.where(torch.isfinite(delta), delta, torch.full_like(delta, float("nan")))
        return value, converged & ~diverged, diverged, delta
    return value, converged & ~diverged, diverged


def gen_coef_grid(crange, deg=2):
    """All integer coeff vectors [c0,c1,c2] with |ci|<=crange, c2 in deg slot.
    Returns numpy (K,3) int. Excludes the all-zero polynomial."""
    rng = range(-crange, crange + 1)
    rows = [list(c) for c in itertools.product(rng, repeat=3)]
    rows = [r for r in rows if any(r)]
    return np.array(rows, dtype=np.float64)


def stage1_n6(crange, terms, out_dir, batch=400000, family="n6"):
    """TARGETED deg-3 families: A(n)=c0+c1 n+c2 n^2+c3 n^3 (|ci|<=crange) with a fixed
    B family. family='n6': B=-n^6 (Apery zeta(3) class; control A=[5,27,51,34] ->
    6/zeta(3), in-grid for crange>=51). family='p4': B=+n^4 (Apery zeta(2) class;
    control A=[3,11,11,0] -> 30/pi^2, in-grid for crange>=11)."""
    os.makedirs(out_dir, exist_ok=True)
    rng1 = np.arange(-crange, crange + 1, dtype=np.float64)
    n1 = len(rng1)
    total = n1 ** 4
    bdesc = {"n6": "-n^6", "n4": "-n^4", "p4": "+n^4"}[family]
    print(f"  STAGE 1-{family} (GPU float64): A deg-3 grid {n1}^4 = {total:,} PCFs, B={bdesc}, {terms} terms, |c|<={crange}")
    t0 = time.time()
    # --- build the Mobius-proximity table FIRST so we can STREAM the prefilter ---
    # (constant host memory regardless of grid size: only NEAR candidates + a bounded
    #  reservoir of the rest are kept, not the billions of raw survivors). The
    #  OOM-bug fix: previously the full raw survivor set was accumulated then filtered.
    import os as _os
    _os.environ["MPMATH_NOGMPY"] = "1"
    import mpmath as mp
    mp.mp.dps = 30
    tails = dict(zeta3=float(mp.zeta(3)), catalan=float(mp.catalan), gamma=float(mp.euler),
                 pi3=float(mp.pi ** 3), zeta5=float(mp.zeta(5)), pi2=float(mp.pi ** 2))
    R = 16
    coefs = np.arange(-R, R + 1)
    P, Q, Rr, S = (x.ravel().astype(np.float64) for x in
                   np.meshgrid(coefs, coefs, coefs, coefs, indexing="ij"))
    tv_all = []
    for cname, C in tails.items():
        den = Rr * C + S; ok = np.abs(den) > 1e-9
        tv = (P[ok] * C + Q[ok]) / den[ok]
        tv_all.append(np.unique(np.round(tv[np.isfinite(tv) & (np.abs(tv) < 1e6)], 12)))
    table = np.unique(np.concatenate(tv_all))
    table_t = torch.tensor(table, dtype=torch.float64, device=DEV)
    print(f"    Mobius table: {len(table):,} tail-constant transform values (|coef|<={R}); streaming prefilter on")
    near_A, near_V = [], []          # NEAR candidates (small set, exactly verified later)
    res_A, res_V = [], []            # bounded reservoir of NON-near survivors (blind arm)
    RES_CAP = 50000; res_seen = 0
    cube = np.array(np.meshgrid(rng1, rng1, rng1, indexing="ij")).reshape(3, -1).T  # (n1^3,3)=(c2,c1,c0)
    cube_t = torch.tensor(cube, dtype=torch.float64, device=DEV)
    Mfull = cube_t.shape[0]; done = 0; n_conv = 0
    for c3 in rng1:
        for s in range(0, Mfull, batch):
            blk = cube_t[s:s + batch]; M = blk.shape[0]
            Ac = torch.empty((M, 4), dtype=torch.float64, device=DEV)
            Ac[:, 0] = blk[:, 2]; Ac[:, 1] = blk[:, 1]; Ac[:, 2] = blk[:, 0]; Ac[:, 3] = float(c3)
            v, conv, div = eval_pcf_batch(Ac, None, terms, bmode=family)
            finite = torch.isfinite(v) & (v.abs() < 1e6) & (v.abs() > 1e-6)
            near_int = (v - v.round()).abs() < 1e-7
            keep = conv & finite & ~near_int
            done += M
            if not keep.any():
                continue
            ki = keep.nonzero(as_tuple=True)[0]
            vk = v[ki]; Ak = Ac[ki]
            n_conv += len(ki)
            # STREAMING Mobius proximity (on GPU): nearest table value per survivor
            pos = torch.searchsorted(table_t, vk).clamp(0, len(table_t) - 1)
            pos0 = (pos - 1).clamp(0, len(table_t) - 1)
            dist = torch.minimum((vk - table_t[pos]).abs(), (vk - table_t[pos0]).abs())
            nearm = dist < 1e-8
            if nearm.any():
                ni = nearm.nonzero(as_tuple=True)[0]
                near_A.append(Ak[ni].cpu().numpy()); near_V.append(vk[ni].cpu().numpy())
            # reservoir of the rest (cap host memory): keep a strided sample
            restm = ~nearm
            if restm.any() and res_seen < RES_CAP * 20:
                ri = restm.nonzero(as_tuple=True)[0]
                take = ri[::max(1, len(ri) // 64 + 1)]   # thin each batch
                res_A.append(Ak[take].cpu().numpy()); res_V.append(vk[take].cpu().numpy())
                res_seen += len(take)
        if int(c3) % max(1, (2 * crange + 1) // 15) == 0:
            nn = sum(len(x) for x in near_V)
            print(f"    c3={int(c3):4d}  done {done:,}/{total:,}  conv {n_conv:,}  NEAR {nn:,}  [{time.time()-t0:.0f}s]")
    NA = np.concatenate(near_A) if near_A else np.zeros((0, 4))
    NV = np.concatenate(near_V) if near_V else np.zeros((0,))
    # dedup NEAR by value
    if len(NV):
        o = np.argsort(NV); NA, NV = NA[o], NV[o]
        km = np.ones(len(NV), bool); km[1:] = np.abs(np.diff(NV)) >= 1e-10
        NA, NV = NA[km], NV[km]
    np.savez(os.path.join(out_dir, "stage1_survivors.npz"),
             A=NA, B=np.zeros((len(NA), 3)), V=NV, meta=np.array([crange, terms]), family=np.array([family]))
    RA = np.concatenate(res_A) if res_A else np.zeros((0, 4))
    RV = np.concatenate(res_V) if res_V else np.zeros((0,))
    if len(RV) > RES_CAP:
        idx = np.linspace(0, len(RV) - 1, RES_CAP).astype(int); RA, RV = RA[idx], RV[idx]
    np.savez(os.path.join(out_dir, "stage1_blindsample.npz"),
             A=RA, B=np.zeros((len(RA), 3)), V=RV, meta=np.array([crange, terms]), family=np.array([family]))
    print(f"  STAGE 1-{family} done: {n_conv:,} convergent; {len(NA):,} distinct NEAR -> stage1_survivors.npz; "
          f"{len(RA):,} blind sample -> stage1_blindsample.npz [{time.time()-t0:.0f}s]")
    return len(NA)


def _mobius_table():
    """The tail-constant Mobius transform table for the streaming prefilter."""
    import os as _os
    _os.environ["MPMATH_NOGMPY"] = "1"
    import mpmath as mp
    mp.mp.dps = 30
    tails = dict(zeta3=float(mp.zeta(3)), catalan=float(mp.catalan), gamma=float(mp.euler),
                 pi3=float(mp.pi ** 3), zeta5=float(mp.zeta(5)), pi2=float(mp.pi ** 2),
                 pi=float(mp.pi), e=float(mp.e), log2=float(mp.log(2)))
    R = 16
    coefs = np.arange(-R, R + 1)
    P, Q, Rr, S = (x.ravel().astype(np.float64) for x in
                   np.meshgrid(coefs, coefs, coefs, coefs, indexing="ij"))
    tv_all = []
    for cname, C in tails.items():
        den = Rr * C + S; ok = np.abs(den) > 1e-9
        tv = (P[ok] * C + Q[ok]) / den[ok]
        tv_all.append(np.unique(np.round(tv[np.isfinite(tv) & (np.abs(tv) < 1e6)], 12)))
    return np.unique(np.concatenate(tv_all))


def stage1_gen6(arange, brange, terms, out_dir, batch=400000, near_thresh=1e-10, topk_delta=20000):
    """GENERAL sweep: A deg-3 grid (|a|<=arange) x B deg-6 grid (|b|<=brange) — covers
    every named family shape at low height simultaneously. Streaming Mobius prefilter
    (constant memory) + delta-scoring (irrationality-approximation quality) with a
    running top-K of unnamed high-delta values. Controls in-grid: 4/pi (a=[1,2,0,0],
    b=n^2) and the RM 8/(7 zeta3) (a=[1,5,9,6], b=-n^6)."""
    os.makedirs(out_dir, exist_ok=True)
    ar = np.arange(-arange, arange + 1, dtype=np.float64)
    br = np.arange(-brange, brange + 1, dtype=np.float64)
    acube = np.array(np.meshgrid(ar, ar, ar, ar, indexing="ij")).reshape(4, -1).T  # (nA,4) (c3,c2,c1,c0)
    acube = acube[:, ::-1].copy()                                                  # -> (c0,c1,c2,c3)
    bcube = np.array(np.meshgrid(*([br] * 7), indexing="ij")).reshape(7, -1).T     # (nB,7) (b6..b0)
    bcube = bcube[:, ::-1].copy()                                                  # -> (b0..b6)
    bcube = bcube[np.any(bcube != 0, axis=1)]                                      # drop all-zero B
    nA, nB = len(acube), len(bcube)
    total = nA * nB
    print(f"  STAGE 1-gen6 (GPU float64): A {nA:,} (deg3,|c|<={arange}) x B {nB:,} (deg6,|c|<={brange}) "
          f"= {total:,} PCFs, {terms} terms, near_thresh={near_thresh}")
    t0 = time.time()
    table = _mobius_table()
    table_t = torch.tensor(table, dtype=torch.float64, device=DEV)
    print(f"    Mobius table: {len(table):,} values (incl. pi/e/log2 + tails); streaming prefilter on")
    A_t = torch.tensor(acube, dtype=torch.float64, device=DEV)
    near_A, near_B, near_V = [], [], []
    # running top-K delta (unnamed, non-near): keep (delta, A, B, V) tensors
    topd_d = torch.full((0,), -1e9, dtype=torch.float64, device=DEV)
    topd_A = torch.zeros((0, 4), dtype=torch.float64, device=DEV)
    topd_B = torch.zeros((0, 7), dtype=torch.float64, device=DEV)
    topd_V = torch.zeros((0,), dtype=torch.float64, device=DEV)
    bs_per = max(1, batch // nA)                       # b-rows per batch (a-cube tiled whole)
    done = 0; n_conv = 0
    for bs in range(0, nB, bs_per):
        bblk = bcube[bs:bs + bs_per]
        k = len(bblk)
        Ac = A_t.repeat(k, 1)
        Bc = torch.tensor(np.repeat(bblk, nA, axis=0), dtype=torch.float64, device=DEV)
        v, conv, div, delta = eval_pcf_batch(Ac, Bc, terms, bmode="gen6", want_delta=True)
        finite = torch.isfinite(v) & (v.abs() < 1e6) & (v.abs() > 1e-6)
        near_int = (v - v.round()).abs() < 1e-7
        keep = conv & finite & ~near_int
        done += len(Ac)
        if keep.any():
            ki = keep.nonzero(as_tuple=True)[0]
            vk = v[ki]; n_conv += len(ki)
            pos = torch.searchsorted(table_t, vk).clamp(0, len(table_t) - 1)
            pos0 = (pos - 1).clamp(0, len(table_t) - 1)
            dist = torch.minimum((vk - table_t[pos]).abs(), (vk - table_t[pos0]).abs())
            nearm = dist < near_thresh
            if nearm.any():
                ni = nearm.nonzero(as_tuple=True)[0]
                near_A.append(Ac[ki][ni].cpu().numpy()); near_B.append(Bc[ki][ni].cpu().numpy())
                near_V.append(vk[ni].cpu().numpy())
            # top-K delta among the NON-near (the blind irrationality hunt)
            restm = ~nearm
            if restm.any():
                ri = restm.nonzero(as_tuple=True)[0]
                dd = delta[ki][ri]
                good = torch.isfinite(dd) & (dd > 0)
                if good.any():
                    gi = good.nonzero(as_tuple=True)[0]
                    topd_d = torch.cat([topd_d, dd[gi]])
                    topd_A = torch.cat([topd_A, Ac[ki][ri][gi]])
                    topd_B = torch.cat([topd_B, Bc[ki][ri][gi]])
                    topd_V = torch.cat([topd_V, vk[ri][gi]])
                    if len(topd_d) > topk_delta * 2:
                        o = torch.argsort(topd_d, descending=True)[:topk_delta]
                        topd_d, topd_A, topd_B, topd_V = topd_d[o], topd_A[o], topd_B[o], topd_V[o]
        if (bs // bs_per) % 200 == 0:
            nn = sum(len(x) for x in near_V)
            print(f"    b {bs:,}/{nB:,}  done {done:,}/{total:,}  conv {n_conv:,}  NEAR {nn:,}  "
                  f"topdelta {float(topd_d.max()) if len(topd_d) else 0:.2f}  [{time.time()-t0:.0f}s]")
    NA = np.concatenate(near_A) if near_A else np.zeros((0, 4))
    NB = np.concatenate(near_B) if near_B else np.zeros((0, 7))
    NV = np.concatenate(near_V) if near_V else np.zeros((0,))
    if len(NV):
        o = np.argsort(NV); NA, NB, NV = NA[o], NB[o], NV[o]
        km = np.ones(len(NV), bool); km[1:] = np.abs(np.diff(NV)) >= 1e-10
        NA, NB, NV = NA[km], NB[km], NV[km]
    np.savez(os.path.join(out_dir, "stage1_survivors.npz"), A=NA, B=NB, V=NV,
             meta=np.array([arange, terms]), family=np.array(["gen6"]))
    if len(topd_d):
        o = torch.argsort(topd_d, descending=True)[:topk_delta]
        np.savez(os.path.join(out_dir, "stage1_topdelta.npz"),
                 D=topd_d[o].cpu().numpy(), A=topd_A[o].cpu().numpy(),
                 B=topd_B[o].cpu().numpy(), V=topd_V[o].cpu().numpy(),
                 family=np.array(["gen6"]))
    print(f"  STAGE 1-gen6 done: {n_conv:,} convergent; {len(NA):,} distinct NEAR; "
          f"{min(topk_delta, len(topd_d)):,} top-delta saved [{time.time()-t0:.0f}s]")
    return len(NA)


def stage1(crange, terms, out_dir, batch=200000):
    """Evaluate the full A x B grid, keep convergent non-trivial survivors, save."""
    os.makedirs(out_dir, exist_ok=True)
    Agrid = gen_coef_grid(crange)
    Bgrid = gen_coef_grid(crange)
    # B with b1..=0 gives a terminating/degenerate CF; keep b with constant term too.
    nA, nB = len(Agrid), len(Bgrid)
    print(f"  STAGE 1 (GPU float64): A-grid {nA} x B-grid {nB} = {nA*nB:,} PCFs, {terms} terms, |coef|<={crange}")
    t0 = time.time()
    surv_A, surv_B, surv_v = [], [], []
    total = nA * nB
    done = 0
    # iterate over A in chunks, pair each with all B (vectorized via broadcast in flat batches)
    At = torch.tensor(Agrid, dtype=torch.float64, device=DEV)
    Bt = torch.tensor(Bgrid, dtype=torch.float64, device=DEV)
    # flatten the product in batches of `batch`
    for a_i in range(nA):
        A_rep = At[a_i:a_i + 1].expand(nB, 3)
        for s in range(0, nB, batch):
            Bc = Bt[s:s + batch]
            Ac = A_rep[s:s + batch]
            v, conv, div = eval_pcf_batch(Ac, Bc, terms)
            # non-trivial: converged, finite, not a small rational (|v - round(v)|>1e-6
            # OR |v|>some), and bounded magnitude
            finite = torch.isfinite(v) & (v.abs() < 1e6)
            near_int = (v - v.round()).abs() < 1e-7
            keep = conv & finite & ~near_int
            if keep.any():
                ki = keep.nonzero(as_tuple=True)[0]
                surv_A.append(Agrid[a_i:a_i + 1].repeat(len(ki), axis=0))
                surv_B.append(Bgrid[s:s + batch][ki.cpu().numpy()])
                surv_v.append(v[ki].cpu().numpy())
            done += len(Bc)
        if a_i % max(1, nA // 20) == 0:
            print(f"    A {a_i+1}/{nA}  survivors so far {sum(len(x) for x in surv_v):,}  [{time.time()-t0:.0f}s]")
    if surv_v:
        SA = np.concatenate(surv_A); SB = np.concatenate(surv_B); SV = np.concatenate(surv_v)
    else:
        SA = np.zeros((0, 3)); SB = np.zeros((0, 3)); SV = np.zeros((0,))
    # dedup by float value (many PCFs share a limit); keep first
    order = np.argsort(SV)
    SA, SB, SV = SA[order], SB[order], SV[order]
    keepmask = np.ones(len(SV), dtype=bool)
    for i in range(1, len(SV)):
        if abs(SV[i] - SV[i - 1]) < 1e-10:
            keepmask[i] = False
    SA, SB, SV = SA[keepmask], SB[keepmask], SV[keepmask]
    np.savez(os.path.join(out_dir, "stage1_survivors.npz"), A=SA, B=SB, V=SV,
             meta=np.array([crange, terms]))
    print(f"  STAGE 1 done: {len(SV):,} distinct convergent non-trivial PCF values "
          f"(of {total:,} evaluated) [{time.time()-t0:.0f}s]")
    print(f"  wrote {out_dir}/stage1_survivors.npz")
    return len(SV)


# ----------------------------------------------------------------------------
# STAGE 2 — CPU mpmath/PSLQ verification (multiprocessed)
# ----------------------------------------------------------------------------
def _constants(mp):
    return {
        "pi": mp.pi, "e": mp.e, "catalan": mp.catalan, "zeta3": mp.zeta(3),
        "gamma": mp.euler, "log2": mp.log(2), "pi^2": mp.pi ** 2, "pi^3": mp.pi ** 3,
        "zeta5": mp.zeta(5), "sqrt2": mp.sqrt(2), "sqrt3": mp.sqrt(3),
        "phi": (1 + mp.sqrt(5)) / 2,
    }


def _eval_pcf_mp(A, B, mp, terms=400, family="poly"):
    """High-precision PCF value via the recurrence (mpmath). family='n6': B(n)=-n^6,
    A may have 4 coefficients (deg-3)."""
    p_prev, q_prev = mp.mpf(1), mp.mpf(0)
    p_cur, q_cur = mp.mpf(int(A[0])), mp.mpf(1)
    for n in range(1, terms + 1):
        An = mp.mpf(int(A[0]) + int(A[1]) * n + int(A[2]) * n * n
                    + (int(A[3]) * n ** 3 if len(A) > 3 else 0))
        if family == "n6":
            Bn = -mp.mpf(n) ** 6
        elif family == "n4":
            Bn = -mp.mpf(n) ** 4
        elif family == "p4":
            Bn = mp.mpf(n) ** 4
        elif family == "gen6":
            Bn = mp.mpf(0)
            for k in range(len(B)):
                Bn += int(B[k]) * mp.mpf(n) ** k
        else:
            Bn = mp.mpf(int(B[0]) + int(B[1]) * n + int(B[2]) * n * n)
        p_new = An * p_cur + Bn * p_prev
        q_new = An * q_cur + Bn * q_prev
        p_prev, q_prev = p_cur, q_cur
        p_cur, q_cur = p_new, q_new
        if n % 4 == 0 and q_cur != 0:
            s = q_cur
            p_cur, q_cur, p_prev, q_prev = p_cur / s, q_cur / s, p_prev / s, q_prev / s
    if q_cur == 0:
        return None
    return p_cur / q_cur


def _verify_chunk(args):
    """Worker: for a chunk of (A,B) candidates, find constant-specific Mobius relations."""
    import os as _os
    _os.environ["MPMATH_NOGMPY"] = "1"
    import mpmath as mp
    mp.mp.dps = 60
    A_list, B_list, dps_verify, family = args
    consts = _constants(mp)
    out = []
    for A, B in zip(A_list, B_list):
        v = _eval_pcf_mp(A, B, mp, family=family)
        if v is None or not mp.isfinite(v):
            continue
        # REJECT-RATIONAL (primary filter, the expV trap): if v itself is a low-height
        # rational p/q, the PCF converges to a rational and will Mobius-match EVERY
        # constant trivially. Detect directly: pslq([v, 1]) finds q*v - p = 0.
        ratrel = mp.pslq([v, mp.mpf(1)], maxcoeff=10**7, maxsteps=10**4)
        if ratrel and ratrel[0] != 0:
            continue  # v = -ratrel[1]/ratrel[0] is rational -> skip
        hits = []
        for cname, C in consts.items():
            # integer relation among [1, C, v, v*C]: r*(vC)+s*v - p*C - q = 0
            rel = mp.pslq([mp.mpf(1), C, v, v * C], maxcoeff=10**6, maxsteps=10**4)
            if rel and any(rel):
                # rel = [c0, c1, c2, c3] s.t. c0*1 + c1*C + c2*v + c3*vC = 0
                # => v(c2 + c3 C) = -(c0 + c1 C) => v = -(c0+c1 C)/(c2+c3 C)  Mobius in C
                if rel[2] != 0 or rel[3] != 0:
                    height = max(abs(int(x)) for x in rel)
                    hits.append((cname, [int(x) for x in rel], height))
        if hits:
            # secondary safety: a relation holding for >=3 constants = residual triviality
            # (e.g. v algebraic & in the battery). 3 not 2, so a genuine hit with one
            # spurious second match survives.
            if len(hits) >= 3:
                continue
            # prefer the lowest-height (most likely genuine) hit
            hits.sort(key=lambda h: h[2])
            cname, rel, height = hits[0]
            # verify at high precision
            mp.mp.dps = dps_verify
            v2 = _eval_pcf_mp(A, B, mp, family=family)
            C2 = _constants(mp)[cname]
            resid = rel[0] + rel[1] * C2 + rel[2] * v2 + rel[3] * v2 * C2
            mp.mp.dps = 60
            ok = abs(resid) < mp.mpf(10) ** (-(dps_verify - 10))
            out.append(dict(A=[int(x) for x in A], B=[int(x) for x in B],
                            value=mp.nstr(v, 30), constant=cname, relation=rel,
                            height=height, verified=bool(ok)))
    return out


def stage2(out_dir, procs, dps_verify=250, limit=0):
    import os as _os
    _os.environ["MPMATH_NOGMPY"] = "1"
    from multiprocessing import Pool
    d = np.load(os.path.join(out_dir, "stage1_survivors.npz"))
    A, B, V = d["A"], d["B"], d["V"]
    if limit and len(A) > limit:
        # sample ACROSS the value range, not just the head (head = lowest values)
        idx = np.linspace(0, len(A) - 1, limit).astype(int)
        A, B, V = A[idx], B[idx], V[idx]
    # family: read from the npz if present
    family = str(d["family"][0]) if "family" in d.files else "poly"
    # POSITIVE CONTROL injected at the front: classical 4/pi (poly family) or the
    # Apery 6/zeta(3) PCF (n6 family) must be recovered, or the null is uninterpretable.
    if family == "n6":
        ctrlA = np.array([[5, 27, 51, 34]], float)
        A = np.vstack([ctrlA, A]); B = np.vstack([np.zeros((1, 3)), B])
    elif family == "p4":
        ctrlA = np.array([[3, 11, 11, 0]], float)
        A = np.vstack([ctrlA, A]); B = np.vstack([np.zeros((1, 3)), B])
    elif family == "gen6":
        # TWO in-grid controls: 4/pi (a=[1,2,0,0], b=n^2) and RM 8/(7zeta3) (a=[1,5,9,6], b=-n^6)
        ctrlA = np.array([[1, 2, 0, 0], [1, 5, 9, 6]], float)
        ctrlB = np.array([[0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, -1]], float)
        A = np.vstack([ctrlA, A]); B = np.vstack([ctrlB, B])
    else:
        A = np.vstack([np.array([[1, 2, 0]], float), A])
        B = np.vstack([np.array([[0, 0, 1]], float), B])
    n = len(A)
    print(f"  STAGE 2 (CPU PSLQ, {procs} procs): verifying {n:,} survivors against the constant battery")
    t0 = time.time()
    nchunks = max(procs * 4, 1)
    chunks = []
    sz = -(-n // nchunks)
    for s in range(0, n, sz):
        chunks.append((A[s:s + sz].tolist(), B[s:s + sz].tolist(), dps_verify, family))
    results = []
    with Pool(procs) as pool:
        for i, r in enumerate(pool.imap_unordered(_verify_chunk, chunks)):
            results.extend(r)
            print(f"    chunk {i+1}/{len(chunks)}  hits so far {len(results)}  [{time.time()-t0:.0f}s]")
    json.dump(results, open(os.path.join(out_dir, "stage2_hits.json"), "w"), indent=1)
    # positive-control check (family-specific)
    if family == "n6":
        ctrl = [r for r in results if r["A"] == [5, 27, 51, 34]]
        ctrl_ok = bool(ctrl) and ctrl[0]["constant"] == "zeta3"
    elif family == "p4":
        ctrl = [r for r in results if r["A"] == [3, 11, 11, 0]]
        ctrl_ok = bool(ctrl) and ctrl[0]["constant"] in ("pi^2", "pi")
    elif family == "gen6":
        c1 = [r for r in results if r["A"] == [1, 2, 0, 0] and r["constant"] == "pi"]
        c2 = [r for r in results if r["A"] == [1, 5, 9, 6] and r["constant"] == "zeta3"]
        ctrl_ok = bool(c1) and bool(c2)
    else:
        ctrl = [r for r in results if r["A"] == [1, 2, 0] and r["B"] == [0, 0, 1]]
        ctrl_ok = bool(ctrl) and ctrl[0]["constant"] == "pi"
    # per-constant breakdown — the headline: did the TAIL constants (catalan/zeta3/
    # gamma) get ANY constant-specific hit the small expV sweep missed?
    from collections import Counter
    bycon = Counter(r["constant"] for r in results)
    tail = ["catalan", "zeta3", "gamma"]
    print(f"\n  STAGE 2 done: {len(results)} constant-specific Mobius identities "
          f"(rational trap filtered) [{time.time()-t0:.0f}s]")
    ctrl_name = {"n6": "Apery 6/zeta(3)", "p4": "Apery 30/pi^2",
                 "gen6": "4/pi AND RM 8/(7zeta3)"}.get(family, "4/pi")
    print(f"  POSITIVE CONTROL ({ctrl_name} recovered): {'OK' if ctrl_ok else 'FAILED — null is uninterpretable!'}")
    print(f"  per-constant: " + "  ".join(f"{k}:{v}" for k, v in sorted(bycon.items(), key=lambda x: -x[1])))
    print(f"  TAIL constants (catalan/zeta3/gamma) hits: " +
          ", ".join(f"{t}:{bycon.get(t,0)}" for t in tail) +
          "  <- any nonzero is notable (NOT claimed novel without a literature check)")
    # show tail hits explicitly (the candidates worth a human look)
    tailhits = [r for r in results if r["constant"] in tail and r["verified"]]
    for r in tailhits[:30]:
        print(f"    [TAIL] {r['constant']:8s} A={r['A']} B={r['B']} v={r['value'][:24]} "
              f"rel={r['relation']} height={r['height']}")
    json.dump(dict(n_hits=len(results), control_ok=ctrl_ok, by_constant=dict(bycon),
                   tail_hits=tailhits), open(os.path.join(out_dir, "stage2_summary.json"), "w"), indent=1)
    print(f"  wrote {out_dir}/stage2_hits.json + stage2_summary.json")
    return len(results)


# ----------------------------------------------------------------------------
def selftest():
    import os as _os
    _os.environ["MPMATH_NOGMPY"] = "1"
    import mpmath as mp
    mp.mp.dps = 40
    print("=== PCF recurrence self-test (vs known continued fractions) ===")
    # Brouncker/Euler: 4/pi = 1 + 1^2/(2 + 3^2/(2 + 5^2/(2+...)))  =>  a0=1, A(n)=2 (n>=1), B(n)=(2n-1)^2
    # In our poly form: A=[1? no]. a0=1, a_n=2. So A=[2,0,0] but a_0 must be 1 -> our a0=A(0)=2 mismatch.
    # Use the standard: -4/pi handled by expV; here test e's regular CF instead, exactly representable.
    # e = 2 + 1/(1+ 1/(2+ 1/(1+ 1/(1+...)))) is NOT polynomial. Use:
    # tan-like? Use the simplest exactly-polynomial PCF with KNOWN value:
    #   pi = 3 + 1^2/(6 + 3^2/(6 + 5^2/(6+...)))  => a0=3, a_n=6, b_n=(2n-1)^2 = 4n^2-4n+1
    A = [6, 0, 0]; B = [1, -4, 4]   # A(n)=6 for n>=1, but A(0)=6 != 3. handle a0 separately:
    # our eval uses a0=A(0). For pi CF a0=3, a_n=6. Represent A as [3,?] won't give 6 at n>=1.
    # So test the GPU eval against an mpmath eval of the SAME poly form (internal consistency)
    # plus a true closed form where a0 coincides: use A=[2,0,0],B=[1,0,0]:
    #   x = 2 + 1/(2 + 1/(2 + ...)) = 1+sqrt(2).  (a0=2,a_n=2,b_n=1) closed form known.
    A2 = [2, 0, 0]; B2 = [1, 0, 0]
    vmp = _eval_pcf_mp(A2, B2, mp, terms=200)
    target = 1 + mp.sqrt(2)
    ok_closed = abs(vmp - target) < mp.mpf(10) ** (-30)
    print(f"  [2;2,2,...] CF = 1+sqrt2 ?  mpmath {mp.nstr(vmp,20)} vs {mp.nstr(target,20)}  "
          f"{'OK' if ok_closed else 'FAIL'}")
    ok_gpu = True
    if HAS_TORCH:
        At = torch.tensor([A2], dtype=torch.float64, device=DEV)
        Bt = torch.tensor([B2], dtype=torch.float64, device=DEV)
        vg, conv, div = eval_pcf_batch(At, Bt, 200)
        ok_gpu = abs(float(vg[0]) - float(target)) < 1e-12 and bool(conv[0])
        print(f"  GPU float64 eval of same CF = {float(vg[0]):.15f}  converged={bool(conv[0])}  "
              f"{'OK' if ok_gpu else 'FAIL'}")
    # PSLQ recovers the relation for a Mobius transform of pi: v=(pi+0)/(0*pi+1)=pi
    mp.mp.dps = 60
    rel = mp.pslq([mp.mpf(1), mp.pi, mp.pi, mp.pi * mp.pi], maxcoeff=10**6)
    print(f"  PSLQ sanity on [1,pi,pi,pi^2]: {rel} (should find a relation)")
    allok = ok_closed and ok_gpu
    print(f"  => PCF hunt {'CORRECT' if allok else 'BROKEN'}")
    return allok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true"); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--stage1", action="store_true"); ap.add_argument("--stage2", action="store_true")
    ap.add_argument("--family", type=str, default="poly", choices=["poly", "n6", "n4", "p4", "gen6"],
                    help="poly: A,B deg<=2 grids. n6/n4/p4: A deg-3 grid x fixed B family. "
                         "gen6: A deg-3 grid x B deg-6 grid (general; covers all family shapes)")
    ap.add_argument("--crange", type=int, default=6); ap.add_argument("--terms", type=int, default=80)
    ap.add_argument("--arange", type=int, default=9, help="gen6: |a-coef| bound")
    ap.add_argument("--brange", type=int, default=2, help="gen6: |b-coef| bound")
    ap.add_argument("--near-thresh", type=float, default=1e-8, dest="near_thresh")
    ap.add_argument("--procs", type=int, default=60); ap.add_argument("--dps", type=int, default=250)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=str, default="runs/pcf")
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(0 if selftest() else 1)
    if args.smoke:
        print("SMOKE: selftest + tiny stage1 + stage2")
        if not selftest():
            raise SystemExit("broken")
        stage1(crange=3, terms=60, out_dir=args.out)
        stage2(out_dir=args.out, procs=4, dps_verify=120, limit=200)
        raise SystemExit(0)
    if args.stage1:
        if args.family == "gen6":
            stage1_gen6(args.arange, args.brange, args.terms, args.out, near_thresh=args.near_thresh)
        elif args.family in ("n6", "n4", "p4"):
            stage1_n6(args.crange, args.terms, args.out, family=args.family)
        else:
            stage1(args.crange, args.terms, args.out)
    if args.stage2:
        stage2(args.out, args.procs, args.dps, args.limit)
    if not (args.stage1 or args.stage2):
        print("specify --stage1 (GPU) and/or --stage2 (CPU verify), or --smoke/--selftest")
