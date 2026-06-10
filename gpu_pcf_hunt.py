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
def eval_pcf_batch(Acoef, Bcoef, terms, renorm_every=1):
    """Acoef,Bcoef: (M,3) float64 tensors [c0,c1,c2] for A(n)=c0+c1 n+c2 n^2.
    Returns (value (M,), converged (M,) bool, diverged (M,) bool). float64.
    Recurrence: p_n=A(n)p_{n-1}+B(n)p_{n-2}, q_n=A(n)q_{n-1}+B(n)q_{n-2},
    p_{-1}=1,q_{-1}=0,p_0=A(0),q_0=1, value=p_N/q_N. Renormalize by q to avoid overflow."""
    M = Acoef.shape[0]
    a0 = Acoef[:, 0]
    p_prev = torch.ones(M, dtype=torch.float64, device=DEV)     # p_{-1}
    q_prev = torch.zeros(M, dtype=torch.float64, device=DEV)    # q_{-1}
    p_cur = a0.clone()                                          # p_0 = A(0)=a0
    q_cur = torch.ones(M, dtype=torch.float64, device=DEV)      # q_0 = 1
    val_prev = torch.full((M,), float("nan"), dtype=torch.float64, device=DEV)
    converged = torch.zeros(M, dtype=torch.bool, device=DEV)
    diverged = torch.zeros(M, dtype=torch.bool, device=DEV)
    for n in range(1, terms + 1):
        An = Acoef[:, 0] + Acoef[:, 1] * n + Acoef[:, 2] * n * n
        Bn = Bcoef[:, 0] + Bcoef[:, 1] * n + Bcoef[:, 2] * n * n
        p_new = An * p_cur + Bn * p_prev
        q_new = An * q_cur + Bn * q_prev
        p_prev, q_prev = p_cur, q_cur
        p_cur, q_cur = p_new, q_new
        # renormalize by q magnitude to keep values bounded
        scale = q_cur.abs().clamp(min=1e-300)
        p_cur = p_cur / scale; q_cur = q_cur / scale
        p_prev = p_prev / scale; q_prev = q_prev / scale
        if n % renorm_every == 0 and n > 8:
            qsafe = torch.where(q_cur.abs() < 1e-300, torch.full_like(q_cur, 1e-300), q_cur)
            val = p_cur / qsafe
            d = (val - val_prev).abs()
            newconv = (d < 1e-13) & torch.isfinite(val)
            converged = converged | newconv
            diverged = diverged | ~torch.isfinite(val)
            val_prev = val
    value = p_cur / torch.where(q_cur.abs() < 1e-300, torch.full_like(q_cur, 1e-300), q_cur)
    diverged = diverged | ~torch.isfinite(value)
    return value, converged & ~diverged, diverged


def gen_coef_grid(crange, deg=2):
    """All integer coeff vectors [c0,c1,c2] with |ci|<=crange, c2 in deg slot.
    Returns numpy (K,3) int. Excludes the all-zero polynomial."""
    rng = range(-crange, crange + 1)
    rows = [list(c) for c in itertools.product(rng, repeat=3)]
    rows = [r for r in rows if any(r)]
    return np.array(rows, dtype=np.float64)


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
        "gamma": mp.euler, "log2": mp.log(2), "pi^2": mp.pi ** 2,
        "sqrt2": mp.sqrt(2), "sqrt3": mp.sqrt(3), "phi": (1 + mp.sqrt(5)) / 2,
    }


def _eval_pcf_mp(A, B, mp, terms=400):
    """High-precision PCF value via the recurrence (mpmath)."""
    p_prev, q_prev = mp.mpf(1), mp.mpf(0)
    p_cur, q_cur = mp.mpf(int(A[0])), mp.mpf(1)
    for n in range(1, terms + 1):
        An = mp.mpf(int(A[0]) + int(A[1]) * n + int(A[2]) * n * n)
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
    A_list, B_list, dps_verify = args
    consts = _constants(mp)
    out = []
    for A, B in zip(A_list, B_list):
        v = _eval_pcf_mp(A, B, mp)
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
            v2 = _eval_pcf_mp(A, B, mp)
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
    # POSITIVE CONTROL injected at the front: the classical 4/pi PCF must be recovered,
    # or the whole stage-2 null is uninterpretable (audit discipline).
    A = np.vstack([np.array([[1, 2, 0]], float), A])
    B = np.vstack([np.array([[0, 0, 1]], float), B])
    n = len(A)
    print(f"  STAGE 2 (CPU PSLQ, {procs} procs): verifying {n:,} survivors against the constant battery")
    t0 = time.time()
    nchunks = max(procs * 4, 1)
    chunks = []
    sz = -(-n // nchunks)
    for s in range(0, n, sz):
        chunks.append((A[s:s + sz].tolist(), B[s:s + sz].tolist(), dps_verify))
    results = []
    with Pool(procs) as pool:
        for i, r in enumerate(pool.imap_unordered(_verify_chunk, chunks)):
            results.extend(r)
            print(f"    chunk {i+1}/{len(chunks)}  hits so far {len(results)}  [{time.time()-t0:.0f}s]")
    json.dump(results, open(os.path.join(out_dir, "stage2_hits.json"), "w"), indent=1)
    # positive-control check
    ctrl = [r for r in results if r["A"] == [1, 2, 0] and r["B"] == [0, 0, 1]]
    ctrl_ok = bool(ctrl) and ctrl[0]["constant"] == "pi"
    # per-constant breakdown — the headline: did the TAIL constants (catalan/zeta3/
    # gamma) get ANY constant-specific hit the small expV sweep missed?
    from collections import Counter
    bycon = Counter(r["constant"] for r in results)
    tail = ["catalan", "zeta3", "gamma"]
    print(f"\n  STAGE 2 done: {len(results)} constant-specific Mobius identities "
          f"(rational trap filtered) [{time.time()-t0:.0f}s]")
    print(f"  POSITIVE CONTROL (4/pi recovered): {'OK' if ctrl_ok else 'FAILED — null is uninterpretable!'}")
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
    ap.add_argument("--crange", type=int, default=6); ap.add_argument("--terms", type=int, default=80)
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
        stage1(args.crange, args.terms, args.out)
    if args.stage2:
        stage2(args.out, args.procs, args.dps, args.limit)
    if not (args.stage1 or args.stage2):
        print("specify --stage1 (GPU) and/or --stage2 (CPU verify), or --smoke/--selftest")
