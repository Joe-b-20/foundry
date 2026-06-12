"""
gpu_exp1b_ca2d.py — the residual-novelty bridge (gpu_exp1_novelty) on 2D CAs: the heaviest, most moonshot-promising substrate.

WHY 2D, WHY GPU: 1D radius-2 novelty search runs in seconds even on the 4060 — it does NOT need a rented GPU. 2D cellular
automata are where emergent complexity famously lives (gliders, guns, universal computation — Conway's Life), and the
NON-TOTALISTIC Moore space (a 512-entry truth table over the 9-cell neighborhood = 2^512 rules) is astronomically vast and
almost entirely UNINSPECTED. Simulating B 2D grids for T steps is ~H*W*9 work per step — orders of magnitude heavier than
1D — so a large 2D novelty hunt is exactly the swing a 4060 cannot run but a 4090 can.

Same signal as gpu_exp1_novelty (imported verbatim — no re-derivation):
  novelty(orbit) = min_NAMED_bpc  −  GENERAL_bpc
  NAMED = dead/saturated (density) OR affine/additive (rule nonlinearity over 9 inputs == 0) OR short temporal period
          OR chaotic (2D damage spreading saturates). GENERAL = zlib (loop) / lzma (final) on the (T,H,W) space-time.
We reshape each 2D space-time (B,T,H,W) -> (B,T,H*W) and reuse the SAME bpc_iid / periodicity / compressor / affine /
nonlinearity helpers, so the only new code is the 2D simulation + 2D damage. Search = novelty-driven evolution over
512-bit Moore truth-tables; INSPECT the top survivors' space-time next session. Honest ceiling unchanged: characterize,
never claim novel; a survivor whose 2D dynamics look genuinely unfamiliar = evidenceable-but-unprovable novelty.

Run small (4060):  python gpu_exp1b_ca2d.py --smoke
Run scale (4090):  python gpu_exp1b_ca2d.py --H 48 --W 48 --T 96 --batch 2048 --pop 512 --gens 40 --out runs/exp1b_2d
"""
from __future__ import annotations
import argparse, time, os, json
import numpy as np
import torch
from gpu_exp1_novelty import (DEV, dev_info, bpc_compress_batch, bpc_iid, periodicity,
                              affine_tables, nonlinearity, sample_luts, mutate_luts, dedup_luts, popcount)

# 9 Moore offsets (dy,dx) with the CENTER included; bit order fixed so the rule is a 512-entry LUT over the 9-bit nbhd.
OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]


def step_moore(X: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    """X: (B,H,W) uint8 ; lut: (B,512) uint8 -> next (B,H,W) uint8. Toroidal boundary, non-totalistic Moore rule."""
    B, H, W = X.shape
    idx = torch.zeros((B, H, W), dtype=torch.long, device=X.device)
    for j, (dy, dx) in enumerate(OFFSETS):
        idx |= torch.roll(X, shifts=(dy, dx), dims=(1, 2)).long() << (8 - j)
    nxt = torch.gather(lut.long(), 1, idx.reshape(B, -1)).reshape(B, H, W)
    return nxt.to(torch.uint8)


def run_moore(lut, H, W, T, seed):
    B = lut.shape[0]
    g = torch.Generator(device=DEV).manual_seed(seed)
    X = (torch.rand((B, H, W), generator=g, device=DEV) < 0.5).to(torch.uint8)
    out = torch.empty((B, T, H, W), dtype=torch.uint8, device=DEV)
    for t in range(T):
        out[:, t] = X
        X = step_moore(X, lut)
    return out


def damage_moore(lut, H, W, T, seed):
    B = lut.shape[0]
    g = torch.Generator(device=DEV).manual_seed(seed)
    A = (torch.rand((B, H, W), generator=g, device=DEV) < 0.5).to(torch.uint8)
    Bp = A.clone(); Bp[:, H // 2, W // 2] ^= 1
    for _ in range(T):
        A = step_moore(A, lut); Bp = step_moore(Bp, lut)
    return (A != Bp).float().mean(dim=(1, 2))


def totalistic_luts(bmask: torch.Tensor, smask: torch.Tensor) -> torch.Tensor:
    """Outer-totalistic 'Life-like' rules as 512-entry Moore LUTs. bmask/smask: (K,) ints in 0..511 = which neighbor
    counts 0..8 trigger birth / survival. The 2^18 Life-like rules are FULLY ENUMERABLE and contain real structure
    (Life=B3/S23 lives here), so they validate the novelty signal AND warm-start the chaos-dominated non-totalistic hunt."""
    cfg = torch.arange(512, device=DEV)
    centerbit = (cfg >> 4) & 1                          # OFFSETS[4]=(0,0) -> bit 4 of the 9-bit index
    count = popcount(cfg) - centerbit                   # number of the 8 NEIGHBORS that are alive (0..8)
    sb = (smask[:, None] >> count[None, :]) & 1
    bb = (bmask[:, None] >> count[None, :]) & 1
    return torch.where(centerbit[None, :].bool(), sb, bb).to(torch.uint8)


# A few hand-known structured Life-like rules (birth_mask, survive_mask as bitsets over counts 0..8) for warm-starting.
def _mask(counts):
    m = 0
    for c in counts:
        m |= (1 << c)
    return m
KNOWN_LIFELIKE = [
    (_mask([3]), _mask([2, 3])),         # Conway Life  B3/S23
    (_mask([3, 6]), _mask([2, 3])),      # HighLife     B36/S23
    (_mask([3, 6, 7, 8]), _mask([3, 4, 6, 7, 8])),  # Day&Night
    (_mask([2]), _mask([])),             # Seeds        B2/S
    (_mask([3, 4]), _mask([3, 4])),      # 34 Life
    (_mask([3, 5, 7]), _mask([1, 3, 5, 8])),        # an arbitrary structured one
]


def nonlinearity_chunked(lut, aff, chunk=256):
    """NL over 9 inputs: lut (B,512) vs aff (1024,512). (B,1024,512) is big -> chunk over B."""
    outs = []
    for i in range(0, lut.shape[0], chunk):
        outs.append(nonlinearity(lut[i:i + chunk], aff))
    return torch.cat(outs)


def gate_and_named(lut, H, W, T, aff):
    pmax = max(4, min(24, T // 6))
    st4 = run_moore(lut, H, W, T, seed=7)              # (B,T,H,W)
    st = st4.reshape(st4.shape[0], T, H * W)           # reuse 1D scorers by flattening each frame
    dens = st.float().mean(dim=(1, 2))
    nl = nonlinearity_chunked(lut, aff)
    dmg = damage_moore(lut, H, W, T, seed=11)
    b_iid = bpc_iid(st)
    per_P, b_per = periodicity(st, pmax)
    # frame-to-frame ACTIVITY over the 2nd half (single-trajectory change rate). Catches noisy/boiling rules even when the
    # DAMAGE measure is fooled (B0 global-flashing absorbs a perturbation -> low damage but ~chaotic dynamics; caught live
    # on the 4090 census where B0... rules scored 1.0 with end-activity 0.95).
    half = st[:, T // 2:]
    activity = (half[:, 1:] != half[:, :-1]).float().mean(dim=(1, 2))
    no_quiescent = lut[:, 0] == 1                     # empty neighborhood -> birth (B0): no stable background, degenerate
    dead = (dens < 0.02) | (dens > 0.98)
    affine = nl == 0
    periodic = b_per < 0.25
    chaotic = (dmg > 0.30) | (activity > 0.45)        # chaos by damage OR by raw boiling activity (upper band)
    settling = activity < 0.04                        # settles to a complex-LOOKING but STATIC debris field (not
                                                      # dynamically complex) — fooled compression+damage; require ongoing
                                                      # activity. Dynamical edge-of-chaos = activity in [0.04, 0.45].
    named = dead | affine | periodic | chaotic | settling | no_quiescent
    named_min_bpc = torch.minimum(b_iid, b_per)
    return st, st4, dict(dens=dens, nl=nl, dmg=dmg, per_P=per_P, b_per=b_per,
                         dead=dead, affine=affine, periodic=periodic, chaotic=chaotic,
                         named=named, named_min_bpc=named_min_bpc)


def novelty_scores(lut, H, W, T, aff, method="zlib"):
    """Same validated edge-of-chaos complex-finder as gpu_exp1_novelty (2D damage band centered lower, ~0.12)."""
    st, st4, f = gate_and_named(lut, H, W, T, aff)
    B = lut.shape[0]
    keep = ~f["named"]
    nov = torch.zeros(B, device=lut.device); gen_bpc = torch.ones(B, device=lut.device)
    if keep.any():
        idx = keep.nonzero(as_tuple=True)[0]
        c = torch.tensor(bpc_compress_batch(st[idx].cpu().numpy(), method), device=lut.device, dtype=torch.float32)
        gen_bpc[idx] = c
        dmg = f["dmg"][idx]
        nov[idx] = (4 * c * (1 - c)) * torch.exp(-((dmg - 0.12) / 0.09) ** 2)   # 2D chaos saturates lower than 1D
    return nov, gen_bpc, f, st4


def census_totalistic(H, W, T, batch, out=None, limit=None):
    """Exhaustively score ALL 2^18 outer-totalistic Life-like rules by residual novelty (a complete census, not a sample).
    Robust: this space contains structure, so it ALWAYS yields a ranked result + warm-start seeds for the moore hunt."""
    aff = affine_tables(9, DEV)
    allmask = torch.arange(512, device=DEV)
    bm, sm = torch.meshgrid(allmask, allmask, indexing="ij")
    bm = bm.reshape(-1); sm = sm.reshape(-1)            # 262144 (birth,survive) pairs
    N = bm.shape[0] if limit is None else min(limit, bm.shape[0])
    best = []                                           # (novelty, bmask, smask)
    for i in range(0, N, batch):
        lut = totalistic_luts(bm[i:i + batch], sm[i:i + batch])
        nov, gen_bpc, f, _ = novelty_scores(lut, H, W, T, aff)
        for j in (nov > 0).nonzero(as_tuple=True)[0].tolist():
            best.append((float(nov[j]), int(bm[i + j]), int(sm[i + j]), float(gen_bpc[j]), int(f["per_P"][j]), float(f["dmg"][j])))
        if (i // batch) % 20 == 0:
            print(f"  census {i:6d}/{N} | structured-so-far {len(best)}")
    best.sort(key=lambda x: -x[0])
    print(f"\n  CENSUS DONE: {len(best)}/{N} Life-like rules are 'structured-but-not-named' (novelty>0).")
    print("    rank  Birth/Survive          novelty gen_bpc per damage")
    for r, (nv, b, s, gb, pp, dm) in enumerate(best[:16]):
        bn = "".join(str(c) for c in range(9) if (b >> c) & 1); sn = "".join(str(c) for c in range(9) if (s >> c) & 1)
        tag = "  <- CONWAY LIFE" if (b, s) == (KNOWN_LIFELIKE[0]) else ""
        print(f"    {r:>3}   B{bn or '-'}/S{sn or '-':<12s}  {nv:.3f}  {gb:.3f}  {pp:3d}  {dm:.3f}{tag}")
    if out:
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "census_totalistic.json"), "w") as fh:
            json.dump([dict(novelty=b[0], birth=b[1], survive=b[2], gen_bpc=b[3]) for b in best[:200]], fh, indent=2)
        if best:
            topm = torch.tensor([[b[1], b[2]] for b in best[:6]], device=DEV)
            np.save(os.path.join(out, "census_top_spacetime.npy"),
                    run_moore(totalistic_luts(topm[:, 0], topm[:, 1]), H, W, T, seed=7).cpu().numpy())
    return best


def evolve(H, W, T, batch, pop, gens, seed, out=None, warmstart=False):
    aff = affine_tables(9, DEV)                         # 1024 affine fns over 9 vars (additive 2D rules -> NL 0)
    m = 512
    g = torch.Generator(device=DEV).manual_seed(seed)
    pool = sample_luts(batch, m, g)
    if warmstart:                                       # the non-totalistic space is chaos-dominated; seed from structure
        seeds = totalistic_luts(torch.tensor([b for b, s in KNOWN_LIFELIKE], device=DEV),
                                torch.tensor([s for b, s in KNOWN_LIFELIKE], device=DEV))
        reps = max(1, batch // (4 * seeds.shape[0]))    # ~25% of the pool = Life-like rules + their mutations
        warm = seeds.repeat(reps, 1)
        warm = mutate_luts(warm, g, maxflip=3)
        pool[:warm.shape[0]] = warm[:pool.shape[0]]
    nov, gen_bpc, f, _ = novelty_scores(pool, H, W, T, aff)
    order = torch.argsort(nov, descending=True)
    elite = pool[order[:pop]].clone()
    hist = []
    for gen in range(gens):
        children = mutate_luts(elite, g, maxflip=4)
        cand = torch.cat([elite, children], dim=0)
        nov_c, gen_c, f_c, _ = novelty_scores(cand, H, W, T, aff)
        order = torch.argsort(nov_c, descending=True)
        elite = cand[order[:pop]].clone()
        hist.append(float(nov_c[order[0]]))
        print(f"  gen {gen:3d}: best novelty {nov_c[order[0]]:.3f} | mean(top20) {nov_c[order[:20]].mean():.3f} | "
              f"survivors {int((nov_c>0).sum())}/{cand.shape[0]}")
    elite = dedup_luts(elite)
    nov_u, gen_u, f_u, st4_u = novelty_scores(elite, H, W, T, aff, method="lzma")
    order = torch.argsort(nov_u, descending=True)
    top = order[:min(16, len(order))]
    results = [dict(novelty=float(nov_u[j]), general_bpc=float(gen_u[j]), named_min_bpc=float(f_u["named_min_bpc"][j]),
                    nl=int(f_u["nl"][j]), period=int(f_u["per_P"][j]), density=float(f_u["dens"][j]),
                    damage=float(f_u["dmg"][j])) for j in top.tolist()]
    if out:
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "survivors2d.json"), "w") as fh:
            json.dump(dict(H=H, W=W, T=T, hist=hist, results=results), fh, indent=2)
        if len(top):
            keep = elite[order[:min(6, len(order))]]
            np.save(os.path.join(out, "top2d_spacetime.npy"), run_moore(keep, H, W, T, seed=7).cpu().numpy())
            np.save(os.path.join(out, "top2d_luts.npy"), keep.cpu().numpy())
    return results, hist


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--H", type=int, default=32); ap.add_argument("--W", type=int, default=32)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--batch", type=int, default=1024); ap.add_argument("--pop", type=int, default=256)
    ap.add_argument("--gens", type=int, default=30); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--mode", choices=["census", "moore"], default="census",
                    help="census = exhaustive 2^18 Life-like (robust primary); moore = non-totalistic 2^512 hunt (stretch)")
    ap.add_argument("--warmstart", action="store_true", help="(moore) seed the chaos-dominated hunt from Life-like rules")
    ap.add_argument("--limit", type=int, default=None, help="(census) only scan the first N of the 2^18 rules")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    if args.smoke:
        args.H, args.W, args.T, args.batch, args.pop, args.gens = 16, 16, 32, 128, 32, 3
    t0 = time.time()
    if args.mode == "census":
        print(f"gpu_exp1b_ca2d CENSUS | device={dev_info()} | ALL 2^18 outer-totalistic Life-like rules | "
              f"H={args.H} W={args.W} T={args.T} batch={args.batch}\n")
        lim = args.limit if args.limit is not None else (1024 if args.smoke else None)
        census_totalistic(args.H, args.W, args.T, args.batch, out=args.out, limit=lim)
        raise SystemExit(0)
    print(f"gpu_exp1b_ca2d MOORE HUNT | device={dev_info()} | non-totalistic Moore (2^512) | H={args.H} W={args.W} "
          f"T={args.T} batch={args.batch} pop={args.pop} gens={args.gens} warmstart={args.warmstart}\n")
    results, hist = evolve(args.H, args.W, args.T, args.batch, args.pop, args.gens, args.seed,
                           out=args.out, warmstart=args.warmstart)
    print(f"\n  === TOP 2D RESIDUAL-NOVELTY SURVIVORS [{time.time()-t0:.0f}s] ===")
    print("    novelty  gen_bpc  named_bpc  NL   per  dens   damage")
    for r in results[:12]:
        print(f"    {r['novelty']:.3f}   {r['general_bpc']:.3f}    {r['named_min_bpc']:.3f}   {r['nl']:3d}  {r['period']:3d}  "
              f"{r['density']:.2f}   {r['damage']:.3f}")
    print("\n  READ: survivors = 2D rules compressible by a general compressor yet matching NO named-structure model,")
    print("  filtered from a 2^512 uninspected space. Next session: animate top2d_spacetime.npy — gliders? guns? stable")
    print("  machines? A genuinely unfamiliar 2D mechanism = evidenceable-but-unprovable novelty; 'just Life-like' = ceiling.")
