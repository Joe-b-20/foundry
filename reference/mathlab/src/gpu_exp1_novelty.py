"""
gpu_exp1_novelty.py — RESIDUAL-NOVELTY bridge at GPU scale (the moonshot swing).

THE CEILING THIS ATTACKS (session-9 synthesis, verbatim): "to find a human-unknown object you need an intrinsic signal
pointed at a structure-class NOT already named, at a scale large enough that the tail is reachable." Every prior bridge
signal (expX/Y/Z/AA/DD) was BUILT to flag a KNOWN structure class (edge-of-chaos, nonlinearity, damage-band) — so scale
alone only resurfaces known classes. This builds a signal pointed at UN-named structure:

    novelty(orbit) = min_over_NAMED_models( bits-per-cell )  −  GENERAL_compressor( bits-per-cell )

A large positive gap = "this orbit is compressible by a general-purpose compressor, yet NONE of my named-structure models
compresses it" = structured in a way I have no name for. That is the only honest operationalization of 'unknown structure'.
We then SEARCH a vast UNCATALOGUED space (radius-2 = 2^32 rules, radius-3 = 2^128) at GPU scale — sampling + a
novelty-DRIVEN evolutionary hunt — and INSPECT the top residual-novelty survivors. (Honest ceiling, unchanged: uncatalogued
!= provably-unknown; survivors are CHARACTERIZED, never claimed novel. The win condition is a survivor whose dynamics look
genuinely unfamiliar on inspection = evidenceable-but-unprovable novelty.)

NAMED-structure description-lengths (each tied to a known CA phenomenon; all GPU-vectorized, cheap):
  bpc_iid      — memoryless entropy H(density): dead/sparse/saturated orbits (Wolfram class 1).
  bpc_periodic — exact short temporal period P: oscillators / still-lifes / simple cyclic (class 2).
  bpc_affine   — rule nonlinearity NL==0 => additive/linear (Sierpinski rules 90/150/...): explained for ~0 bits.
  (chaos)      — damage-spreading saturation (~0.5): positive Lyapunov => INCOMPRESSIBLE noise (class 3). EXCLUDED, not novel.
GENERAL description-length (structure-agnostic, strong):
  bpc_lzma     — lzma of the packed space-time bytes (BWT+range-coding; catches repetition/gliders/long-range). DEFAULT.
  bpc_neural   — (--neural) ONE next-row predictor SHARED across the whole searched population. Shared weights cannot
                 memorize any single rule's local map, so it can only exploit emergent regularity that generalizes ACROSS
                 rules — a genuinely rule-agnostic 'general structure' detector. The GPU-distinctive axis.

Two-stage search keeps the expensive general compressor off the hot path: (1) a cheap GPU gate rejects dead/affine/
short-periodic/chaotic rules en masse; (2) the survivors get the general compressor + novelty ranking; (3) a
novelty-driven evolutionary loop hunts deeper into the tail.

Run small (4060 sanity):  python gpu_exp1_novelty.py --smoke
Run scale  (RunPod 4090):  python gpu_exp1_novelty.py --radius 2 --batch 16384 --gens 40 --pop 4096 --out runs/exp1_r2
"""
from __future__ import annotations
import argparse, lzma, time, os, json
import numpy as np
import torch

# ---------------------------------------------------------------------------
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def dev_info():
    if DEV == "cuda":
        p = torch.cuda.get_device_properties(0)
        return f"{p.name} {p.total_memory/1e9:.0f}GB sm_{p.major}{p.minor}"
    return "cpu"


# ---------------------------------------------------------------------------
# Batched 1D CA simulation. rules: LUTs (B, 2^k) uint8; state (B, W) uint8.
# ---------------------------------------------------------------------------
def sample_luts(n: int, m: int, gen) -> torch.Tensor:
    """n random rule truth-tables of size m=2^k (the GENOME is the LUT itself — no 64-bit rule-number limit, any radius)."""
    return (torch.rand((n, m), generator=gen, device=DEV) < 0.5).to(torch.uint8)


def mutate_luts(luts: torch.Tensor, gen, maxflip: int = 3) -> torch.Tensor:
    """Flip 1..maxflip random truth-table entries per child."""
    n, m = luts.shape
    out = luts.clone()
    nflip = torch.randint(1, maxflip + 1, (n, 1), generator=gen, device=DEV)
    for s in range(maxflip):
        do = (nflip > s)                                  # (n,1) still flipping?
        pos = torch.randint(0, m, (n, 1), generator=gen, device=DEV)
        flip = torch.zeros_like(out).scatter_(1, pos, 1)
        out = torch.where((do & (flip > 0)), out ^ 1, out)
    return out


def dedup_luts(luts: torch.Tensor) -> torch.Tensor:
    """Drop duplicate truth-tables (pack bits -> unique rows)."""
    arr = luts.detach().cpu().numpy().astype(np.uint8)
    packed = np.packbits(arr, axis=1)
    _, idx = np.unique(packed, axis=0, return_index=True)
    return luts[torch.tensor(sorted(idx.tolist()), device=luts.device)]


def lut_to_int(lut: torch.Tensor):
    """Truth-table -> rule number (only meaningful / readable for small k<=5; else returns None)."""
    if lut.shape[0] > 32:
        return None
    v = 0
    for i in range(lut.shape[0]):
        v |= int(lut[i].item()) << i
    return v


def step_ca(X: torch.Tensor, lut: torch.Tensor, radius: int) -> torch.Tensor:
    """X: (B, W) uint8 ; lut: (B, 2^k) uint8 -> next (B, W) uint8. Periodic boundary."""
    B, W = X.shape
    k = 2 * radius + 1
    idx = torch.zeros((B, W), dtype=torch.long, device=X.device)
    for j, off in enumerate(range(radius, -radius - 1, -1)):
        idx |= torch.roll(X, shifts=off, dims=1).long() << (k - 1 - j)
    return torch.gather(lut.long(), 1, idx).to(torch.uint8)


def run_ca(lut: torch.Tensor, radius: int, W: int, T: int, seed: int) -> torch.Tensor:
    """-> space-time (B, T, W) uint8 from a fixed random init (same init across the batch for a fair compare)."""
    B = lut.shape[0]
    g = torch.Generator(device=DEV).manual_seed(seed)
    X = (torch.rand((B, W), generator=g, device=DEV) < 0.5).to(torch.uint8)
    out = torch.empty((B, T, W), dtype=torch.uint8, device=DEV)
    for t in range(T):
        out[:, t] = X
        X = step_ca(X, lut, radius)
    return out


def damage(lut: torch.Tensor, radius: int, W: int, T: int, seed: int) -> torch.Tensor:
    """Final Hamming fraction between an orbit and a 1-cell-perturbed twin. ~0.5 chaos, ~0 heals, intermediate glider."""
    B = lut.shape[0]
    g = torch.Generator(device=DEV).manual_seed(seed)
    A = (torch.rand((B, W), generator=g, device=DEV) < 0.5).to(torch.uint8)
    Bp = A.clone(); Bp[:, W // 2] ^= 1
    for _ in range(T):
        A = step_ca(A, lut, radius); Bp = step_ca(Bp, lut, radius)
    return (A != Bp).float().mean(dim=1)


# ---------------------------------------------------------------------------
# NAMED-structure detectors (all batched on GPU).
# ---------------------------------------------------------------------------
def density(st: torch.Tensor) -> torch.Tensor:
    return st.float().mean(dim=(1, 2))  # (B,)


def bpc_iid(st: torch.Tensor) -> torch.Tensor:
    """Memoryless entropy H(p) in bits/cell — the 'no structure beyond density' (dead/sparse) model."""
    p = density(st).clamp(1e-6, 1 - 1e-6)
    return -(p * torch.log2(p) + (1 - p) * torch.log2(1 - p))


def periodicity(st: torch.Tensor, pmax: int):
    """Smallest temporal period P in 1..pmax such that the last (T-pmax) rows are EXACTLY P-periodic (else P=0), and a
    two-part-code bits/cell: a periodic orbit ~ store the first `pmax` rows (transient) + one period of P rows; non-periodic
    -> 1.0 bpc. Fully batched. Returns (P (B,), bpc (B,))."""
    B, T, W = st.shape
    a = st[:, pmax:]                              # (B, T-pmax, W) the rows we test for periodicity
    P = torch.zeros(B, dtype=torch.long, device=st.device)
    for p in range(1, pmax + 1):
        b = st[:, pmax - p: T - p]               # same rows shifted back by p
        match = (a == b).all(dim=2).all(dim=1)   # (B,) exactly p-periodic over the tail
        P = torch.where((P == 0) & match, torch.full_like(P, p), P)
    bits = (pmax + P).float() * W
    bpc = torch.where(P > 0, bits / (T * W), torch.ones(B, device=st.device))
    return P, bpc


def affine_tables(k: int, device) -> torch.Tensor:
    """All 2^(k+1) affine functions of k vars as (M, 2^k) uint8 truth tables: aff(x) = c0 ^ parity(mask & x)."""
    n = 1 << k
    xs = torch.arange(n, device=device)
    masks = torch.arange(n, device=device)
    par = (popcount(masks[:, None] & xs[None, :]) & 1).to(torch.uint8)   # (n, n)
    tabs = torch.cat([par, par ^ 1], dim=0)                              # add c0=1 versions
    return tabs


def popcount(x: torch.Tensor) -> torch.Tensor:
    x = x.clone()
    c = torch.zeros_like(x)
    while x.any():
        c += (x & 1); x >>= 1
    return c


def nonlinearity(lut: torch.Tensor, aff: torch.Tensor, chunk: int = 8192) -> torch.Tensor:
    """Min Hamming distance of each rule's truth table to ANY affine function. NL==0 <=> additive/linear rule. (B,)
    Chunked over B: the intermediate (B, M, n) tensor is huge at radius-3 (M=256, n=128) and would OOM otherwise."""
    outs = []
    for i in range(0, lut.shape[0], chunk):
        d = (lut[i:i + chunk, None, :] != aff[None, :, :]).sum(dim=2)
        outs.append(d.min(dim=1).values)
    return torch.cat(outs)


# ---------------------------------------------------------------------------
# GENERAL compressors.
# ---------------------------------------------------------------------------
def bpc_compress_batch(st_cpu: np.ndarray, method: str = "zlib") -> np.ndarray:
    """General-compressor bits/cell per pattern (CPU). zlib = fast (search loop); lzma = stronger (final ranking).
    st_cpu: (B, T, W) uint8 -> (B,) float."""
    import zlib as _zlib
    B, T, W = st_cpu.shape
    out = np.empty(B, dtype=np.float64)
    filt = [{"id": lzma.FILTER_LZMA2, "preset": 6}]
    for i in range(B):
        packed = np.packbits(st_cpu[i].reshape(-1)).tobytes()
        if method == "lzma":
            comp = lzma.compress(packed, format=lzma.FORMAT_RAW, filters=filt)
        else:
            comp = _zlib.compress(packed, 6)
        out[i] = 8.0 * len(comp) / (T * W)
    return out


class SharedRowPredictor(torch.nn.Module):
    """ONE causal next-row predictor shared across all rules: predicts row t from a window of rows [t-L, t-1] via a small
    1D-conv stack over width. Shared weights can't store per-rule local maps -> only captures cross-rule emergent regularity.
    Its bits/cell on an orbit = a rule-agnostic 'general compressibility'."""
    def __init__(self, L=4, hid=64):
        super().__init__()
        self.L = L
        self.net = torch.nn.Sequential(
            torch.nn.Conv1d(L, hid, 7, padding=3), torch.nn.ReLU(),
            torch.nn.Conv1d(hid, hid, 7, padding=3), torch.nn.ReLU(),
            torch.nn.Conv1d(hid, 1, 7, padding=3),
        )

    def logits(self, st_f):                      # st_f: (B, T, W) float
        B, T, W = st_f.shape
        L = self.L
        # build windows: for each target row t in [L, T), context = rows [t-L, t-1] as L channels
        ctx = torch.stack([st_f[:, t - L:t] for t in range(L, T)], dim=1)   # (B, T-L, L, W)
        ctx = ctx.reshape(-1, L, W)                                          # (B*(T-L), L, W)
        return self.net(ctx).reshape(B, T - L, W)                           # (B, T-L, W)

    @torch.no_grad()
    def bpc(self, st_f, chunk=64):
        # CHUNK over the batch: logits() expands to (B*(T-L), hid, W); at scale B~10^4 this is 100+ GiB, so process
        # in small chunks under no_grad. (Smoke B is tiny so this never mattered locally — a scale-only OOM.)
        outs = []
        for i in range(0, st_f.shape[0], chunk):
            sf = st_f[i:i + chunk]
            lg = self.logits(sf)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(lg, sf[:, self.L:], reduction="none")
            outs.append((loss.mean(dim=(1, 2)) / np.log(2)))
        return torch.cat(outs)                               # nats->bits, (B,)


def train_shared_predictor(radius, k, W, T, steps, bs, seed, hid=64, L=4):
    """Train the shared predictor on orbits drawn from the population being searched (random LUTs)."""
    torch.manual_seed(seed)
    m = 1 << k
    model = SharedRowPredictor(L=L, hid=hid).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    g = torch.Generator(device=DEV).manual_seed(seed)
    for s in range(steps):
        lut = sample_luts(bs, m, g)
        st = run_ca(lut, radius, W, T, seed=1000 + s).float()
        opt.zero_grad()
        lg = model.logits(st)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(lg, st[:, L:])
        loss.backward(); opt.step()
    return model


# ---------------------------------------------------------------------------
# Scoring: named-min bpc, general bpc, novelty gap. Cheap GPU gate first.
# ---------------------------------------------------------------------------
def gate_and_named(lut, radius, W, T, aff):
    """Return per-rule named-structure facts (all GPU). 'named' = dead/saturated OR affine OR short-periodic OR chaotic."""
    pmax = max(4, min(24, T // 6))                 # period budget scales with orbit height
    st = run_ca(lut, radius, W, T, seed=7)
    dens = density(st)
    nl = nonlinearity(lut, aff)
    dmg = damage(lut, radius, W, T, seed=11)
    b_iid = bpc_iid(st)
    per_P, b_per = periodicity(st, pmax)
    dead = (dens < 0.03) | (dens > 0.97)
    affine = nl == 0
    periodic = b_per < 0.25                        # a short temporal period (oscillator / simple cyclic)
    chaotic = dmg > 0.40                           # damage saturates -> incompressible noise (class 3)
    ordered = dmg < 0.02                           # damage ~0 -> frozen / trivially-ordered (the expAA LOW-damage cut)
    named = dead | affine | periodic | chaotic | ordered
    named_min_bpc = torch.minimum(b_iid, b_per)    # the best NAMED code length (chaos/affine handled by the gate)
    return st, dict(dens=dens, nl=nl, dmg=dmg, b_iid=b_iid, b_per=b_per, per_P=per_P,
                    dead=dead, affine=affine, periodic=periodic, chaotic=chaotic,
                    named=named, named_min_bpc=named_min_bpc)


def novelty_scores(lut, radius, W, T, aff, neural=None, method="zlib"):
    """Interestingness = the VALIDATED edge-of-chaos complex-finder (expAA/expDD), gated to not-named rules, with a NEW
    neural higher-order-structure tilt:
        interest = 4c(1-c) * exp(-((dmg-0.18)/0.12)^2)        c = general-compressor bits/cell (edge-of-chaos peaks 0.5)
        novelty  = interest * (1 + neural_tilt)                neural_tilt = max(0, (zlib - neural)/zlib): structure the
                                                               cross-population neural model finds that zlib MISSES.
    This ranks class-4 / glider-rich rules on top (NOT frozen/trivial — the raw 'named_min - general' residual mistakenly
    rewarded maximal compressibility). The honest moonshot lever is SCALE + uncatalogued SPACE + the neural axis +
    inspection, not a brand-new selection mechanism."""
    st, f = gate_and_named(lut, radius, W, T, aff)
    B = lut.shape[0]
    keep = ~f["named"]
    nov = torch.zeros(B, device=lut.device)
    gen_bpc = torch.ones(B, device=lut.device)
    if keep.any():
        idx = keep.nonzero(as_tuple=True)[0]
        c = torch.tensor(bpc_compress_batch(st[idx].cpu().numpy(), method), device=lut.device, dtype=torch.float32)
        gen_bpc[idx] = c
        dmg = f["dmg"][idx]
        interest = (4 * c * (1 - c)) * torch.exp(-((dmg - 0.18) / 0.12) ** 2)
        tilt = torch.zeros_like(interest)
        if neural is not None:
            g_neu = neural.bpc(st[idx].float())
            tilt = ((c - g_neu) / c.clamp(min=1e-3)).clamp(0, 1)   # neural beats zlib => higher-order structure
        nov[idx] = interest * (1 + tilt)
    return nov, gen_bpc, f, st


# ---------------------------------------------------------------------------
# Novelty-driven evolutionary hunt over the uncatalogued rule space.
# ---------------------------------------------------------------------------
def evolve(radius, k, W, T, batch, pop, gens, seed, neural=None, out=None):
    aff = affine_tables(k, DEV)
    m = 1 << k                                       # truth-table size = genome length
    g = torch.Generator(device=DEV).manual_seed(seed)

    # seed population: score a big random batch of LUTs, keep the top `pop` by novelty
    pool = sample_luts(batch, m, g)
    nov, gen_bpc, f, _ = novelty_scores(pool, radius, W, T, aff, neural)
    order = torch.argsort(nov, descending=True)
    elite = pool[order[:pop]].clone()
    hist = []
    for gen in range(gens):
        children = mutate_luts(elite, g)             # flip 1-3 truth-table entries per child
        cand = torch.cat([elite, children], dim=0)
        nov_c, gen_c, f_c, _ = novelty_scores(cand, radius, W, T, aff, neural)
        order = torch.argsort(nov_c, descending=True)
        elite = cand[order[:pop]].clone()
        hist.append(float(nov_c[order[0]]))
        print(f"  gen {gen:3d}: best interest {nov_c[order[0]]:.3f} | mean(top20) {nov_c[order[:20]].mean():.3f} | "
              f"complex-survivors {int((nov_c>0).sum())}/{cand.shape[0]}")
    # final: dedup elite, re-score with the STRONGER lzma compressor, characterize top survivors
    elite = dedup_luts(elite)
    nov_u, gen_u, f_u, st_u = novelty_scores(elite, radius, W, T, aff, neural, method="lzma")
    order = torch.argsort(nov_u, descending=True)
    top = order[:min(32, len(order))]
    results = []
    for j in top.tolist():
        results.append(dict(rule=lut_to_int(elite[j]), novelty=float(nov_u[j]), general_bpc=float(gen_u[j]),
                            named_min_bpc=float(f_u["named_min_bpc"][j]), nl=int(f_u["nl"][j]), period=int(f_u["per_P"][j]),
                            density=float(f_u["dens"][j]), damage=float(f_u["dmg"][j])))
    if out:
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "survivors.json"), "w") as fh:
            json.dump(dict(radius=radius, k=k, W=W, T=T, hist=hist, results=results), fh, indent=2)
        if len(top):                                  # save top-8 survivors' space-time for next-session inspection
            keep = elite[order[:min(8, len(order))]]
            stb = run_ca(keep, radius, W, T, seed=7).cpu().numpy()
            np.save(os.path.join(out, "top_spacetime.npy"), stb)
            np.save(os.path.join(out, "top_luts.npy"), keep.cpu().numpy())
    return results, hist


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny config for 4060 correctness check")
    ap.add_argument("--radius", type=int, default=2)
    ap.add_argument("--W", type=int, default=128)
    ap.add_argument("--T", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8192, help="seed random sample size")
    ap.add_argument("--pop", type=int, default=2048)
    ap.add_argument("--gens", type=int, default=30)
    ap.add_argument("--neural", action="store_true", help="also train+use the shared neural general compressor")
    ap.add_argument("--neural-steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    if args.smoke:
        args.W, args.T, args.batch, args.pop, args.gens = 48, 48, 256, 64, 3
        args.neural_steps = 20
    k = 2 * args.radius + 1
    print(f"gpu_exp1_novelty | device={dev_info()} | radius={args.radius} k={k} (rule space 2^{1<<k}) "
          f"W={args.W} T={args.T} batch={args.batch} pop={args.pop} gens={args.gens} neural={args.neural}\n")
    t0 = time.time()

    neural = None
    if args.neural:
        print(f"  training shared neural general-compressor ({args.neural_steps} steps)...")
        neural = train_shared_predictor(args.radius, k, args.W, args.T, args.neural_steps, 32, args.seed)
        print(f"    done [{time.time()-t0:.0f}s]\n")

    results, hist = evolve(args.radius, k, args.W, args.T, args.batch, args.pop, args.gens, args.seed,
                           neural=neural, out=args.out)
    print(f"\n  === TOP COMPLEX SURVIVORS (edge-of-chaos + glider-band damage, not-named, neural-tilted) [{time.time()-t0:.0f}s] ===")
    print("    rule             novelty  gen_bpc  named_bpc  NL  per  dens   damage")
    for r in results[:16]:
        rid = f"0x{r['rule']:08X}" if r['rule'] is not None else "(big-k lut)"
        print(f"    {rid:14s}   {r['novelty']:.3f}   {r['general_bpc']:.3f}    {r['named_min_bpc']:.3f}    "
              f"{r['nl']:2d}  {r['period']:3d}  {r['density']:.2f}   {r['damage']:.3f}")
    print("\n  READ: survivors are CANDIDATES the GPU filtered out of a vast uncatalogued space — compressible by a general")
    print("  compressor yet matching NO named-structure model. Next session: INSPECT top_spacetime.npy (period? gliders?")
    print("  particle types?). Honest ceiling: characterize, do not claim novel. A survivor whose dynamics look genuinely")
    print("  unfamiliar = evidenceable-but-unprovable novelty; a survivor that's 'just class-4' = the ceiling holding.")
