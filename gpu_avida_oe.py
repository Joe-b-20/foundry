"""
gpu_avida_oe.py — THE TARGET-FREE MOONSHOT SWING: open-ended evolution of INTERESTING computation on a composable substrate.

gpu_avida proved a composable substrate UNLOCKS complexity growth — but with a logic-TASK reward (functions named). This is
the target-free version: reward computation that is INTRINSICALLY interesting (structured but non-trivial), naming NO target,
and see if evolution discovers a NOVEL function/procedure. Two ingredients make novelty even POSSIBLE here:
  (1) a RICH op set with CROSS-BIT ops (shift, add, sub — carry propagation mixes bits), so the organism can compute far more
      than the 16 bitwise-boolean functions; the function space is huge and largely un-named.
  (2) a TARGET-FREE merit = the project's validated EDGE-OF-CHAOS bridge signal, applied to the evolved function's I/O map:
      probe the organism on a grid of (a,b), and reward output grids that are STRUCTURED but not TRIVIAL — 4c(1-c) on the
      grid's zlib ratio (trivial constant/projection -> too compressible -> 0; chaos/pseudorandom -> incompressible -> 0;
      structured-rich -> intermediate -> high), times a depends-on-BOTH-inputs factor. + a NOVELTY term (different from the
      population's functions) to keep diversity and explore the space.
Then INSPECT the top organisms: WHAT did they evolve to compute? Honest test: known structure (+, ^, shift-mixes) = ceiling
holds; a structured map that resists identification = evidenceable-but-unprovable novelty.

Run:  python gpu_avida_oe.py --selftest
      python gpu_avida_oe.py --smoke
      python gpu_avida_oe.py --N 8192 --gens 4000 --out runs/avida_oe    (on the 4090)
"""
from __future__ import annotations
import argparse, time, os, json, zlib
import numpy as np
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MASK = 0xFFFF
# 0 nop 1 push_a 2 push_b 3 push_1 4 push_0 | 5 nand 6 and 7 or 8 xor 9 add 10 sub | 11 shl 12 shr | 13 dup 14 swap 15 drop
NOP, PA, PB, P1, P0, NAND, AND, OR, XOR, ADD, SUB, SHL, SHR, DUP, SWAP, DROP = range(16)
NUM_OPS = 16
OPN = ["nop", "a", "b", "1", "0", "nand", "and", "or", "xor", "add", "sub", "shl", "shr", "dup", "swap", "drop"]


def run_vm(prog, a, b, S):
    """Rich stack VM. prog (M,P) opcodes; a,b (M,). Returns FULL final stack (M,S)."""
    M, P = prog.shape
    st = torch.zeros((M, S), dtype=torch.long, device=DEV); sp = torch.zeros(M, dtype=torch.long, device=DEV)
    ar = torch.arange(M, device=DEV); z = torch.zeros(M, dtype=torch.long, device=DEV)
    for p in range(P):
        op = prog[:, p].long()
        top = st[ar, (sp - 1).clamp(0, S - 1)]; sec = st[ar, (sp - 2).clamp(0, S - 1)]
        binres = torch.where(op == NAND, (~(top & sec)) & MASK, torch.where(op == AND, top & sec,
                 torch.where(op == OR, top | sec, torch.where(op == XOR, top ^ sec,
                 torch.where(op == ADD, (top + sec) & MASK, torch.where(op == SUB, (sec - top) & MASK, z))))))
        unres = torch.where(op == SHL, (top << 1) & MASK, torch.where(op == SHR, top >> 1, top))
        pushval = torch.where(op == PA, a, torch.where(op == PB, b, torch.where(op == P1, torch.full_like(z, MASK),
                  torch.where(op == P0, z, torch.where(op == DUP, top, z)))))
        is_bin = (op >= NAND) & (op <= SUB)
        is_un = (op == SHL) | (op == SHR)
        is_push = ((op >= PA) & (op <= P0)) | (op == DUP)
        # main write: bin->sp-2, un->sp-1, push->sp, swap->sp-1(=sec)
        wpos = torch.where(is_bin, sp - 2, torch.where(is_un, sp - 1, torch.where(op == SWAP, sp - 1, sp))).clamp(0, S - 1)
        wval = torch.where(is_bin, binres, torch.where(is_un, unres, torch.where(op == SWAP, sec, pushval)))
        does = is_bin | is_un | is_push | (op == SWAP)
        st[ar, wpos] = torch.where(does, wval, st[ar, wpos])
        # swap second write: sp-2 <- old top
        w2 = (sp - 2).clamp(0, S - 1)
        st[ar, w2] = torch.where(op == SWAP, top, st[ar, w2])
        spdelta = torch.where(is_push, 1, torch.where(is_bin | (op == DROP), -1, 0))
        sp = (sp + spdelta).clamp(0, S)
    return st


def selftest():
    print("=== rich VM self-test ===")
    S = 8
    def run1(ops, a, b):
        return run_vm(torch.tensor([ops + [NOP] * (12 - len(ops))], device=DEV), torch.tensor([a], device=DEV),
                      torch.tensor([b], device=DEV), S)
    t = run1([PA, PB, ADD], 1000, 337); okA = int(t[0].max()) == 1337  # add somewhere on stack
    t = run1([PA, PB, XOR], 0xF0F0, 0x0FF0); v = (0xF0F0 ^ 0x0FF0); okB = (t[0] == v).any().item()
    t = run1([PA, SHL], 0x0081, 0); okC = (t[0] == 0x0102).any().item()
    t = run1([PA, PB, SUB], 500, 200); okD = (t[0] == 300).any().item()  # top-sec = a-b
    t = run1([PA, PB, SWAP, SUB], 200, 500); okE = (t[0] == 300).any().item()  # swap -> b-a=300
    for nm, ok in [("a b add=1337", okA), ("a b xor", okB), ("a shl", okC), ("a b sub", okD), ("swap sub", okE)]:
        print(f"  {nm:16s} {'OK' if ok else 'FAIL'}")
    allok = okA and okB and okC and okD and okE
    print(f"  => VM {'CORRECT' if allok else 'BROKEN'}")
    return allok


def probe(prog, S, G=16):
    """Run each organism on a GxG grid of (a,b). Returns output grids (N, G, G) (low byte)."""
    N, P = prog.shape
    vals = (torch.arange(G, device=DEV) * (MASK // G)).long()
    A = vals[:, None].expand(G, G).reshape(-1); B = vals[None, :].expand(G, G).reshape(-1)   # (G*G,)
    a = A[None, :].expand(N, G * G).reshape(-1); b = B[None, :].expand(N, G * G).reshape(-1)
    pe = prog[:, None, :].expand(N, G * G, P).reshape(N * G * G, P)
    st = run_vm(pe, a, b, S)
    out = st[:, S - 1].reshape(N, G, G)                  # use stack TOP as the organism's "output"
    return out


def merit_oe(prog, S, pop_sigs=None):
    """TARGET-FREE: edge-of-chaos structure of the output grid x depends-on-both-inputs x novelty."""
    out = probe(prog, S)                                 # (N,G,G)
    N, G, _ = out.shape
    grids = (out & 0xFF).to(torch.uint8).cpu().numpy()   # low byte grid per organism
    c = np.array([len(zlib.compress(g.tobytes(), 6)) / (G * G) for g in grids])
    edge = torch.tensor(4 * c * (1 - c), device=DEV, dtype=torch.float32)
    of = out.float()
    dep_a = (of.std(dim=1).mean(dim=1) / MASK * 3).clamp(max=1.0)    # varies along a-axis (smooth, scaled)
    dep_b = (of.std(dim=2).mean(dim=1) / MASK * 3).clamp(max=1.0)    # varies along b-axis
    # SMOOTH bootstrap: (dep_a+dep_b) rewards dependence on EITHER input (random progs can reach 1-input funcs);
    # the (0.3+edge) factor then pulls toward STRUCTURED maps; structured 2-input funcs (richer grids) win on edge.
    merit = (dep_a + dep_b) * (0.3 + edge) * (1 + dep_a * dep_b)    # last factor favours depending on BOTH
    sig = (out.reshape(N, -1) & 0xFF).float()                       # behavioral signature (the function's outputs)
    nov = torch.ones(N, device=DEV)
    if pop_sigs is not None and len(pop_sigs):
        # novelty = mean distance to a sample of recent signatures (cheap kNN-ish)
        ref = pop_sigs[torch.randint(0, len(pop_sigs), (min(64, len(pop_sigs)),), device=DEV)]
        d = (sig[:, None, :] - ref[None, :, :]).abs().mean(dim=2).mean(dim=1) / 255
        nov = d.clamp(max=0.5) * 2
    return merit * (0.5 + 0.5 * nov), edge, dep_a * dep_b, sig


def decode(row):
    return " ".join(OPN[int(o)] for o in row)


# 2026-06-09 closure version (audit §1.3): the original named suite had 10 entries —
# the expX "5-entry label list" artifact recurring at the load-bearing position. This
# suite has ~60 functions, evaluated as EXACT match on 64 probe pairs (8 structured +
# 56 pseudo-random), so a match is essentially a proof and a non-match is meaningful.
def _named_suite(aa, bb):
    M = MASK
    f = {}
    f["0"] = torch.zeros_like(aa); f["0xFFFF"] = torch.full_like(aa, M)
    f["a"] = aa; f["b"] = bb; f["~a"] = (~aa) & M; f["~b"] = (~bb) & M
    f["a<<1"] = (aa << 1) & M; f["a>>1"] = aa >> 1; f["b<<1"] = (bb << 1) & M; f["b>>1"] = bb >> 1
    f["a+1"] = (aa + 1) & M; f["a-1"] = (aa - 1) & M; f["b+1"] = (bb + 1) & M; f["b-1"] = (bb - 1) & M
    f["a&b"] = aa & bb; f["a|b"] = aa | bb; f["a^b"] = aa ^ bb
    f["~(a&b)"] = (~(aa & bb)) & M; f["~(a|b)"] = (~(aa | bb)) & M; f["~(a^b)"] = (~(aa ^ bb)) & M
    f["a&~b"] = aa & (~bb) & M; f["~a&b"] = (~aa) & bb & M; f["a|~b"] = (aa | ((~bb) & M)) & M; f["~a|b"] = (((~aa) & M) | bb) & M
    f["a+b"] = (aa + bb) & M; f["a-b"] = (aa - bb) & M; f["b-a"] = (bb - aa) & M
    f["a+b+1"] = (aa + bb + 1) & M; f["a-b-1"] = (aa - bb - 1) & M
    f["2a"] = (2 * aa) & M; f["2b"] = (2 * bb) & M; f["3a"] = (3 * aa) & M; f["3b"] = (3 * bb) & M
    f["2a+b"] = (2 * aa + bb) & M; f["a+2b"] = (aa + 2 * bb) & M; f["2a+2b"] = (2 * (aa + bb)) & M
    f["2a-b"] = (2 * aa - bb) & M; f["2b-a"] = (2 * bb - aa) & M
    f["(a+b)>>1"] = ((aa + bb) & M) >> 1; f["(a^b)>>1"] = (aa ^ bb) >> 1
    f["(a&b)<<1"] = ((aa & bb) << 1) & M                       # the carry vector
    f["(a+b)^a"] = ((aa + bb) & M) ^ aa; f["(a+b)^b"] = ((aa + bb) & M) ^ bb
    f["(a+b)^a^b"] = ((aa + bb) & M) ^ aa ^ bb                 # carry-propagation pattern
    f["a+(a&b)"] = (aa + (aa & bb)) & M; f["a+(a|b)"] = (aa + (aa | bb)) & M
    f["a+(a^b)"] = (aa + (aa ^ bb)) & M; f["(a|b)+(a&b)"] = ((aa | bb) + (aa & bb)) & M   # = a+b identity
    f["(a|b)-(a&b)"] = ((aa | bb) - (aa & bb)) & M             # = a^b identity
    f["a-(a&b)"] = (aa - (aa & bb)) & M                        # = a&~b identity
    f["max"] = torch.maximum(aa, bb); f["min"] = torch.minimum(aa, bb)
    f["|a-b|"] = (torch.maximum(aa, bb) - torch.minimum(aa, bb)) & M
    f["a*b"] = (aa * bb) & M; f["a*a"] = (aa * aa) & M; f["b*b"] = (bb * bb) & M
    f["~(a+b)"] = (~(aa + bb)) & M; f["-(a+b)"] = (-(aa + bb)) & M; f["-a"] = (-aa) & M; f["-b"] = (-bb) & M
    f["a^(b<<1)"] = aa ^ ((bb << 1) & M); f["a^(b>>1)"] = aa ^ (bb >> 1)
    f["(a^b)<<1"] = ((aa ^ bb) << 1) & M
    # loop-reachable extensions (added for gpu_avida_loop — repeated shl/add composites,
    # so the loop arm's obvious discoveries still count as "named")
    f["a<<2"] = (aa << 2) & M; f["a<<3"] = (aa << 3) & M; f["a<<4"] = (aa << 4) & M
    f["b<<2"] = (bb << 2) & M; f["b<<3"] = (bb << 3) & M
    f["a>>2"] = aa >> 2; f["a>>3"] = aa >> 3; f["b>>2"] = bb >> 2
    f["4a"] = (4 * aa) & M; f["5a"] = (5 * aa) & M; f["6a"] = (6 * aa) & M
    f["8a"] = (8 * aa) & M; f["4b"] = (4 * bb) & M; f["8b"] = (8 * bb) & M
    f["4a+b"] = (4 * aa + bb) & M; f["a+4b"] = (aa + 4 * bb) & M
    f["3a+b"] = (3 * aa + bb) & M; f["a+3b"] = (aa + 3 * bb) & M; f["3a+3b"] = (3 * (aa + bb)) & M
    f["(a+b)<<2"] = ((aa + bb) << 2) & M; f["(a^b)<<2"] = ((aa ^ bb) << 2) & M
    return f


_PROBE_CACHE = None

def _probes():
    global _PROBE_CACHE
    if _PROBE_CACHE is None:
        rng = np.random.default_rng(12345)
        ra = rng.integers(0, MASK + 1, 56); rb = rng.integers(0, MASK + 1, 56)
        aa = torch.tensor([3, 100, 255, 1000, 40000, 12345, 7, 65535] + list(ra), device=DEV)
        bb = torch.tensor([5, 7, 200, 999, 25000, 11111, 3, 1] + list(rb), device=DEV)
        _PROBE_CACHE = (aa, bb)
    return _PROBE_CACHE


def describe(prog_row, S):
    """Characterize what one organism computes: exact-match against the ~60-entry named
    suite on 64 probe pairs, plus the nearest-named bit-similarity for non-matches."""
    p = torch.tensor(prog_row[None], device=DEV)
    aa, bb = _probes()
    out = run_vm(p.expand(len(aa), -1), aa, bb, S)[:, S - 1]
    known = _named_suite(aa, bb)
    match = [k for k, v in known.items() if bool((out == v).all())]
    # graded view for the unmatched: nearest named function by per-bit agreement
    best_nm, best_sim = "", 0.0
    if not match:
        ob = ((out[:, None] >> torch.arange(16, device=DEV)[None, :]) & 1)
        for k, v in known.items():
            vb = ((v[:, None] >> torch.arange(16, device=DEV)[None, :]) & 1)
            sim = float((ob == vb).float().mean())
            if sim > best_sim:
                best_sim, best_nm = sim, k
    pairs = " ".join(f"{int(x)},{int(y)}->{int(o)}" for x, y, o in list(zip(aa, bb, out))[:5])
    return match, pairs, best_nm, round(best_sim, 3)


def evolve(N, P, S, gens, seed, out_dir, snap=10, topk=5):
    """2026-06-09 closure version (audit §1.3): snapshot every `snap` gens (default 10,
    was 200), describe the top-`topk` organisms (not just the best), persist every
    snapshot, and maintain a FIRST-MATCH table (named function -> first gen at which
    any top-k organism exactly matched it on 64 probes) — the waypoint evidence the
    original run lacked."""
    os.makedirs(out_dir, exist_ok=True)
    g = torch.Generator(device=DEV).manual_seed(seed)
    prog = torch.randint(0, NUM_OPS, (N, P), generator=g, device=DEV)
    log = []; first_match = {}; t0 = time.time(); sigs = None
    for gen in range(gens + 1):
        merit, edge, nontriv, sig = merit_oe(prog, S, sigs)
        sigs = sig[torch.randperm(N, device=DEV)[:256]]                # archive a sample of behaviors for novelty
        if gen % snap == 0 or gen == gens:
            top = torch.argsort(merit, descending=True)[:topk].cpu().numpy()
            tops = []
            for r, bi in enumerate(top):
                match, pairs, near, sim = describe(prog[bi].cpu().numpy(), S)
                for k in match:
                    first_match.setdefault(k, gen)
                tops.append(dict(rank=r, merit=round(float(merit[bi]), 3), edge=round(float(edge[bi]), 3),
                                 match=match, nearest=near, near_sim=sim, pairs=pairs,
                                 prog=decode(prog[bi].cpu().numpy())))
            m = dict(gen=gen, best=round(float(merit.max()), 3), mean=round(float(merit.mean()), 3),
                     best_edge=tops[0]["edge"], match=tops[0]["match"], pairs=tops[0]["pairs"],
                     top=tops, first_match=dict(first_match))
            log.append(m)
            json.dump(log, open(os.path.join(out_dir, "oe_log.json"), "w"), indent=1)
            np.save(os.path.join(out_dir, "prog.npy"), prog.cpu().numpy())
            neartxt = "" if tops[0]["match"] else f" nearest={tops[0]['nearest']}@{tops[0]['near_sim']}"
            print(f"  gen {gen:5d}: best={m['best']:.3f} mean={m['mean']:.3f} edge={m['best_edge']:.2f} "
                  f"match={tops[0]['match']}{neartxt} | first_match={dict(first_match)} [{time.time()-t0:.0f}s]")
        if gen == gens:
            break
        w = (0.02 + merit).clamp(min=1e-3)
        parents = torch.multinomial(w, N, replacement=True, generator=g)
        prog = prog[parents].clone()
        nm = max(1, int(0.5 * N)); idx = torch.randint(0, N, (nm,), generator=g, device=DEV)
        pos = torch.randint(0, P, (nm,), generator=g, device=DEV)
        prog[idx, pos] = torch.randint(0, NUM_OPS, (nm,), generator=g, device=DEV)
    return log


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true"); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--N", type=int, default=8192); ap.add_argument("--P", type=int, default=24)
    ap.add_argument("--S", type=int, default=12); ap.add_argument("--gens", type=int, default=3000)
    ap.add_argument("--snap", type=int, default=10); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--out", type=str, default="runs/avida_oe")
    args = ap.parse_args()
    if args.selftest or args.smoke:
        if not selftest() and args.smoke:
            raise SystemExit("VM broken")
    if args.selftest and not args.smoke:
        raise SystemExit(0)
    if args.smoke:
        args.N, args.P, args.gens, args.snap = 2048, 20, 200, 50
    print(f"\ngpu_avida_oe | device={DEV} | TARGET-FREE edge-of-chaos | N={args.N} P={args.P} gens={args.gens}\n")
    print("  WATCH: which functions evolve target-free (match=[...]). KNOWN (+,^,shift) = ceiling holds; an unidentified")
    print("  structured map = evidenceable-but-unprovable novelty. Then INSPECT runs_pod/avida_oe top organisms.\n")
    evolve(args.N, args.P, args.S, args.gens, args.seed, args.out, args.snap, args.topk)
