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


def describe(prog_row, S):
    """Characterize what one organism computes: its output on sample (a,b), and which KNOWN function (if any) it matches."""
    p = torch.tensor(prog_row[None], device=DEV)
    aa = torch.tensor([3, 100, 255, 1000, 40000, 12345, 7, 65535], device=DEV)
    bb = torch.tensor([5, 7, 200, 999, 25000, 11111, 3, 1], device=DEV)
    out = run_vm(p.expand(len(aa), -1), aa, bb, S)[:, S - 1]
    known = {"a+b": (aa + bb) & MASK, "a-b": (aa - bb) & MASK, "a^b": aa ^ bb, "a&b": aa & bb, "a|b": aa | bb,
             "~(a&b)": (~(aa & bb)) & MASK, "a": aa, "b": bb, "a<<1": (aa << 1) & MASK, "(a+b)^a": ((aa + bb) & MASK) ^ aa}
    match = [k for k, v in known.items() if bool((out == v).all())]
    pairs = " ".join(f"{int(x)},{int(y)}->{int(o)}" for x, y, o in list(zip(aa, bb, out))[:5])
    return match, pairs


def evolve(N, P, S, gens, seed, out_dir, snap=200):
    os.makedirs(out_dir, exist_ok=True)
    g = torch.Generator(device=DEV).manual_seed(seed)
    prog = torch.randint(0, NUM_OPS, (N, P), generator=g, device=DEV)
    log = []; t0 = time.time(); sigs = None
    for gen in range(gens + 1):
        merit, edge, nontriv, sig = merit_oe(prog, S, sigs)
        sigs = sig[torch.randperm(N, device=DEV)[:256]]                # archive a sample of behaviors for novelty
        if gen % snap == 0 or gen == gens:
            bi = int(merit.argmax())
            match, pairs = describe(prog[bi].cpu().numpy(), S)
            m = dict(gen=gen, best=round(float(merit.max()), 3), mean=round(float(merit.mean()), 3),
                     best_edge=round(float(edge[bi]), 3), match=match, pairs=pairs, best_prog=decode(prog[bi].cpu().numpy()))
            log.append(m); json.dump(log, open(os.path.join(out_dir, "oe_log.json"), "w"), indent=1)
            np.save(os.path.join(out_dir, "prog.npy"), prog.cpu().numpy())
            print(f"  gen {gen:5d}: best={m['best']:.3f} mean={m['mean']:.3f} edge={m['best_edge']:.2f} "
                  f"match={match} | {pairs[:50]} [{time.time()-t0:.0f}s]")
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
    ap.add_argument("--snap", type=int, default=200); ap.add_argument("--seed", type=int, default=1)
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
    evolve(args.N, args.P, args.S, args.gens, args.seed, args.out, args.snap)
