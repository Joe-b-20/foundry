"""
gpu_avida.py — COMPOSABLE-substrate computation evolution: does a smooth fitness landscape unlock the complexity growth that
the self-modifying byte-tape (gpu_metabolism, BFF) could NOT reach?

The arc: pure survival SETTLES (no complexity pressure); computation-coupling on BFF PLATEAUS (stuck at partial NAND ~54%,
never reaching reliable/composed functions = the LANDSCAPE wall — no smooth path to complexity on a byte-tape). HYPOTHESIS:
the obstruction is the SUBSTRATE, not the idea. Avida's open-ended complexity growth depends on COMPOSABLE primitives. This
swaps the substrate for the cleanest composable one — a NAND-COMPLETE STACK MACHINE — where every logic function is built by
COMPOSING nands, so XOR/EQU are reachable by INCREMENTAL program edits (a smooth landscape). Decisive test: does the task
LADDER now climb NOT->AND/OR->XOR->EQU, where BFF stalled at partial NAND?

Substrate: a straight-line stack program (P ops). Inputs a,b (16-bit). Ops: nop, push_a, push_b, push_0, push_1, NAND,
dup, drop. NAND is the only logic primitive — AND/OR/NOT/XOR/EQU must be COMPOSED from it (XOR(a,b) = a few nands; reachable
with push_a/push_b/nand alone, re-pushing inputs). Output = stack top. MERIT = difficulty-weighted match to the logic suite
(target-driven, to test the MECHANISM cleanly). Reproduction = merit-proportional + opcode mutation. (Self-replication was
already shown emergent+settling; the open question here is purely the COMPUTATIONAL substrate's landscape.)

Run:  python gpu_avida.py --smoke
      python gpu_avida.py --N 8192 --P 28 --gens 6000 --out runs/avida   (on the 4090)
"""
from __future__ import annotations
import argparse, time, os, json
import numpy as np
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MASK = 0xFFFF
NOP, PUSH_A, PUSH_B, PUSH_0, PUSH_1, NAND, DUP, DROP = range(8)
NUM_OPS = 8
OPNAMES = ["nop", "a", "b", "0", "1", "nand", "dup", "drop"]
# All non-trivial 2-input boolean functions, weighted by NAND-depth -> a SMOOTH difficulty ladder. The implications
# (a&~b, ~a&b) are the STEPPING STONES: XOR = (a&~b) | (~a&b), so rewarding them gives a gradient from NAND up to XOR/EQU.
FUNCS = [("NOTa", lambda a, b: (~a) & MASK, 1.0), ("NOTb", lambda a, b: (~b) & MASK, 1.0),
         ("NAND", lambda a, b: (~(a & b)) & MASK, 1.0), ("AND", lambda a, b: a & b, 2.0),
         ("a&~b", lambda a, b: a & ((~b) & MASK), 2.0), ("~a&b", lambda a, b: ((~a) & MASK) & b, 2.0),
         ("OR", lambda a, b: a | b, 3.0), ("~a|b", lambda a, b: ((~a) & MASK) | b, 3.0),
         ("a|~b", lambda a, b: a | ((~b) & MASK), 3.0), ("NOR", lambda a, b: (~(a | b)) & MASK, 3.0),
         ("XOR", lambda a, b: a ^ b, 5.0), ("EQU", lambda a, b: (~(a ^ b)) & MASK, 5.0)]


def run_vm(prog, a, b, S):
    """Execute M straight-line stack programs on inputs a,b. prog: (M,P) opcodes; a,b: (M,). Returns output = stack top (M,)."""
    M, P = prog.shape
    stack = torch.zeros((M, S), dtype=torch.long, device=DEV)
    sp = torch.zeros(M, dtype=torch.long, device=DEV)
    ar = torch.arange(M, device=DEV)
    z = torch.zeros(M, dtype=torch.long, device=DEV)
    for p in range(P):
        op = prog[:, p].long()
        top = stack[ar, (sp - 1).clamp(0, S - 1)]
        sec = stack[ar, (sp - 2).clamp(0, S - 1)]
        is_push = ((op >= PUSH_A) & (op <= PUSH_1)) | (op == DUP)
        is_nand = op == NAND
        pushval = torch.where(op == PUSH_A, a, torch.where(op == PUSH_B, b, torch.where(op == PUSH_0, z,
                  torch.where(op == PUSH_1, torch.full_like(z, MASK), torch.where(op == DUP, top,
                  torch.where(op == NAND, (~(top & sec)) & MASK, z))))))
        spdelta = torch.where(is_push, 1, torch.where(is_nand | (op == DROP), -1, 0))
        writepos = torch.where(is_nand, sp - 2, sp).clamp(0, S - 1)
        does_write = is_push | is_nand
        cur = stack[ar, writepos]
        stack[ar, writepos] = torch.where(does_write, pushval, cur)
        sp = (sp + spdelta).clamp(0, S)
    return stack                                         # FULL final stack (M, S) — multi-output (Avida-style)


def evaluate(prog, S, B, g):
    N, P = prog.shape
    a = torch.randint(0, MASK + 1, (N, B), generator=g, device=DEV)
    b = torch.randint(0, MASK + 1, (N, B), generator=g, device=DEV)
    pe = prog[:, None, :].repeat(1, B, 1).reshape(N * B, P)
    out = run_vm(pe, a.reshape(-1), b.reshape(-1), S).reshape(N, B, S)   # (N,B,S) every stack cell is an "output"
    return out, a, b


def merit_logic(out, a, b):
    """Credit an organism for EVERY rewarded function it leaves at ANY stack position (multi-output) -> stepping stones.
    merit = sum_f w_f * max_over_stack_positions( frac of trials where that cell == f(a,b) )."""
    N, B, S = out.shape
    merit = torch.zeros(N, device=out.device)
    fracs = []
    for i, (nm, f, w) in enumerate(FUNCS):
        tgt = f(a, b)[:, :, None]                        # (N,B,1)
        frac_s = (out == tgt).float().mean(dim=1)        # (N,S) accuracy at each stack position
        frac = frac_s.max(dim=1).values                  # (N,) best position for function f
        fracs.append(frac)
        merit = merit + w * frac
    fr = torch.stack(fracs, 1)                            # (N, nfunc)
    which = (fr * torch.tensor([w for _, _, w in FUNCS], device=out.device)).argmax(dim=1)
    return merit, which, fr


def decode(prog_row):
    return " ".join(OPNAMES[int(o)] for o in prog_row)


def evolve(N, P, S, B, gens, seed, out_dir, snap=200):
    os.makedirs(out_dir, exist_ok=True)
    g = torch.Generator(device=DEV).manual_seed(seed)
    prog = torch.randint(0, NUM_OPS, (N, P), generator=g, device=DEV)
    log = []; t0 = time.time()
    for gen in range(gens + 1):
        out, a, b = evaluate(prog, S, B, g)
        merit, which, fracs = merit_logic(out, a, b)
        if gen % snap == 0 or gen == gens:
            bi = int(merit.argmax())
            # ladder: how many organisms RELIABLY (>0.99) compute each function
            ladder = " ".join(f"{FUNCS[i][0]}:{int((fracs[:, i] > 0.99).sum())}" for i in range(len(FUNCS))
                              if int((fracs[:, i] > 0.99).sum()) > 0)
            bestfn = FUNCS[int(which[bi])][0] if which[bi] >= 0 else "-"
            m = dict(gen=gen, best_merit=round(float(merit.max()), 2), mean=round(float(merit.mean()), 3),
                     bestfn=bestfn, ladder=ladder, best_prog=decode(prog[bi].cpu().numpy()))
            log.append(m); json.dump(log, open(os.path.join(out_dir, "avida_log.json"), "w"), indent=1)
            np.save(os.path.join(out_dir, "prog.npy"), prog.cpu().numpy())
            print(f"  gen {gen:5d}: best_merit={m['best_merit']:.2f} mean={m['mean']:.3f} bestfn={bestfn:4s} "
                  f"| ladder[{ladder}] [{time.time()-t0:.0f}s]")
        if gen == gens:
            break
        w = (0.05 + merit).clamp(min=1e-3)
        parents = torch.multinomial(w, N, replacement=True, generator=g)
        prog = prog[parents].clone()
        nm = max(1, int(0.5 * N))                       # mutate ~half the pop, 1 opcode each
        idx = torch.randint(0, N, (nm,), generator=g, device=DEV)
        pos = torch.randint(0, P, (nm,), generator=g, device=DEV)
        prog[idx, pos] = torch.randint(0, NUM_OPS, (nm,), generator=g, device=DEV)
    return log


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--N", type=int, default=8192); ap.add_argument("--P", type=int, default=28)
    ap.add_argument("--S", type=int, default=12); ap.add_argument("--B", type=int, default=16)
    ap.add_argument("--gens", type=int, default=6000); ap.add_argument("--snap", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--out", type=str, default="runs/avida")
    args = ap.parse_args()
    if args.smoke:
        args.N, args.P, args.gens, args.snap = 4096, 24, 400, 50
    print(f"gpu_avida | device={DEV} | NAND-complete stack machine | N={args.N} P={args.P} S={args.S} B={args.B} gens={args.gens}\n")
    print("  WATCH the LADDER: NOT/AND/OR (easy) then NAND then XOR/EQU (must COMPOSE nands). Climbing to XOR/EQU = the")
    print("  composable substrate UNLOCKS complexity growth where BFF plateaued. INSPECT best_prog for the evolved expression.\n")
    evolve(args.N, args.P, args.S, args.B, args.gens, args.seed, args.out, args.snap)
