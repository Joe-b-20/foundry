"""
gpu_metabolism.py — COMPUTATION-COUPLED evolution: does rewarding COMPUTATION drive the open-ended complexity growth that
pure survival-selection LACKED? (The Avida insight, pointed at the moonshot.)

The deep-dive (gpu_alife) showed pure survival/replication selection produces a stable replicator and SETTLES — no
complexity growth (the fastest copier wins; nothing pressures MORE complexity). The fix (Avida; Lenski et al. 2003): give
organisms a computational METABOLISM — replication merit scales with performing non-trivial INPUT->OUTPUT computation — and
complex features evolve via genuinely novel paths. This tests it on the project's self-modifying-program substrate.

Organism = a BFF tape, layout [ CODE | INPUT(a,b) | OUTPUT ]. Each generation: inject B random inputs, run the organism
(BFF, it must navigate to read input + write output), read its output. MERIT (two modes):
  --merit logic     : reward matching a SUITE of logic functions of (a,b), difficulty-weighted (NOT<AND/OR<XOR/EQU). Climbing
                      this ladder = the Avida complexity-growth result. (Uses targets -> tests the MECHANISM, not target-free.)
  --merit intrinsic : TARGET-FREE — reward output that is a non-trivial, input-DEPENDENT, non-copy transform (the bridge,
                      applied to the I/O map). Riskier (can reward chaos); the honest moonshot version.
Reproduction is merit-proportional (organisms that compute replicate more) + mutation. We watch whether merit/complexity
GROWS over generations (open-ended) or plateaus, and INSPECT what procedures evolve (novel?).

Run:  python gpu_metabolism.py --smoke
      python gpu_metabolism.py --merit logic --N 8192 --gens 4000 --out runs/metab_logic     (on the 4090)
"""
from __future__ import annotations
import argparse, time, os, json
import numpy as np
import torch
from gpu_weird_soup import run_programs, DEV

OPS = "<>{}-+.,[]"
# logic functions of two input bytes a,b (bitwise), with difficulty weights (XOR/EQU need composition = harder)
FUNCS = [("NOTa", lambda a, b: (~a) & 255, 1.0), ("AND", lambda a, b: a & b, 2.0), ("OR", lambda a, b: a | b, 2.0),
         ("NAND", lambda a, b: (~(a & b)) & 255, 4.0), ("NOR", lambda a, b: (~(a | b)) & 255, 4.0),
         ("XOR", lambda a, b: a ^ b, 5.0), ("EQU", lambda a, b: (~(a ^ b)) & 255, 5.0)]


def decode(row):
    return "".join(OPS[b & 15] if (b & 15) < 10 else "." for b in row)


def evaluate(pop, CL, K, B, g):
    """Run each of N organisms on B random (a,b) inputs; return outputs (N,B) and the inputs a,b (N,B)."""
    N, L = pop.shape
    a = torch.randint(0, 256, (N, B), generator=g, device=DEV)
    b = torch.randint(0, 256, (N, B), generator=g, device=DEV)
    tapes = pop[:, None, :].repeat(1, B, 1).reshape(N * B, L).clone()
    tapes[:, CL] = a.reshape(-1).to(torch.uint8)            # INPUT a at cell CL
    tapes[:, CL + 1] = b.reshape(-1).to(torch.uint8)        # INPUT b at cell CL+1
    tapes = run_programs(tapes, K)
    out = tapes[:, CL + 2].reshape(N, B).long()             # OUTPUT at cell CL+2
    return out, a, b


def merit_logic(out, a, b):
    """max over logic functions of (difficulty-weight x match-fraction), + a small input-dependence base to bootstrap."""
    N, B = out.shape
    base = 0.3 * (out.float().std(dim=1) / 255.0)           # rewards output varying with input (bootstraps the gradient)
    best = torch.zeros(N, device=out.device); which = torch.full((N,), -1, device=out.device, dtype=torch.long)
    for i, (name, f, w) in enumerate(FUNCS):
        frac = (out == f(a, b)).float().mean(dim=1)          # consistency of output == f(a,b) across trials
        score = w * frac
        upd = score > best
        best = torch.where(upd, score, best); which = torch.where(upd, torch.full_like(which, i), which)
    return base + best, which


def merit_intrinsic(out, a, b):
    """TARGET-FREE: reward output that DEPENDS on input (varies) and is NOT a copy of input. (Chaos-risk noted.)"""
    spread = out.float().std(dim=1) / 255.0                 # input-dependence
    copy = ((out == a) | (out == b)).float().mean(dim=1)    # penalize echoing an input
    return spread * (1 - copy), torch.full((out.shape[0],), -1, device=out.device, dtype=torch.long)


def mutate(pop, CL, g, kflip=2):
    N, L = pop.shape
    out = pop.clone()
    for _ in range(kflip):
        pos = torch.randint(0, CL, (N,), generator=g, device=DEV)    # mutate CODE region only (not the I/O cells)
        val = torch.randint(0, 256, (N,), generator=g, device=DEV, dtype=torch.uint8)
        out[torch.arange(N, device=DEV), pos] = val
    return out


def evolve(N, CL, K, B, gens, mode, seed, out_dir, snap=200):
    L = CL + 3                                              # code + a + b + output
    os.makedirs(out_dir, exist_ok=True)
    g = torch.Generator(device=DEV).manual_seed(seed)
    pop = torch.randint(0, 256, (N, L), generator=g, device=DEV, dtype=torch.uint8)
    meritfn = merit_logic if mode == "logic" else merit_intrinsic
    log = []; t0 = time.time()
    for gen in range(gens + 1):
        out, a, b = evaluate(pop, CL, K, B, g)
        merit, which = meritfn(out, a, b)
        if gen % snap == 0 or gen == gens:
            bi = int(merit.argmax())
            fname = FUNCS[int(which[bi])][0] if which[bi] >= 0 else "-"
            # how many organisms reliably compute each function (logic mode)
            ladder = ""
            if mode == "logic":
                for i, (nm, f, w) in enumerate(FUNCS):
                    nrel = int(((out == f(a, b)).float().mean(dim=1) > 0.9).sum())
                    if nrel: ladder += f"{nm}:{nrel} "
            m = dict(gen=gen, best_merit=round(float(merit.max()), 3), mean_merit=round(float(merit.mean()), 3),
                     best_func=fname, ladder=ladder.strip(), best_code=decode(pop[bi].cpu().numpy()))
            log.append(m); json.dump(log, open(os.path.join(out_dir, "metab_log.json"), "w"), indent=1)
            np.save(os.path.join(out_dir, "pop.npy"), pop.cpu().numpy())
            print(f"  gen {gen:5d}: best_merit={m['best_merit']:.2f} mean={m['mean_merit']:.3f} bestfn={fname:4s} "
                  f"| ladder[{ladder.strip()}] [{time.time()-t0:.0f}s]")
        if gen == gens:
            break
        # merit-proportional reproduction + mutation
        w = (0.05 + merit).clamp(min=1e-3)
        parents = torch.multinomial(w, N, replacement=True, generator=g)
        pop = mutate(pop[parents], CL, g)
    return log


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--merit", choices=["logic", "intrinsic"], default="logic")
    ap.add_argument("--N", type=int, default=8192); ap.add_argument("--CL", type=int, default=29)
    ap.add_argument("--K", type=int, default=120); ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--gens", type=int, default=3000); ap.add_argument("--snap", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--out", type=str, default="runs/metab")
    args = ap.parse_args()
    if args.smoke:
        args.N, args.CL, args.K, args.gens, args.snap = 2048, 29, 100, 300, 50
    print(f"gpu_metabolism | device={DEV} | merit={args.merit} N={args.N} CL={args.CL} K={args.K} B={args.B} gens={args.gens}\n")
    print("  WATCH (logic): the LADDER climbing NOT->AND/OR->XOR/EQU = computation-coupling DRIVES complexity growth (what")
    print("  pure survival lacked). Plateau at trivial = it doesn't bootstrap. Then INSPECT best_code for the evolved procedure.\n")
    evolve(args.N, args.CL, args.K, args.B, args.gens, args.merit, args.seed, args.out, args.snap)
