"""
expEE_evolve.py — the expDD lesson applied to the PROGRAM space: SAMPLING failed to find deep computation (deep machines
are astronomically rare among random tiny TMs — expEE_logicaldepth.py found a best halter of only ~34 steps in 25k samples),
so DRIVE the search instead. Use computational DEPTH (runtime to halt = Bennett logical depth / the Busy-Beaver objective)
as an evolutionary fitness and ask: does an intrinsic-DEPTH-driven search discover deep, structured computation that
sampling could not? This is the generative-bridge result (expDD) on Turing machines, and it is literally evolutionary
Busy-Beaver hunting (known to work — so the test is whether DRIVING beats SAMPLING by a large, clean margin, and whether the
deep machines found are STRUCTURED, target-free).

Fitness = runtime if the machine HALTS within T_max (else 0). Mutation = perturb a few transition-table entries. Controls:
the sampling baseline's best halter, and a RANDOM-fitness driver (does evolution-itself, not depth, find deep halters?).
Run: python expEE_evolve.py
"""
from __future__ import annotations
import argparse, math, random, zlib
import numpy as np
from expEE_logicaldepth import random_tm, run_tm, comp_ratio, describe, HALT


def fitness(T, T_max, W, mode):
    halted, runtime, rows, nz = run_tm(T, T_max, W)
    if mode == "random":
        # deterministic pseudo-random fitness from the table bits (control: evolution itself, not depth)
        h = 0
        for s in range(len(T)):
            for b in (0, 1):
                w, mv, nx = T[s][b]; h = (h * 131 + w * 7 + (mv + 1) * 3 + (nx + 2)) & 0xFFFFFFFF
        return (h / 0xFFFFFFFF), halted, runtime, nz
    # DEPTH driver: reward long FINITE computation; non-halting (incl. ran-off-tape) scores 0
    return (runtime if halted else 0), halted, runtime, nz


def mutate(T, n, rng):
    T2 = [row[:] for row in T]
    for _ in range(rng.randint(1, 2)):
        s = rng.randint(0, n - 1); b = rng.randint(0, 1)
        w = rng.randint(0, 1); mv = rng.choice((-1, 1))
        nx = HALT if rng.random() < 1.0 / (2 * n) else rng.randint(0, n - 1)
        T2[s][b] = (w, mv, nx)
    return T2


def evolve(n, T_max, W, mode, pop=60, gens=60, seed=0):
    rng = random.Random(seed)
    population = [random_tm(n, rng) for _ in range(pop)]
    scored = [(fitness(T, T_max, W, mode), T) for T in population]
    best_trace = []
    for g in range(gens):
        scored.sort(key=lambda x: -x[0][0])
        best_trace.append(scored[0][0])
        elite = [T for _, T in scored[:max(2, pop // 5)]]
        children = [T for T in elite]
        while len(children) < pop:
            children.append(mutate(elite[rng.randint(0, len(elite) - 1)], n, rng))
        scored = [(fitness(T, T_max, W, mode), T) for T in children]
    scored.sort(key=lambda x: -x[0][0])
    return scored[0][1], scored[0][0], best_trace


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--Tmax", type=int, default=8000)
    ap.add_argument("--W", type=int, default=801)
    ap.add_argument("--pop", type=int, default=60)
    ap.add_argument("--gens", type=int, default=60)
    args = ap.parse_args()
    n, T_max, W = args.n, args.Tmax, args.W
    print(f"DEPTH-DRIVEN evolution of {n}-state TMs (evolutionary Busy-Beaver / logical-depth hunt). T_max={T_max}.\n")

    # sampling baseline (the same budget as one evolutionary run, for a fair SAMPLE-vs-DRIVE comparison)
    rng = random.Random(123)
    budget = args.pop * args.gens
    best_sample = 0
    for _ in range(budget):
        halted, runtime, rows, nz = run_tm(random_tm(n, rng), T_max, W)
        if halted and runtime > best_sample:
            best_sample = runtime
    print(f"  SAMPLING baseline ({budget} random machines, same budget as one ev run): deepest HALTER = {best_sample} steps\n")

    for mode in ("depth", "random"):
        bestT, (score, halted, runtime, nz), trace = evolve(n, T_max, W, mode, pop=args.pop, gens=args.gens, seed=1)
        tr = [int(s[0]) if isinstance(s, tuple) else int(s) for s in trace]
        print(f"  DRIVER={mode:6s}: best HALTER runtime={runtime if halted else 0:5d}  (halted={halted}, nz={nz})")
        if mode == "depth":
            halted2, rt, nz2, span, block, period = describe(bestT, T_max, W)
            c = comp_ratio(run_tm(bestT, T_max, W)[2])
            print(f"            depth-trace (best halter runtime by gen): {tr[::max(1,len(tr)//10)]}")
            print(f"            the evolved DEEP halter: runtime={rt} steps, {nz2} cells written, space-time comp={c:.3f}")
            print(f"            final tape block: [{block}]")
    print("\n  READ: if DEPTH-driven evolution finds halters running ORDERS OF MAGNITUDE longer than sampling (and longer than")
    print("  the RANDOM-fitness control), the intrinsic depth signal DRIVES discovery of deep computation that sampling can't")
    print("  reach — the program-space analog of expDD (sample fails -> drive works). Honest: evolutionary Busy-Beaver search")
    print("  is known to work; the point here is the SAMPLE-vs-DRIVE gap on the iterative-program space, not a novel machine.")
