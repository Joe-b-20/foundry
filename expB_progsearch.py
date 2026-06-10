"""
expB_progsearch.py — Exp B: gradient-free program synthesis over a minimal
digit-serial register VM, for addition. The discovered program IS the algorithm.

VM model (the symbolic analog of Exp A's neural Mealy machine):
- Process digit positions LSB-first. A persistent register C threads across
  positions (init 0). We do NOT call it carry or tell the search what it's for.
- Per position, a straight-line program computes from inputs (A=a_t, B=b_t, C):
  an OUTPUT digit and a NEXT value of C. Same program at every position
  => any correct program length-generalizes by construction.
- Primitives: add, sub, mul, intdiv, mod, ge, min, max, plus constants {0,1,base}.
  Carry is NOT a primitive; the search must discover it (e.g. C' = (A+B+C) div base).

Search: simple genetic programming (tournament selection + mutation + random
restarts). Fitness = per-position output accuracy (dense signal). Final report
is EXACT whole-number length-generalization via core_data (no partial credit).

Pure Python — runs anywhere, no torch. Run:  python expB_progsearch.py
"""
from __future__ import annotations
import random
from typing import List, Tuple

import core_data as cd

OPS = ("add", "sub", "mul", "div", "mod", "ge", "min", "max")
CAP = 10_000   # clamp register values to keep the VM finite & safe
SIGNED = False # if True, sub may go negative and mod/div are Python-signed
               # (lets the search express (a-b-borrow)%base directly)


def apply_op(op, x, y, base):
    if op == "add": r = x + y
    elif op == "sub": r = (x - y) if SIGNED else (x - y if x >= y else 0)
    elif op == "mul": r = x * y
    elif op == "div": r = x // y if y != 0 else 0
    elif op == "mod": r = x % y if y != 0 else x
    elif op == "ge":  r = 1 if x >= y else 0
    elif op == "min": r = min(x, y)
    elif op == "max": r = max(x, y)
    else: raise ValueError(op)
    if r > CAP: r = CAP
    elif r < -CAP: r = -CAP
    return r


# ----------------------------------------------------------------------------
# Genome: program over indexed values.
#   indices 0..2 = inputs A, B, C ; 3..5 = consts 0,1,base
#   each step i computes val[6+i] = op(val[p], val[q]), p,q < 6+i
#   out_idx -> which value is the OUTPUT digit ; c_idx -> next value of C
# ----------------------------------------------------------------------------
N_INPUT = 6  # A,B,C,const0,const1,constBASE

class Program:
    def __init__(self, steps, out_idx, c_idx):
        self.steps = steps        # list of (op, p, q)
        self.out_idx = out_idx
        self.c_idx = c_idx

    def n_values(self):
        return N_INPUT + len(self.steps)

    def run_position(self, A, B, C, base):
        vals = [A, B, C, 0, 1, base]
        for (op, p, q) in self.steps:
            vals.append(apply_op(op, vals[p], vals[q], base))
        out = vals[self.out_idx]
        c_next = vals[self.c_idx]
        return out, c_next

    def predict(self, a, b, base):
        width = max(_ndigits(a, base), _ndigits(b, base))
        L = width + 1
        ad = cd.to_digits(a, L, base); bd = cd.to_digits(b, L, base)
        C = 0
        digits = []
        for t in range(L):
            out, C = self.run_position(ad[t], bd[t], C, base)
            if not (0 <= out < base):
                return None            # invalid digit -> wrong
            if C > CAP:
                C = CAP
            digits.append(out)
        return cd.from_digits(digits, base)

    def __str__(self):
        names = ["A", "B", "C", "0", "1", "BASE"]
        lines = []
        for i, (op, p, q) in enumerate(self.steps):
            lines.append(f"  v{N_INPUT+i} = {op}({names[p] if p<N_INPUT else 'v'+str(p)}, "
                         f"{names[q] if q<N_INPUT else 'v'+str(q)})")
        out_name = names[self.out_idx] if self.out_idx < N_INPUT else f"v{self.out_idx}"
        c_name = names[self.c_idx] if self.c_idx < N_INPUT else f"v{self.c_idx}"
        lines.append(f"  OUT = {out_name} ; C' = {c_name}")
        return "\n".join(lines)


def _ndigits(n, base):
    if n == 0: return 1
    k = 0
    while n > 0: n //= base; k += 1
    return k


# ----------------------------------------------------------------------------
# Random programs & mutation
# ----------------------------------------------------------------------------
def random_program(rng, max_steps=5):
    n_steps = rng.randint(1, max_steps)
    steps = []
    for i in range(n_steps):
        hi = N_INPUT + i
        op = rng.choice(OPS)
        p = rng.randrange(hi); q = rng.randrange(hi)
        steps.append((op, p, q))
    n_vals = N_INPUT + n_steps
    out_idx = rng.randrange(n_vals)
    c_idx = rng.randrange(n_vals)
    return Program(steps, out_idx, c_idx)


def mutate(prog, rng, max_steps=6):
    steps = [list(s) for s in prog.steps]
    out_idx, c_idx = prog.out_idx, prog.c_idx
    choice = rng.random()
    if choice < 0.35 and steps:                       # tweak a step
        i = rng.randrange(len(steps))
        hi = N_INPUT + i
        what = rng.randrange(3)
        if what == 0: steps[i][0] = rng.choice(OPS)
        elif what == 1: steps[i][1] = rng.randrange(hi)
        else: steps[i][2] = rng.randrange(hi)
    elif choice < 0.55 and len(steps) < max_steps:     # add step
        i = len(steps); hi = N_INPUT + i
        steps.append([rng.choice(OPS), rng.randrange(hi), rng.randrange(hi)])
    elif choice < 0.70 and len(steps) > 1:             # remove last step
        steps.pop()
        nv = N_INPUT + len(steps)
        out_idx = min(out_idx, nv - 1); c_idx = min(c_idx, nv - 1)
    elif choice < 0.85:                                # repoint OUT
        out_idx = rng.randrange(N_INPUT + len(steps))
    else:                                              # repoint C'
        c_idx = rng.randrange(N_INPUT + len(steps))
    steps = [tuple(s) for s in steps]
    return Program(steps, out_idx, c_idx)


# ----------------------------------------------------------------------------
# Fitness: dense per-position output accuracy on a fixed eval set
# ----------------------------------------------------------------------------
def make_eval_set(n, width, base, seed, op="add"):
    rng = random.Random(seed)
    pairs = []
    while len(pairs) < n:
        a = rng.randint(0, base**width - 1); b = rng.randint(0, base**width - 1)
        if op == "sub" and a < b:
            a, b = b, a            # keep results non-negative
        pairs.append((a, b))
    return pairs


def fitness(prog, pairs, base, op="add"):
    """Per-position output accuracy (threading the program's own C). Fully correct
    program -> 1.0. Cheaper & denser than whole-number exact match for search."""
    total = 0; correct = 0
    for a, b in pairs:
        width = max(_ndigits(a, base), _ndigits(b, base)); L = width + 1
        ad = cd.to_digits(a, L, base); bd = cd.to_digits(b, L, base)
        true_out = cd.to_digits(cd.exact_result(a, b, op), L, base)
        C = 0
        for t in range(L):
            out, C = prog.run_position(ad[t], bd[t], C, base)
            if C > CAP: C = CAP
            total += 1
            if out == true_out[t]:
                correct += 1
    return correct / total


# ----------------------------------------------------------------------------
# GP search
# ----------------------------------------------------------------------------
def search(base=10, train_width=3, op="add", pop=400, gens=400, n_eval=400,
           tourn=6, seed=0, restart_every=120, verbose=True):
    rng = random.Random(seed)
    pairs = make_eval_set(n_eval, train_width, base, seed=seed + 1, op=op)
    population = [random_program(rng) for _ in range(pop)]
    scored = [(fitness(p, pairs, base, op), p) for p in population]
    best = max(scored, key=lambda t: t[0])
    stale = 0
    for g in range(gens):
        scored.sort(key=lambda t: t[0], reverse=True)
        if scored[0][0] > best[0]:
            best = scored[0]; stale = 0
        else:
            stale += 1
        if verbose and (g % 25 == 0 or scored[0][0] == 1.0):
            print(f"  gen {g:4d}  best_fit {scored[0][0]:.4f}  steps {len(scored[0][1].steps)}")
        if scored[0][0] == 1.0:
            best = scored[0]; break
        # next generation: elitism + tournament-selected mutants
        elite = [p for _, p in scored[:max(2, pop // 20)]]
        newpop = list(elite)
        # periodic injection of fresh random programs to escape stagnation
        if stale and stale % restart_every == 0:
            newpop += [random_program(rng) for _ in range(pop // 4)]
        while len(newpop) < pop:
            contenders = [scored[rng.randrange(len(scored))] for _ in range(tourn)]
            parent = max(contenders, key=lambda t: t[0])[1]
            newpop.append(mutate(parent, rng))
        population = newpop
        scored = [(fitness(p, pairs, base, op), p) for p in population]
    return best[1], best[0]   # (program, fitness)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", type=str, default="add", choices=["add", "sub"])
    ap.add_argument("--base", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--gens", type=int, default=400)
    ap.add_argument("--signed", action="store_true",
                    help="allow signed sub + Python-signed mod/div (test borrow hypothesis)")
    args = ap.parse_args()
    SIGNED = args.signed
    base, op = args.base, args.op
    print(f"(SIGNED={SIGNED})")
    widths = (1, 2, 3, 4, 6, 8, 12, 16, 20, 30)
    print(f"=== Exp B: program synthesis for {op} (base={base}) ===")
    # a few independent restarts with different seeds
    best_prog, best_fit = None, -1.0
    for seed in range(args.seeds):
        prog, fit = search(base=base, op=op, gens=args.gens, seed=seed, verbose=(seed == 0))
        print(f"seed {seed}: best fitness {fit:.4f}")
        if fit > best_fit:
            best_fit, best_prog = fit, prog
        if best_fit == 1.0:
            break
    print(f"\nBest program (fitness {best_fit:.4f}):")
    print(best_prog)
    def _pred(a, b):
        r = best_prog.predict(a, b, base)
        return r if r is not None else -1   # None = invalid digit; -1 never matches
    rep = cd.length_gen_report(_pred, op, base=base, widths=widths, n_per_width=1000)
    print("\nEXACT length-generalization (whole-number match):")
    print("  " + "  ".join(f"w{w}:{rep[w]:.3f}" for w in widths))
    if min(rep.values()) > 0.999:
        print("\n>>> Discovered a program that length-generalizes PERFECTLY (exact).")
