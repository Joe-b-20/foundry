"""
expC_compose.py — Exp C: discover FULL multi-digit x multi-digit multiplication as a
HIERARCHICAL COMPOSITION of simpler discovered operations (the project's capstone).

Full n x n multiplication is NOT a finite-state transduction (see expMul_full.py), so
no fixed-state machine length-generalizes on it. But it DECOMPOSES:
   A * B = sum_j  SHL(  MULDIGIT(A, B_j),  j )   over digits B_j of B.
i.e. a loop over B's digits, each step: multiply A by a single digit (an FSM we
discovered in expA_mul1), shift left by the position j, and add to an accumulator
(the carry FSM we discovered in expA_mealy).

Here a "loop machine" iterates j over B's digits; a short straight-line program (the
loop BODY) is searched by GP to update the accumulator. The search rediscovers long
multiplication. Because the body is position-independent and the primitives length-
generalize, the composed multiplier length-generalizes EXACTLY (verified to 12x12+).

Primitives are pluggable: 'exact' (python) to validate the discovery, or 'fsm' to use
the actually-EXTRACTED add/mul-digit FSMs so the whole thing stands on discovered
pieces. Run: python expC_compose.py [exact|fsm]
"""
from __future__ import annotations
import random
import core_data as cd

# ----------------------------------------------------------------------------
# Whole-number primitives (pluggable). Default = exact; swap to extracted FSMs.
# ----------------------------------------------------------------------------
class Primitives:
    def __init__(self, base=10, add_fn=None, muldigit_fn=None):
        self.base = base
        self.add = add_fn or (lambda x, y: x + y)
        self.muldigit = muldigit_fn or (lambda x, d: x * d)   # x times single digit d
    def shl(self, x, k):                                       # shift left k digit positions
        if k > 64:                                             # guard: huge shift => blowup
            raise OverflowError
        return x * (self.base ** k)

BITCAP = 2048   # ~616 digits; any intermediate bigger than this => treat program as invalid


# ----------------------------------------------------------------------------
# Loop machine: acc=0; for j over digits(B) LSB-first: acc = body(acc,A,Bj,j); return acc
#   value indices: 0=acc 1=A 2=Bj 3=j 4=const0 5=const1
#   ops: ADD(p,q)=add(vp,vq)   MULDIGIT(p,q)=muldigit(vp,vq)   SHL(p,q)=shl(vp,vq)
# ----------------------------------------------------------------------------
N_IN = 6
HLOPS = ("ADD", "MULDIGIT", "SHL")

class LoopProg:
    def __init__(self, steps, acc_idx):
        self.steps = steps          # list of (op, p, q)
        self.acc_idx = acc_idx      # which value becomes the new acc

    def n_values(self):
        return N_IN + len(self.steps)

    def run(self, A, B, prim):
        base = prim.base
        Bdigits = cd.to_digits(B, max(cd_ndigits(B, base), 1), base)
        acc = 0
        for j, Bj in enumerate(Bdigits):
            vals = [acc, A, Bj, j, 0, 1]
            for (op, p, q) in self.steps:
                x, y = vals[p], vals[q]
                if op == "ADD": r = prim.add(x, y)
                elif op == "MULDIGIT": r = prim.muldigit(x, y)
                elif op == "SHL": r = prim.shl(x, y)
                else: raise ValueError(op)
                if r.bit_length() > BITCAP:        # guard against big-int blowup
                    raise OverflowError
                vals.append(r)
            acc = vals[self.acc_idx]
        return acc

    def __str__(self):
        names = ["acc", "A", "Bj", "j", "0", "1"]
        out = []
        for i, (op, p, q) in enumerate(self.steps):
            pn = names[p] if p < N_IN else f"v{p}"
            qn = names[q] if q < N_IN else f"v{q}"
            out.append(f"  v{N_IN+i} = {op}({pn}, {qn})")
        an = names[self.acc_idx] if self.acc_idx < N_IN else f"v{self.acc_idx}"
        out.append(f"  acc' = {an}")
        return "\n".join(out)


def cd_ndigits(n, base):
    if n == 0: return 1
    k = 0
    while n > 0: n //= base; k += 1
    return k


# ----------------------------------------------------------------------------
# GP over loop bodies
# ----------------------------------------------------------------------------
def random_prog(rng, max_steps=4):
    ns = rng.randint(1, max_steps); steps = []
    for i in range(ns):
        hi = N_IN + i
        steps.append((rng.choice(HLOPS), rng.randrange(hi), rng.randrange(hi)))
    return LoopProg(steps, rng.randrange(N_IN + ns))


def mutate(prog, rng, max_steps=5):
    steps = [list(s) for s in prog.steps]; acc_idx = prog.acc_idx
    c = rng.random()
    if c < 0.4 and steps:
        i = rng.randrange(len(steps)); hi = N_IN + i; w = rng.randrange(3)
        if w == 0: steps[i][0] = rng.choice(HLOPS)
        elif w == 1: steps[i][1] = rng.randrange(hi)
        else: steps[i][2] = rng.randrange(hi)
    elif c < 0.6 and len(steps) < max_steps:
        i = len(steps); hi = N_IN + i
        steps.append([rng.choice(HLOPS), rng.randrange(hi), rng.randrange(hi)])
    elif c < 0.75 and len(steps) > 1:
        steps.pop(); acc_idx = min(acc_idx, N_IN + len(steps) - 1)
    else:
        acc_idx = rng.randrange(N_IN + len(steps))
    return LoopProg([tuple(s) for s in steps], acc_idx)


def fitness(prog, pairs, prim):
    ok = 0
    for a, b in pairs:
        try:
            if prog.run(a, b, prim) == a * b:
                ok += 1
        except Exception:
            pass
    return ok / len(pairs)


def search(prim, train_width=3, pop=300, gens=300, n_eval=300, seed=0, verbose=True):
    rng = random.Random(seed)
    pairs = [(rng.randint(0, prim.base**train_width - 1),
              rng.randint(0, prim.base**train_width - 1)) for _ in range(n_eval)]
    population = [random_prog(rng) for _ in range(pop)]
    scored = [(fitness(p, pairs, prim), p) for p in population]
    best = max(scored, key=lambda t: t[0])
    for g in range(gens):
        scored.sort(key=lambda t: t[0], reverse=True)
        if scored[0][0] > best[0]: best = scored[0]
        if verbose and (g % 20 == 0 or scored[0][0] == 1.0):
            print(f"  gen {g:4d} best_fit {scored[0][0]:.4f} steps {len(scored[0][1].steps)}")
        if scored[0][0] == 1.0: best = scored[0]; break
        elite = [p for _, p in scored[:max(2, pop // 20)]]
        newpop = list(elite)
        while len(newpop) < pop:
            cont = [scored[rng.randrange(len(scored))] for _ in range(6)]
            newpop.append(mutate(max(cont, key=lambda t: t[0])[1], rng))
        population = newpop
        scored = [(fitness(p, pairs, prim), p) for p in population]
    return best[1], best[0]


def lengen_report(prog, prim, widths=(1, 2, 3, 4, 6, 8, 10, 12), n=400, seed=7):
    rep = {}
    for w in widths:
        rng = random.Random(seed + w); ok = 0
        for _ in range(n):
            a = rng.randint(0, prim.base**w - 1); b = rng.randint(0, prim.base**w - 1)
            try:
                if prog.run(a, b, prim) == a * b: ok += 1
            except Exception:
                pass
        rep[w] = ok / n
    return rep


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "exact"
    base = 10
    if mode == "exact":
        prim = Primitives(base=base)
        print("=== Exp C: discover full multiplication as a loop composition (EXACT primitives) ===")
        best, fit = None, -1
        for seed in range(3):
            p, f = search(prim, seed=seed, verbose=(seed == 0))
            print(f"seed {seed}: fitness {f:.4f}")
            if f > fit: fit, best = f, p
            if fit == 1.0: break
        print(f"\nDiscovered loop body (fitness {fit:.4f}):")
        print(best)
        rep = lengen_report(best, prim)
        print("\nEXACT length-generalization of the composed multiplier:")
        print("  " + "  ".join(f"w{w}:{rep[w]:.3f}" for w in rep))
        if min(rep.values()) > 0.999:
            print("\n>>> Full multiplication length-generalizes EXACTLY as a discovered composition.")
    else:
        # Ground the discovered composition on the EXTRACTED carry FSM (everything
        # reduces to the one discovered algorithm). No re-search; discovery already
        # shown in 'exact' mode. Here we verify the composition stays exact.
        from expC_fsm_primitives import load_fsm_primitives
        add_fn, muldigit_fn, info = load_fsm_primitives(base=base)
        prim = Primitives(base=base, add_fn=add_fn, muldigit_fn=muldigit_fn)
        print(f"=== Exp C: full multiplication grounded on EXTRACTED FSM primitives ===")
        print(f"    {info}")
        # the composition discovered in exact mode:
        best = LoopProg([("MULDIGIT", 1, 2), ("SHL", 6, 3), ("ADD", 0, 7)], 8)
        print("Discovered composition (loop body):"); print(best)
        rep = lengen_report(best, prim, widths=(1, 2, 3, 4, 6, 8, 10, 12))
        print("EXACT length-gen of the FSM-grounded multiplier:")
        print("  " + "  ".join(f"w{w}:{rep[w]:.3f}" for w in rep))
        if min(rep.values()) > 0.999:
            print("\n>>> Exact arbitrary-length multiplication, built ENTIRELY on the "
                  "extracted carry FSM. The discovery hierarchy add->muldigit->mult is verified.")
