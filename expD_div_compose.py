"""
expD_div_compose.py — division CAPSTONE: discover single-digit-divisor long division as a
COMPOSITION grounded on the DISCOVERED subtraction (borrow) FSM.

Parallel to expC_compose (multiplication = repeated ADDITION): here division = repeated
SUBTRACTION. The neural Mealy machine could not learn the per-digit division map end-to-end
(see expA_div1 + expD_discrete: best ~0.69 / ~0.27 at w20). But long division decomposes:
   rem = 0
   for a_t in digits(A) MSB-first:
       val   = rem*base + a_t
       q_t   = how many times d fits in val      # repeated subtraction of d
       rem   = val - q_t*d
   quotient = the q_t sequence (MSB-first)
The inner "how many times d fits in val" is REPEATED SUBTRACTION, grounded on the extracted
borrow FSM (the arithmetic is discovered; loop control / compare is exact, exactly as expC
used an exact `for _ in range(d)` with FSM-grounded addition).

GP discovers the OUTER loop body over high-level ops {SHL, ADD, QDIG, RDIG}, where QDIG/RDIG
are the repeated-subtraction quotient/remainder primitives. Because the body is position-
independent and the primitives are exact at any length, the composed divider length-
generalizes EXACTLY. 'exact' mode validates the discovery; 'fsm' mode grounds subtraction on
the actually-extracted borrow FSM.

Run: python expD_div_compose.py [exact|fsm]
"""
from __future__ import annotations
import random
import core_data as cd


# ----------------------------------------------------------------------------
# Primitives (pluggable). sub is either python (exact) or the extracted borrow FSM.
# QDIG/RDIG = quotient/remainder by REPEATED SUBTRACTION (grounded on sub).
# ----------------------------------------------------------------------------
class DivPrimitives:
    def __init__(self, base=10, sub_fn=None):
        self.base = base
        self.sub = sub_fn or (lambda x, y: x - y)   # grounded on borrow FSM in 'fsm' mode

    def shl(self, x, k):
        if k > 64:
            raise OverflowError
        return x * (self.base ** k)

    def add(self, x, y):
        return x + y

    def combine(self, rem, a_t):
        """Shift a new digit into the running value: rem*base + a_t. The fundamental MSB-first
        digit-accumulation op (as primitive as SHL). Session-1 lesson: primitive vocabulary
        gates what GP can discover; with SHL+ADD the val-formation is 2 ops deep and GP stalls."""
        return rem * self.base + a_t

    def _qr(self, val, d):
        """quotient and remainder of val//d by repeated subtraction of d (grounded sub).
        Loop control (the >= compare) is exact; the SUBTRACTION is the discovered op."""
        if d <= 0:
            raise ZeroDivisionError
        q = 0
        guard = 0
        while val >= d:
            val = self.sub(val, d)
            q += 1
            guard += 1
            if guard > 10 * self.base:                # safety: val<base*d so q<base; never hit if correct
                raise OverflowError
        return q, val

    def qdig(self, val, d):
        return self._qr(val, d)[0]

    def rdig(self, val, d):
        return self._qr(val, d)[1]


# ----------------------------------------------------------------------------
# Long-division loop machine.
#   state carried across digits: rem (starts 0)
#   per digit a_t (MSB-first): values = [rem, a_t, d, base, 0, 1]; run body; then
#     emit digit = values[emit_idx]; rem = values[rem_idx]
# ----------------------------------------------------------------------------
N_IN = 6   # rem, a_t, d, base, 0, 1
OPS = ("SHL", "ADD", "COMBINE", "QDIG", "RDIG")


class DivLoopProg:
    def __init__(self, steps, emit_idx, rem_idx):
        self.steps = steps          # list of (op, p, q)
        self.emit_idx = emit_idx
        self.rem_idx = rem_idx

    def run_qdigits(self, A, d, prim, width=None):
        """Return (emitted quotient-digit list, carried-rem-after-each-step list, W), MSB-first.
        The rem trace lets fitness score the FULL long-division step contract
        (rem,a_t,d)->(q,rem'), giving GP direct signal for the remainder update (the carried
        state is otherwise unsupervised, which creates a strong emit-only local optimum)."""
        base = prim.base
        W = width or max(_nd(A, base), 1)
        am = list(reversed(cd.to_digits(A, W, base)))   # MSB-first
        rem = 0
        qdigits, rems = [], []
        for a_t in am:
            vals = [rem, a_t, d, base, 0, 1]
            for (op, p, q) in self.steps:
                x, y = vals[p], vals[q]
                if op == "SHL": r = prim.shl(x, y)
                elif op == "ADD": r = prim.add(x, y)
                elif op == "COMBINE": r = prim.combine(x, y)
                elif op == "QDIG": r = prim.qdig(x, y)
                elif op == "RDIG": r = prim.rdig(x, y)
                else: raise ValueError(op)
                if isinstance(r, int) and r.bit_length() > 2048:
                    raise OverflowError
                vals.append(r)
            qdigits.append(vals[self.emit_idx])
            rem = vals[self.rem_idx]
            rems.append(rem)
        return qdigits, rems, W

    def run_quotient(self, A, d, prim, width=None):
        qdigits, _, _ = self.run_qdigits(A, d, prim, width)
        q = 0
        for dig in qdigits:
            q = q * prim.base + dig
        return q

    def __str__(self):
        names = ["rem", "a_t", "d", "base", "0", "1"]
        out = []
        for i, (op, p, q) in enumerate(self.steps):
            pn = names[p] if p < N_IN else f"v{p}"
            qn = names[q] if q < N_IN else f"v{q}"
            out.append(f"  v{N_IN+i} = {op}({pn}, {qn})")
        en = names[self.emit_idx] if self.emit_idx < N_IN else f"v{self.emit_idx}"
        rn = names[self.rem_idx] if self.rem_idx < N_IN else f"v{self.rem_idx}"
        out.append(f"  emit {en} ; rem' = {rn}")
        return "\n".join(out)


def _nd(n, base):
    if n == 0: return 1
    k = 0
    while n > 0: n //= base; k += 1
    return k


# ----------------------------------------------------------------------------
# GP over loop bodies
# ----------------------------------------------------------------------------
def random_prog(rng, max_steps=4):
    ns = rng.randint(2, max_steps); steps = []
    for i in range(ns):
        hi = N_IN + i
        steps.append((rng.choice(OPS), rng.randrange(hi), rng.randrange(hi)))
    tot = N_IN + ns
    return DivLoopProg(steps, rng.randrange(tot), rng.randrange(tot))


def mutate(prog, rng, max_steps=5):
    steps = [list(s) for s in prog.steps]
    emit_idx, rem_idx = prog.emit_idx, prog.rem_idx
    c = rng.random()
    if c < 0.4 and steps:
        i = rng.randrange(len(steps)); hi = N_IN + i; w = rng.randrange(3)
        if w == 0: steps[i][0] = rng.choice(OPS)
        elif w == 1: steps[i][1] = rng.randrange(hi)
        else: steps[i][2] = rng.randrange(hi)
    elif c < 0.55 and len(steps) < max_steps:
        i = len(steps); hi = N_IN + i
        steps.append([rng.choice(OPS), rng.randrange(hi), rng.randrange(hi)])
        # KEY: adding a step is useless unless an output points at it. The long-division
        # rem-update needs a new RDIG step AND rem_idx->it simultaneously; bridge that here.
        new_idx = N_IN + i
        r = rng.random()
        if r < 0.45: rem_idx = new_idx
        elif r < 0.6: emit_idx = new_idx
    elif c < 0.65 and len(steps) > 2:
        steps.pop()
    elif c < 0.82:
        emit_idx = rng.randrange(N_IN + len(steps))
    else:
        rem_idx = rng.randrange(N_IN + len(steps))
    tot = N_IN + len(steps)
    return DivLoopProg([tuple(s) for s in steps], min(emit_idx, tot - 1), min(rem_idx, tot - 1))


def fitness(prog, pairs, prim):
    """Score the FULL long-division step contract: at each MSB-first position the program must
    emit the correct quotient digit AND carry the correct remainder. Scoring the (q, rem) PAIR
    (not just q) gives GP signal for the otherwise-unsupervised remainder update. Fitness
    1.0 <=> exact. (Per-digit-only fitness stalls at ~0.57: emit-right-but-rem-stuck optimum.)"""
    base = prim.base
    tot = 0; ok = 0
    for a, d in pairs:
        W = max(_nd(a, base), 1)
        # true MSB-first per-step quotient digit and remainder
        am = list(reversed(cd.to_digits(a, W, base)))
        tq, tr, r = [], [], 0
        for a_t in am:
            v = r * base + a_t; tq.append(v // d); tr.append(v % d); r = v % d
        try:
            pq, pr, _ = prog.run_qdigits(a, d, prim, width=W)
        except Exception:
            pq = pr = [None] * W
        for i in range(W):
            tot += 2
            if i < len(pq) and pq[i] == tq[i]: ok += 1
            if i < len(pr) and pr[i] == tr[i]: ok += 1
    return ok / max(tot, 1)


def make_pairs(rng, n, width, base):
    return [(rng.randint(0, base ** width - 1), rng.randint(1, base - 1)) for _ in range(n)]


def search(prim, train_width=3, pop=500, gens=600, n_eval=300, seed=0, verbose=True):
    rng = random.Random(seed)
    pairs = make_pairs(rng, n_eval, train_width, prim.base)
    population = [random_prog(rng) for _ in range(pop)]
    scored = [(fitness(p, pairs, prim), p) for p in population]
    best = max(scored, key=lambda t: t[0])
    for g in range(gens):
        scored.sort(key=lambda t: t[0], reverse=True)
        if scored[0][0] > best[0]: best = scored[0]
        if verbose and (g % 25 == 0 or scored[0][0] == 1.0):
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


def lengen_report(prog, prim, widths=(1, 2, 3, 4, 6, 8, 10, 12, 16, 20), n=400, seed=7):
    rep = {}
    for w in widths:
        rng = random.Random(seed + w); ok = 0
        for _ in range(n):
            a = rng.randint(0, prim.base ** w - 1); d = rng.randint(1, prim.base - 1)
            try:
                if prog.run_quotient(a, d, prim) == a // d: ok += 1
            except Exception:
                pass
        rep[w] = ok / n
    return rep


def canonical_body():
    """The schoolbook long-division step, as a DivLoopProg: val=rem*base+a_t; q=val//d;
    rem'=val%d (both via repeated subtraction). This is the reduction we VERIFY; GP attempts
    to rediscover it separately."""
    return DivLoopProg([("COMBINE", 0, 1), ("QDIG", 6, 2), ("RDIG", 6, 2)],
                       emit_idx=7, rem_idx=8)


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "exact"
    base = 10
    if mode == "exact":
        prim = DivPrimitives(base=base)
        print("=== Division capstone: discover long division as repeated subtraction (EXACT sub) ===")
        best, fit = None, -1
        for seed in range(6):
            p, f = search(prim, seed=seed, verbose=(seed == 0))
            print(f"seed {seed}: fitness {f:.4f}")
            if f > fit: fit, best = f, p
            if fit == 1.0: break
        print(f"\nBest GP body (fitness {fit:.4f}):")
        print(best)
        if fit >= 0.999:
            rep = lengen_report(best, prim)
            print("EXACT length-gen of the GP-DISCOVERED divider:")
            print("  " + "  ".join(f"w{w}:{rep[w]:.3f}" for w in rep))
            if min(rep.values()) > 0.999:
                print("\n>>> GP discovered long division (repeated subtraction); length-generalizes EXACTLY.")
        else:
            print(f"\n[GP stalled at {fit:.3f} — the two-output long-division step is a hard GP "
                  f"landscape. Verifying the REDUCTION directly instead:]")
            cb = canonical_body()
            print(cb)
            rep = lengen_report(cb, prim)
            print("EXACT length-gen of the canonical long-division reduction (exact sub):")
            print("  " + "  ".join(f"w{w}:{rep[w]:.3f}" for w in rep))
            if min(rep.values()) > 0.999:
                print("\n>>> Single-digit-divisor long division = repeated subtraction; "
                      "length-generalizes EXACTLY (verified reduction).")
    else:
        # Ground subtraction on the EXTRACTED borrow FSM and verify the canonical reduction.
        cb = canonical_body()
        print("Long-division reduction (loop body):"); print(cb)
        import torch, expA_mealy as E
        model = E.NeuralMealy(base=base, state_dim=1)
        model.load_state_dict(torch.load("runs/expA_mealy_sub_d1.pt", map_location=E.DEVICE))
        model.to(E.DEVICE).eval()
        _, sub_fsm, info = E.extract_fsm(model, base=base, probe_width=3)
        # checkpoint trained on subtraction; the extracted transducer's predict fn IS borrow.
        rng = random.Random(0)
        for _ in range(3000):
            x = rng.randint(0, 10**6); y = rng.randint(0, x)
            assert sub_fsm(x, y) == x - y, (x, y, sub_fsm(x, y))
        print(f"borrow FSM ({info['n_states']} states) subtracts EXACTLY on 3000 checks.")
        prim = DivPrimitives(base=base, sub_fn=sub_fsm)
        rep = lengen_report(cb, prim)
        print("EXACT length-gen of the divider with subtraction grounded on the borrow FSM:")
        print("  " + "  ".join(f"w{w}:{rep[w]:.3f}" for w in rep))
        if min(rep.values()) > 0.999:
            print("\n>>> Exact arbitrary-length single-digit-divisor division, built on the "
                  "EXTRACTED borrow FSM. Division IS repeated subtraction. Curriculum +,-,x,/ complete.")
