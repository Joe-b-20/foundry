"""
expF_unified.py — ONE op-conditioned Mealy machine for ALL FOUR ops (+, -, x, /).
Compares CO-TRAINING (all ops from the start) vs SEQUENTIAL (one op at a time), then evaluates
on mixed problems + edge cases.

Honest scoping (forced by proven walls in TRACKER):
- Full nxn mult is NOT finite-state -> x is single-digit-MULTIPLIER (the learnable slice).
- Division by a base-coprime divisor is NOT learnable end-to-end -> / is single-digit-DIVISOR,
  and it will length-generalize ONLY for divisors that DIVIDE the base (2,5 in base 10). The eval
  reports per-divisor-class so the wall is visible, not hidden.
- +,-,x are LSB-first; / is MSB-first. The model is fed each op in its natural digit order and
  told which op via a one-hot op-code. This is the honest "one model, all four".

Run: python expF_unified.py --steps_cotrain 24000 --steps_phase 6000 --dim 8 --hidden 96
"""
from __future__ import annotations
import argparse, random
import torch, torch.nn as nn

import core_data as cd
import expA_mealy as E

DEVICE = E.DEVICE
OPS = ["add", "sub", "mul", "div"]
NOP = len(OPS)
DIV_POOL = list(range(1, 10))   # divisors sampled in training; can be restricted (e.g. [1,2,5])


class UnifiedMealy(nn.Module):
    def __init__(self, base=10, state_dim=8, hidden=96):
        super().__init__()
        self.base = base; self.state_dim = state_dim
        in_dim = state_dim + 2 * base + NOP        # [state ; onehot(a) ; onehot(b) ; onehot(op)]
        self.f = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh(),
                               nn.Linear(hidden, state_dim), nn.Tanh())
        self.g = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh(),
                               nn.Linear(hidden, base))
        self.s0 = nn.Parameter(torch.zeros(state_dim))

    def step(self, s, x):
        z = torch.cat([s, x], dim=-1)
        return self.g(z), self.f(z)

    def forward(self, a_oh, b_oh, op_oh):
        N, L, _ = a_oh.shape
        s = self.s0.unsqueeze(0).expand(N, -1)
        outs = []
        for t in range(L):
            x = torch.cat([a_oh[:, t], b_oh[:, t], op_oh[:, t]], dim=-1)
            logits, s = self.step(s, x)
            outs.append(logits)
        return torch.stack(outs, dim=1)


def digits_msb(n, width, base):
    return list(reversed(cd.to_digits(n, width, base)))


def make_batch(op, n, width, base, seed):
    """Per-op batch in the op's NATURAL digit order. add/sub/mul LSB-first (L=width+1);
    div MSB-first (L=width). b is broadcast (single digit) for mul/div."""
    rng = random.Random(seed); op_idx = OPS.index(op)
    if op in ("add", "sub"):
        L = width + 1
        A = [rng.randint(0, base ** width - 1) for _ in range(n)]
        B = [rng.randint(0, base ** width - 1) for _ in range(n)]
        if op == "sub":
            for i in range(n):
                if A[i] < B[i]: A[i], B[i] = B[i], A[i]
        a_oh = E.onehot_seq(A, L, base); b_oh = E.onehot_seq(B, L, base)
        tgt = torch.zeros(n, L, dtype=torch.long)
        for i in range(n):
            tgt[i] = torch.tensor(cd.to_digits(cd.exact_result(A[i], B[i], op), L, base))
    elif op == "mul":
        L = width + 1
        A = [rng.randint(0, base ** width - 1) for _ in range(n)]
        Bd = [rng.randint(0, base - 1) for _ in range(n)]
        a_oh = E.onehot_seq(A, L, base); b_oh = torch.zeros(n, L, base)
        for i, b in enumerate(Bd): b_oh[i, :, b] = 1.0
        tgt = torch.zeros(n, L, dtype=torch.long)
        for i in range(n):
            tgt[i] = torch.tensor(cd.to_digits(A[i] * Bd[i], L, base))
    else:  # div, MSB-first
        L = width
        A = [rng.randint(0, base ** width - 1) for _ in range(n)]
        D = [rng.choice(DIV_POOL) for _ in range(n)]
        a_oh = torch.zeros(n, L, base); b_oh = torch.zeros(n, L, base)
        tgt = torch.zeros(n, L, dtype=torch.long)
        for i in range(n):
            am = digits_msb(A[i], L, base); qm = digits_msb(A[i] // D[i], L, base)
            for t in range(L):
                a_oh[i, t, am[t]] = 1.0; b_oh[i, t, D[i]] = 1.0; tgt[i, t] = qm[t]
    op_oh = torch.zeros(n, L, NOP); op_oh[:, :, op_idx] = 1.0
    return a_oh, b_oh, op_oh, tgt


@torch.no_grad()
def predict_fn(model, op, base=10):
    model.eval(); op_idx = OPS.index(op)
    def predict(a, b):
        if op in ("add", "sub"):
            width = max(E._ndigits(a, base), E._ndigits(b, base)); L = width + 1
            a_oh = E.onehot_seq([a], L, base).to(DEVICE); b_oh = E.onehot_seq([b], L, base).to(DEVICE)
        elif op == "mul":
            width = E._ndigits(a, base); L = width + 1
            a_oh = E.onehot_seq([a], L, base).to(DEVICE)
            b_oh = torch.zeros(1, L, base, device=DEVICE); b_oh[0, :, b % base] = 1.0
        else:  # div MSB-first
            width = E._ndigits(a, base); L = width
            a_oh = torch.zeros(1, L, base, device=DEVICE); b_oh = torch.zeros(1, L, base, device=DEVICE)
            am = digits_msb(a, L, base)
            for t in range(L):
                a_oh[0, t, am[t]] = 1.0; b_oh[0, t, b % base] = 1.0
        op_oh = torch.zeros(1, L if op != "div" else width, NOP, device=DEVICE); op_oh[:, :, op_idx] = 1.0
        out = model(a_oh, b_oh, op_oh)[0].argmax(-1).tolist()
        if op == "div":
            q = 0
            for dig in out: q = q * base + dig
            return q
        return cd.from_digits(out, base)
    return predict


# ----------------------------------------------------------------------------
# Eval helpers
# ----------------------------------------------------------------------------
def _sample_operands(op, width, base, rng):
    a = rng.randint(0, base ** width - 1)
    if op in ("add", "sub"):
        b = rng.randint(0, base ** width - 1)
        if op == "sub" and a < b: a, b = b, a
    elif op == "mul":
        b = rng.randint(0, base - 1)
    else:
        b = rng.randint(1, base - 1)
    return a, b


def acc_at_width(model, op, base, width, n=400, seed=0, div_filter=None):
    """div_filter (a set of divisors) is sampled DIRECTLY, decoupled from the training pool —
    so we can probe learnable {1,2,5} vs walled {3,4,6,..} regardless of what div was trained on."""
    rng = random.Random(seed + width * 7); pred = predict_fn(model, op, base); ok = 0
    for _ in range(n):
        if op == "div":
            pool = sorted(div_filter) if div_filter else DIV_POOL
            a = rng.randint(0, base ** width - 1); b = rng.choice(pool)
        else:
            a, b = _sample_operands(op, width, base, rng)
        if pred(a, b) == cd.exact_result(a, b, op): ok += 1
    return ok / n


def lengen_table(model, base, widths=(1, 2, 3, 4, 6, 8, 12, 16, 20)):
    print("  per-op length generalization (exact-acc):")
    for op in OPS:
        if op == "div":
            good = {1, 2, 5}; bad = {3, 4, 6, 7, 8, 9}
            rg = {w: acc_at_width(model, op, base, w, n=600, div_filter=good) for w in widths}
            rb = {w: acc_at_width(model, op, base, w, n=600, div_filter=bad) for w in widths}
            print("    div /[1,2,5]  : " + "  ".join(f"w{w}:{rg[w]:.3f}" for w in widths))
            print("    div /[3,4,6..]: " + "  ".join(f"w{w}:{rb[w]:.3f}" for w in widths))
        else:
            r = {w: acc_at_width(model, op, base, w, n=600) for w in widths}
            print(f"    {op:3s}          : " + "  ".join(f"w{w}:{r[w]:.3f}" for w in widths))


def mixed_eval(model, base, width=4, n=4000, seed=999):
    """A test bag mixing all four ops uniformly at a fixed width."""
    rng = random.Random(seed); per = {op: [0, 0] for op in OPS}; ok = 0
    preds = {op: predict_fn(model, op, base) for op in OPS}
    for _ in range(n):
        op = rng.choice(OPS); a, b = _sample_operands(op, width, base, rng)
        per[op][1] += 1
        if preds[op](a, b) == cd.exact_result(a, b, op):
            per[op][0] += 1; ok += 1
    print(f"  MIXED test (n={n}, width={width}): overall exact-acc {ok/n:.3f}")
    for op in OPS:
        c, t = per[op]; print(f"    {op:3s}: {c}/{t} = {c/max(t,1):.3f}")
    return ok / n


EDGE = [
    ("add", 999, 1), ("add", 9999, 1), ("add", 0, 0), ("add", 1234567, 0),
    ("sub", 1000, 1), ("sub", 100000, 1), ("sub", 555, 555), ("sub", 7, 0),
    ("mul", 999, 9), ("mul", 1234567, 0), ("mul", 1234567, 1), ("mul", 99999, 9),
    ("div", 100, 2), ("div", 1000000, 2), ("div", 12345, 5),      # divisors that divide base -> should pass
    ("div", 12345, 3), ("div", 12345, 7), ("div", 98765, 9),      # base-coprime -> expected WALL
    ("div", 5, 9), ("div", 8, 8), ("div", 0, 7),                  # a<d ; a==d ; 0/d
]


def edge_eval(model, base):
    print("  EDGE CASES (op a b -> expected | got | PASS/FAIL):")
    npass = 0
    for op, a, b in EDGE:
        exp = cd.exact_result(a, b, op); got = predict_fn(model, op, base)(a, b)
        ok = (got == exp); npass += ok
        note = ""
        if op == "div" and b not in (1, 2, 5):
            note = "  (base-coprime divisor: WALL expected)"
        print(f"    {op} {a} {b:>2} -> {exp:>8} | {got:>8} | {'PASS' if ok else 'FAIL'}{note}")
    print(f"  edge pass: {npass}/{len(EDGE)}")
    return npass


# ----------------------------------------------------------------------------
# Training regimes
# ----------------------------------------------------------------------------
def train_cotrain(model, steps, base=10, bs=256, lr=1e-2, train_widths=(1, 2, 3, 4, 5), log=4000):
    model.to(DEVICE); opt = torch.optim.Adam(model.parameters(), lr=lr); lossf = nn.CrossEntropyLoss()
    wr = random.Random(1); orr = random.Random(2)
    for step in range(1, steps + 1):
        op = orr.choice(OPS); w = wr.choice(train_widths)
        a, b, o, t = make_batch(op, bs, w, base, seed=step)
        a, b, o, t = a.to(DEVICE), b.to(DEVICE), o.to(DEVICE), t.to(DEVICE)
        loss = lossf(model(a, b, o).reshape(-1, base), t.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % log == 0 or step == 1:
            print(f"    step {step:6d} op={op:3s} loss {loss.item():.4f}")
    return model


def train_sequential(model, steps_per_phase, base=10, bs=256, lr=1e-2,
                     train_widths=(1, 2, 3, 4, 5), order=OPS):
    """One op at a time (curriculum order). Single optimizer across phases. After each phase,
    eval ALL ops at width 4 -> forgetting matrix."""
    model.to(DEVICE); opt = torch.optim.Adam(model.parameters(), lr=lr); lossf = nn.CrossEntropyLoss()
    wr = random.Random(1); fmatrix = []
    for phase, op in enumerate(order):
        for step in range(1, steps_per_phase + 1):
            w = wr.choice(train_widths)
            a, b, o, t = make_batch(op, bs, w, base, seed=phase * 1000000 + step)
            a, b, o, t = a.to(DEVICE), b.to(DEVICE), o.to(DEVICE), t.to(DEVICE)
            loss = lossf(model(a, b, o).reshape(-1, base), t.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
        row = {o2: acc_at_width(model, o2, base, 4, n=400,
                                div_filter={1, 2, 5} if o2 == "div" else None) for o2 in order}
        fmatrix.append((op, row))
        print(f"    after phase {phase+1} (trained {op:3s}): " +
              "  ".join(f"{o2}={row[o2]:.2f}" for o2 in order))
    print("  FORGETTING MATRIX (rows=after training this op; cols=acc on each op @w4; div=/[1,2,5]):")
    print("            " + "  ".join(f"{o:>5s}" for o in order))
    for op, row in fmatrix:
        print(f"    {op:8s}  " + "  ".join(f"{row[o]:.2f} " for o in order))
    return fmatrix


def blended_schedule(focus=0.60, order=OPS):
    """Phase k introduces op_k with weight `focus`; the remaining (1-focus) is split UNIFORMLY
    over the earlier ops (rehearsal). phase1 = 100% add; phase2 = 60% sub / 40% add; etc."""
    sched = []
    for k, op in enumerate(order):
        earlier = order[:k]
        if not earlier:
            w = {op: 1.0}
        else:
            w = {op: focus}
            share = (1.0 - focus) / len(earlier)
            for e in earlier:
                w[e] = share
        sched.append((op, w))
    return sched


def sample_op(weights, rng):
    r = rng.random(); c = 0.0
    items = list(weights.items())
    for o, p in items:
        c += p
        if r <= c:
            return o
    return items[-1][0]


def train_blended(model, steps_per_phase, base=10, bs=256, lr=1e-2,
                  train_widths=(1, 2, 3, 4, 5), focus=0.60, order=OPS):
    """Blended curriculum / rehearsal: like sequential (curriculum order) but each phase keeps
    practicing earlier ops per blended_schedule(). Forgetting matrix after each phase."""
    sched = blended_schedule(focus, order)
    model.to(DEVICE); opt = torch.optim.Adam(model.parameters(), lr=lr); lossf = nn.CrossEntropyLoss()
    wr = random.Random(1); orng = random.Random(3); fmatrix = []
    for phase, (op, weights) in enumerate(sched):
        wstr = " ".join(f"{k}:{v:.2f}" for k, v in weights.items())
        print(f"  phase {phase+1} introduce {op:3s}  weights[{wstr}]")
        for step in range(1, steps_per_phase + 1):
            cur = sample_op(weights, orng); w = wr.choice(train_widths)
            a, b, o, t = make_batch(cur, bs, w, base, seed=phase * 1000000 + step)
            a, b, o, t = a.to(DEVICE), b.to(DEVICE), o.to(DEVICE), t.to(DEVICE)
            loss = lossf(model(a, b, o).reshape(-1, base), t.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
        row = {o2: acc_at_width(model, o2, base, 4, n=400,
                                div_filter={1, 2, 5} if o2 == "div" else None) for o2 in order}
        fmatrix.append((op, row))
        print(f"    after phase {phase+1}: " + "  ".join(f"{o2}={row[o2]:.2f}" for o2 in order))
    print("  FORGETTING MATRIX (rows=after each phase; cols=acc@w4; div=/[1,2,5]):")
    print("            " + "  ".join(f"{o:>5s}" for o in order))
    for op, row in fmatrix:
        print(f"    {op:8s}  " + "  ".join(f"{row[o]:.2f} " for o in order))
    return fmatrix


def dynamic_weights(model, base, order=OPS, floor=0.05):
    """Adaptive curriculum weights: measure each op's acc@w4, weight sampling by its DEFICIT
    (1-acc) with a retention floor. Worse ops -> more batches. (div measured on {1,2,5}.)"""
    accs = {op: acc_at_width(model, op, base, 4, n=200,
                             div_filter={1, 2, 5} if op == "div" else None) for op in order}
    deficits = {op: max(0.0, 1.0 - accs[op]) for op in order}
    s = sum(deficits.values())
    if s < 1e-6:
        w = {op: 1.0 / len(order) for op in order}
    else:
        w = {op: floor + (1.0 - floor * len(order)) * (deficits[op] / s) for op in order}
    return w, accs


def train_dynamic_blended(model, steps_phase, steps_balance, base=10, bs=256, lr=1e-2,
                          train_widths=(1, 2, 3, 4, 5), focus=0.60, reweight_every=2000, order=OPS):
    """Phases 1-4: introduce ops with focus+rehearsal (prevents forgetting). Phase 5: DYNAMIC
    balancing — re-measure per-op acc every `reweight_every` steps and weight sampling by deficit
    (auto-allocates to under-trained ops, e.g. div/mul). Reports the weights it picks over time."""
    model.to(DEVICE); opt = torch.optim.Adam(model.parameters(), lr=lr); lossf = nn.CrossEntropyLoss()
    wr = random.Random(1); orng = random.Random(3)
    for phase, (op, weights) in enumerate(blended_schedule(focus, order)):
        wstr = " ".join(f"{k}:{v:.2f}" for k, v in weights.items())
        print(f"  phase {phase+1} introduce {op:3s}  weights[{wstr}]")
        for step in range(1, steps_phase + 1):
            cur = sample_op(weights, orng); w = wr.choice(train_widths)
            a, b, o, t = make_batch(cur, bs, w, base, seed=phase * 1000000 + step)
            a, b, o, t = a.to(DEVICE), b.to(DEVICE), o.to(DEVICE), t.to(DEVICE)
            loss = lossf(model(a, b, o).reshape(-1, base), t.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
        row = {o2: acc_at_width(model, o2, base, 4, n=300,
                                div_filter={1, 2, 5} if o2 == "div" else None) for o2 in order}
        print(f"    after phase {phase+1}: " + "  ".join(f"{o2}={row[o2]:.2f}" for o2 in order))

    print(f"  ===== PHASE 5: DYNAMIC BALANCING ({steps_balance} steps, reweight every {reweight_every}) =====")
    weights = None
    for step in range(1, steps_balance + 1):
        if (step - 1) % reweight_every == 0:
            weights, accs = dynamic_weights(model, base, order)
            astr = " ".join(f"{k}={accs[k]:.2f}" for k in order)
            wstr = " ".join(f"{k}:{weights[k]:.2f}" for k in order)
            print(f"    step {step:5d}  acc[{astr}]  ->  weights[{wstr}]")
        cur = sample_op(weights, orng); w = wr.choice(train_widths)
        a, b, o, t = make_batch(cur, bs, w, base, seed=9 * 1000000 + step)
        a, b, o, t = a.to(DEVICE), b.to(DEVICE), o.to(DEVICE), t.to(DEVICE)
        loss = lossf(model(a, b, o).reshape(-1, base), t.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def make_hybrid_qdiv(model, base):
    """Return qdiv(a,d) = a//d for ANY single digit d, computed by long division whose inner
    repeated subtraction uses the MODEL'S OWN learned subtraction. Exact + length-generalizing
    even for base-coprime divisors the neural head can't do in one pass."""
    sub = predict_fn(model, "sub", base)
    def qdiv(a, d):
        W = E._ndigits(a, base); am = digits_msb(a, W, base); rem = 0; qs = []
        for at in am:
            val = rem * base + at; q = 0; guard = 0
            while val >= d:
                val = sub(val, d); q += 1; guard += 1
                if guard > base + 1: break          # safety (val<base*d => q<base)
            qs.append(q); rem = val
        Q = 0
        for dd in qs: Q = Q * base + dd
        return Q
    return qdiv


def hybrid_division_demo(model, base, divisors=(2, 3, 5, 7, 9)):
    qdiv = make_hybrid_qdiv(model, base)
    print("  HYBRID division via the model's OWN subtraction (repeated subtraction), ALL divisors:")
    for d in divisors:
        rep = {}
        for w in (3, 6, 12, 20):
            rng = random.Random(d * 100 + w); ok = 0
            for _ in range(120):
                a = rng.randint(0, base ** w - 1)
                if qdiv(a, d) == a // d: ok += 1
            rep[w] = ok / 120
        tag = "(neural wall)" if base % d else "(neural OK)"
        print(f"    /{d} {tag:13s}: " + "  ".join(f"w{w}:{rep[w]:.3f}" for w in rep))


def system_mixed_eval(model, base, width=4, n=4000, seed=777):
    """COMPLETE SYSTEM: a fully-mixed bag over all four ops with ALL division divisors 1..9,
    where +,-,x use the neural heads and / is routed through the hybrid (composed repeated
    subtraction). Shows the system handles all four ops for ALL inputs, not just learnable divisors."""
    rng = random.Random(seed); qdiv = make_hybrid_qdiv(model, base)
    preds = {op: predict_fn(model, op, base) for op in ("add", "sub", "mul")}
    per = {op: [0, 0] for op in OPS}; ok = 0
    for _ in range(n):
        op = rng.choice(OPS); a, b = _sample_operands(op, width, base, rng)
        got = qdiv(a, b) if op == "div" else preds[op](a, b)
        per[op][1] += 1
        if got == cd.exact_result(a, b, op):
            per[op][0] += 1; ok += 1
    print(f"  COMPLETE-SYSTEM mixed (n={n}, w={width}, /-> hybrid, ALL divisors): overall {ok/n:.3f}")
    for op in OPS:
        c, t = per[op]; print(f"    {op:3s}: {c}/{t} = {c/max(t,1):.3f}")
    return ok / n


def make_solver(model, base):
    """Apply any op to a running value: neural heads for +,-,x; hybrid (own-subtraction) for /."""
    qdiv = make_hybrid_qdiv(model, base)
    p = {op: predict_fn(model, op, base) for op in ("add", "sub", "mul")}
    def apply(v, op, operand):
        return qdiv(v, operand) if op == "div" else p[op](v, operand)
    return apply


def system_edge_eval(model, base):
    """The 21 edge cases, scored through the COMPLETE system (division via hybrid)."""
    apply = make_solver(model, base)
    print("  COMPLETE-SYSTEM edge cases (division -> hybrid):")
    npass = 0
    for op, a, b in EDGE:
        exp = cd.exact_result(a, b, op); got = apply(a, op, b); ok = (got == exp); npass += ok
        print(f"    {op} {a} {b:>2} -> {exp:>8} | {got:>8} | {'PASS' if ok else 'FAIL'}")
    print(f"  system edge pass: {npass}/{len(EDGE)}")
    return npass


def chained_eval(model, base, n=1000, width=3, seed=4242, examples=6):
    """Problems that use ALL FOUR ops in ONE expression: a random permutation of {+,-,x,/} applied
    left-to-right, feeding the model's OWN output into the next op. Exact-match vs ground truth.
    (x operand = single digit; / operand = 1..9; - operand <= running value to stay non-negative.)"""
    import random
    apply = make_solver(model, base); rng = random.Random(seed)
    ok = 0; shown = []
    for i in range(n):
        order = ["add", "sub", "mul", "div"]; rng.shuffle(order)
        start = rng.randint(1, base ** width - 1)
        tv = start; steps = []
        for op in order:
            if op == "add":   operand = rng.randint(0, base ** width - 1)
            elif op == "sub": operand = rng.randint(0, tv)            # keep >= 0
            elif op == "mul": operand = rng.randint(0, base - 1)      # single digit
            else:             operand = rng.randint(1, base - 1)      # nonzero single digit
            tv = cd.exact_result(tv, operand, op); steps.append((op, operand))
        mv = start
        for op, operand in steps:
            mv = apply(mv, op, operand)
        ok += (mv == tv)
        if i < examples:
            sym = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
            expr = f"{start}" + "".join(f" {sym[o]}{v}" for o, v in steps)
            shown.append(f"    {expr} = {tv} | got {mv} | {'OK' if mv == tv else 'X'}")
    return ok, n, shown


def full_eval(model, base, tag):
    print(f"\n----- EVAL: {tag} -----")
    lengen_table(model, base)
    mixed_eval(model, base)
    edge_eval(model, base)


if __name__ == "__main__":
    import os
    os.makedirs("runs", exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps_cotrain", type=int, default=24000)
    ap.add_argument("--steps_phase", type=int, default=6000)
    ap.add_argument("--dim", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--div_learnable", action="store_true",
                    help="restrict division training divisors to {1,2,5} (disentangle interference)")
    ap.add_argument("--cotrain_only", action="store_true")
    ap.add_argument("--blended_only", action="store_true",
                    help="run ONLY the blended curriculum (rehearsal) regime")
    ap.add_argument("--focus", type=float, default=0.60)
    ap.add_argument("--dynamic", action="store_true",
                    help="blended curriculum + DYNAMIC balancing phase (auto-weight by deficit)")
    ap.add_argument("--steps_balance", type=int, default=10000)
    ap.add_argument("--system_eval", type=str, default=None,
                    help="load this checkpoint and run the complete-system eval (/-> hybrid)")
    args = ap.parse_args()
    base = 10
    if args.smoke:
        args.steps_cotrain = 3000; args.steps_phase = 1500
    if args.div_learnable:
        DIV_POOL = [1, 2, 5]
        print(f"  [div training restricted to divisors {DIV_POOL}]")
    print(f"device={DEVICE}  UnifiedMealy d={args.dim} hidden={args.hidden}")

    if args.system_eval:
        m = UnifiedMealy(base=base, state_dim=args.dim, hidden=args.hidden)
        m.load_state_dict(torch.load(args.system_eval, map_location=DEVICE)); m.to(DEVICE)
        print(f"  loaded {args.system_eval}")
        system_mixed_eval(m, base)
        print("\n  ALL-FOUR-OPS-IN-ONE-PROBLEM (chained expressions, each uses +,-,x,/):")
        for w in (3, 6, 10):
            ok, ntot, shown = chained_eval(m, base, n=1000, width=w)
            print(f"  width {w:2d}: {ok}/{ntot} = {ok/ntot:.3f} exact")
            for line in shown[:4]:
                print(line)
        print()
        system_edge_eval(m, base)
        hybrid_division_demo(m, base, divisors=(2, 3, 4, 5, 6, 7, 8, 9))
        raise SystemExit(0)

    if args.dynamic:
        DIV_POOL = [1, 2, 5]   # divisor resolution: neural div trained ONLY on learnable divisors
        print(f"\n############ REGIME 4: DYNAMIC BLENDED (rehearsal + adaptive balancing) ############")
        print(f"  [neural division restricted to learnable divisors {DIV_POOL}; "
              f"coprime divisors handled by composition — see hybrid demo]")
        torch.manual_seed(0)
        m_dy = UnifiedMealy(base=base, state_dim=args.dim, hidden=args.hidden)
        print(f"  params: {sum(p.numel() for p in m_dy.parameters())}")
        train_dynamic_blended(m_dy, args.steps_phase, args.steps_balance, base=base, focus=args.focus)
        full_eval(m_dy, base, "DYNAMIC BLENDED")
        hybrid_division_demo(m_dy, base)
        torch.save(m_dy.state_dict(), "runs/expF_unified_dynamic.pt")
        print("\nsaved runs/expF_unified_dynamic.pt")
        raise SystemExit(0)

    if args.blended_only:
        print(f"\n############ REGIME 3: BLENDED CURRICULUM (rehearsal, focus={args.focus}) ############")
        torch.manual_seed(0)
        m_bl = UnifiedMealy(base=base, state_dim=args.dim, hidden=args.hidden)
        print(f"  params: {sum(p.numel() for p in m_bl.parameters())}")
        train_blended(m_bl, args.steps_phase, base=base, focus=args.focus)
        full_eval(m_bl, base, f"BLENDED (focus={args.focus})")
        torch.save(m_bl.state_dict(), "runs/expF_unified_blended.pt")
        print("\nsaved runs/expF_unified_blended.pt")
        raise SystemExit(0)

    print("\n############ REGIME 1: CO-TRAINING (all ops from the start) ############")
    torch.manual_seed(0)
    m_co = UnifiedMealy(base=base, state_dim=args.dim, hidden=args.hidden)
    print(f"  params: {sum(p.numel() for p in m_co.parameters())}")
    train_cotrain(m_co, args.steps_cotrain, base=base)
    full_eval(m_co, base, "CO-TRAINED")

    if args.cotrain_only:
        raise SystemExit(0)
    print("\n############ REGIME 2: SEQUENTIAL (one op at a time, add->sub->mul->div) ############")
    torch.manual_seed(0)
    m_sq = UnifiedMealy(base=base, state_dim=args.dim, hidden=args.hidden)
    train_sequential(m_sq, args.steps_phase, base=base)
    full_eval(m_sq, base, "SEQUENTIAL (final)")

    if not args.smoke:
        torch.save(m_co.state_dict(), "runs/expF_unified_cotrain.pt")
        print("\nsaved runs/expF_unified_cotrain.pt")
