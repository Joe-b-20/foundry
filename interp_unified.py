"""
interp_unified.py — DEEP interpretability of the unified 4-op Mealy model
(runs/expF_unified_dynamic.pt, UnifiedMealy d=8 hidden=96).

Question: what is the 8-dim continuous recurrent state actually doing for each op,
and how does the op-code route the computation?

Sections:
  A1  per-op state ACTIVITY (which of the 8 dims are live for which op)
  A2  DECODE the true latent (carry / borrow / mult-carry / remainder) from the state,
      and find WHICH dim hosts it per op  -> is the state partitioned across ops?
  A3  op-code as FUNCTION SELECTOR (same digits, op-code flips the computed function)
  A4  concrete digit-by-digit TRACES (watch the carry/borrow/rem flip)
  A5  routing geometry: how separate are the op directions in the hidden layer
"""
import random
import numpy as np
import torch
import core_data as cd
import expA_mealy as E
import expF_unified as F

base = 10
DEV = F.DEVICE
OPS = F.OPS  # ["add","sub","mul","div"]
torch.manual_seed(0)

m = F.UnifiedMealy(base=base, state_dim=8, hidden=96)
m.load_state_dict(torch.load("runs/expF_unified_dynamic.pt", map_location=DEV)); m.to(DEV); m.eval()


# ---------------------------------------------------------------- tracing infra
def op_input_seq(op, a, b):
    """(a_digit, b_digit) per step in the op's NATURAL digit order; b is constant for mul/div."""
    if op in ("add", "sub"):
        w = max(E._ndigits(a, base), E._ndigits(b, base)); L = w + 1
        ad = cd.to_digits(a, L, base); bd = cd.to_digits(b, L, base)
        return [(ad[t], bd[t]) for t in range(L)]
    elif op == "mul":
        w = E._ndigits(a, base); L = w + 1
        ad = cd.to_digits(a, L, base)
        return [(ad[t], b % base) for t in range(L)]
    else:  # div, MSB-first
        L = E._ndigits(a, base); am = F.digits_msb(a, L, base)
        return [(am[t], b % base) for t in range(L)]


@torch.no_grad()
def trace(op, a, b):
    """Return seq, states[0..L] (s[0]=s0), out_digits[0..L-1]."""
    seq = op_input_seq(op, a, b); op_idx = OPS.index(op)
    s = m.s0.unsqueeze(0).to(DEV)
    states = [s.squeeze(0).cpu().numpy().copy()]; outs = []
    for (da, db) in seq:
        x = torch.zeros(1, 2 * base + F.NOP, device=DEV)
        x[0, da] = 1.0; x[0, base + db] = 1.0; x[0, 2 * base + op_idx] = 1.0
        logits, s = m.step(s, x)
        outs.append(int(logits.argmax(-1))); states.append(s.squeeze(0).cpu().numpy().copy())
    return seq, states, outs


def true_latent(op, seq):
    """Incoming latent at each position t, aligned with states[t]: carry/borrow/mcarry/rem."""
    lat = []
    if op == "add":
        c = 0
        for (da, db) in seq: lat.append(c); c = (da + db + c) // base
    elif op == "sub":
        bw = 0
        for (da, db) in seq: lat.append(bw); bw = 1 if (da - db - bw) < 0 else 0
    elif op == "mul":
        c = 0
        for (da, db) in seq: lat.append(c); c = (da * db + c) // base
    else:  # div, rem update rem' = (rem*base + a_t) % d
        r = 0
        for (da, db) in seq: lat.append(r); r = (r * base + da) % db
    return lat


def collect(op, n=500, widths=(2, 3, 4, 6), seed=0, div_d=None):
    rng = random.Random(seed); S = []; Lat = []
    for _ in range(n):
        w = rng.choice(widths); a = rng.randint(0, base ** w - 1)
        if op in ("add", "sub"):
            b = rng.randint(0, base ** w - 1)
            if op == "sub" and a < b: a, b = b, a
        elif op == "mul":
            b = rng.randint(0, base - 1)
        else:
            b = div_d if div_d else rng.choice([2, 5])
        seq, states, outs = trace(op, a, b); lat = true_latent(op, seq)
        for t in range(len(seq)):
            S.append(states[t]); Lat.append(lat[t])
    return np.array(S), np.array(Lat)


# ---------------------------------------------------------------- A1: activity
print("=" * 78)
print("A1. PER-OP STATE ACTIVITY  (std of each of the 8 state dims, per op)")
print("    a dim with ~0 std is SATURATED/unused for that op; high std = it carries info")
print("=" * 78)
act = {}
for op in OPS:
    S, _ = collect(op, n=400)
    act[op] = S.std(0)
print("    dim:        " + "  ".join(f"d{i}" for i in range(8)))
for op in OPS:
    print(f"    {op:3s} std:   " + "  ".join(f"{v:.2f}" for v in act[op]))
print("    (live dims, std>0.15):")
for op in OPS:
    live = [i for i in range(8) if act[op][i] > 0.15]
    print(f"      {op}: dims {live}")


# ---------------------------------------------------------------- A2: decode
print("\n" + "=" * 78)
print("A2. DECODE the TRUE latent from the state, and find WHICH dim hosts it")
print("=" * 78)


def best_single_dim_binary(S, y):
    """Best accuracy predicting binary y from a sign threshold on a single dim."""
    best = (-1, -1.0, +1)
    for d in range(S.shape[1]):
        x = S[:, d]
        for sgn in (+1, -1):
            pred = (x * sgn > 0).astype(int)
            acc = (pred == y).mean()
            if acc > best[1]: best = (d, acc, sgn)
    return best  # (dim, acc, sign)


def nearest_centroid_decode(S, y):
    """Accuracy of nearest-centroid (in full 8-dim) decoding of categorical y."""
    classes = sorted(set(y.tolist())); cents = {c: S[y == c].mean(0) for c in classes}
    C = np.stack([cents[c] for c in classes]); ck = np.array(classes)
    d2 = ((S[:, None, :] - C[None, :, :]) ** 2).sum(-1)
    pred = ck[d2.argmin(1)]
    return (pred == y).mean(), classes, cents


# add / sub: binary latent
for op in ("add", "sub"):
    S, Lat = collect(op, n=600)
    d, acc, sgn = best_single_dim_binary(S, Lat)
    name = "CARRY" if op == "add" else "BORROW"
    print(f"  {op} ({name}, binary): best single dim = d{d} (sign {sgn:+d}) -> "
          f"threshold acc {acc:.4f}   [frac of latent=1: {Lat.mean():.2f}]")
    # full-state nearest centroid for reference
    nc, _, _ = nearest_centroid_decode(S, Lat)
    print(f"       full-8d nearest-centroid acc {nc:.4f}")

# mul: 9-valued multiplicative carry
S, Lat = collect("mul", n=900)
nc, classes, cents = nearest_centroid_decode(S, Lat)
counts = {c: int((Lat == c).sum()) for c in classes}
print(f"  mul (MULT-CARRY, values {classes}): nearest-centroid decode acc {nc:.4f}")
print(f"       class counts: {counts}")
# how many dims does it span? PCA on mul states
Sc = S - S.mean(0); U, sv, Vt = np.linalg.svd(Sc, full_matrices=False)
evr = (sv ** 2) / (sv ** 2).sum()
print(f"       PCA explained-variance ratio (top 6): {np.round(evr[:6], 3)}")
# centroid of each carry value projected on top-2 PCs (shows the ordered manifold)
P = Sc @ Vt[:2].T
print("       carry-value -> mean(PC1,PC2)  (note monotone ordering = a 1-D manifold):")
for c in classes:
    mp = P[Lat == c].mean(0); print(f"         carry {c}: ({mp[0]:+.2f}, {mp[1]:+.2f})  n={counts[c]}")

# div /2 and /5: remainder
for dd in (2, 5):
    S, Lat = collect("div", n=600, div_d=dd)
    nc, classes, _ = nearest_centroid_decode(S, Lat)
    d, acc, sgn = best_single_dim_binary(S, (Lat > 0).astype(int)) if dd == 2 else (None, None, None)
    extra = f"  | best single-dim(rem>0) d{d} acc {acc:.3f}" if dd == 2 else ""
    print(f"  div /{dd} (REMAINDER, values {classes}): nearest-centroid decode acc {nc:.4f}{extra}")


# ---------------------------------------------------------------- A3: selector
print("\n" + "=" * 78)
print("A3. OP-CODE AS FUNCTION SELECTOR  (state=s0, SAME digits, flip only the op-code)")
print("    out at step0 should be (a+b)%10 / (a-b)%10 / (a*b)%10 / a//b  per op")
print("=" * 78)


@torch.no_grad()
def step0_out(op, da, db):
    op_idx = OPS.index(op)
    x = torch.zeros(1, 2 * base + F.NOP, device=DEV)
    x[0, da] = 1.0; x[0, base + db] = 1.0; x[0, 2 * base + op_idx] = 1.0
    logits, _ = m.step(m.s0.unsqueeze(0).to(DEV), x)
    return int(logits.argmax(-1))


print("    (a,b) | add got/exp | sub got/exp | mul got/exp | div got/exp")
tests = [(7, 3), (4, 5), (9, 9), (6, 2), (8, 5), (3, 7), (0, 4)]
exp_fns = {"add": lambda a, b: (a + b) % base, "sub": lambda a, b: (a - b) % base,
           "mul": lambda a, b: (a * b) % base, "div": lambda a, b: a // b if b else 0}
agree = {op: 0 for op in OPS}; tot = 0
for (a, b) in tests:
    row = f"    ({a},{b}) "
    for op in OPS:
        if op == "div" and b == 0:
            row += f"| {op}:  -  "; continue
        got = step0_out(op, a, b); exp = exp_fns[op](a, b)
        ok = got == exp; agree[op] += ok; row += f"| {op}:{got}/{exp}{'' if ok else '!'} "
    tot += 1
    print(row)
print("    per-op agreement with the selected function: " +
      "  ".join(f"{op}:{agree[op]}/{tot}" for op in OPS))


# ---------------------------------------------------------------- A4: traces
print("\n" + "=" * 78)
print("A4. CONCRETE TRACES (input digits in natural order; watch latent dim flip)")
print("=" * 78)
# pick, per op, the dim that hosts the latent (from A2)
host = {}
for op in ("add", "sub"):
    S, Lat = collect(op, n=600); d, acc, sgn = best_single_dim_binary(S, Lat); host[op] = (d, sgn)
for op, a, b in [("add", 47, 58), ("sub", 803, 67), ("mul", 268, 7), ("div", 1234, 2)]:
    seq, states, outs = trace(op, a, b); lat = true_latent(op, seq)
    label = {"add": f"{a}+{b}", "sub": f"{a}-{b}", "mul": f"{a}*{b}", "div": f"{a}/{b}"}[op]
    expv = cd.exact_result(a, b, op)
    print(f"  {label} = {expv}   (digits shown in {'LSB' if op!='div' else 'MSB'}-first order)")
    hd = host.get(op, (None, None))[0]
    for t, (da, db) in enumerate(seq):
        sv = states[t]
        hoststr = f"  hostdim d{hd}={sv[hd]:+.2f}" if hd is not None else ""
        print(f"    t{t}: a={da} b={db} -> out={outs[t]}   true_latent={lat[t]}{hoststr}"
              f"   state=[{','.join(f'{v:+.1f}' for v in sv)}]")
    # reconstruct the number from outs
    if op == "div":
        q = 0
        for dg in outs: q = q * base + dg
    else:
        q = cd.from_digits(outs, base)
    print(f"    -> decoded answer {q}  {'OK' if q == expv else 'MISMATCH'}")


# ---------------------------------------------------------------- A5: routing
print("\n" + "=" * 78)
print("A5. ROUTING GEOMETRY  (how the op-code steers the hidden layer of f and g)")
print("=" * 78)
# first linear layer of f and g: weight (hidden, 32); op-code occupies input cols 2*base : 2*base+4
for name, net in [("f(next-state)", m.f), ("g(output)", m.g)]:
    W = net[0].weight.detach().cpu().numpy()  # (hidden, 32)
    opcols = W[:, 2 * base:2 * base + F.NOP]  # (hidden, 4) — the op-shift directions
    digcols = W[:, :2 * base]
    # norm of each op's steering vector vs the digit-input weights
    opnorm = np.linalg.norm(opcols, axis=0)  # per-op
    print(f"  {name}: ||op-shift|| per op {np.round(opnorm,2)}   "
          f"||digit-weights||_fro {np.linalg.norm(digcols):.1f}")
    # cosine similarity between the 4 op steering vectors
    cs = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            cs[i, j] = (opcols[:, i] @ opcols[:, j]) / (np.linalg.norm(opcols[:, i]) * np.linalg.norm(opcols[:, j]) + 1e-9)
    print(f"        cosine(op_i, op_j) over hidden units  (rows/cols = {OPS}):")
    for i in range(4):
        print(f"          {OPS[i]:3s} " + "  ".join(f"{cs[i,j]:+.2f}" for j in range(4)))

# causal: ablate the op-code (zero it) -> does add break?
print("  CAUSAL ablation: zero the op-code one-hot during an add, measure accuracy")


@torch.no_grad()
def add_acc_opzeroed(zero_op=False, n=300):
    rng = random.Random(5); ok = 0
    for _ in range(n):
        a = rng.randint(0, 999); b = rng.randint(0, 999)
        seq = op_input_seq("add", a, b); s = m.s0.unsqueeze(0).to(DEV); outs = []
        for (da, db) in seq:
            x = torch.zeros(1, 2 * base + F.NOP, device=DEV)
            x[0, da] = 1.0; x[0, base + db] = 1.0
            if not zero_op: x[0, 2 * base + 0] = 1.0
            logits, s = m.step(s, x); outs.append(int(logits.argmax(-1)))
        ok += cd.from_digits(outs, base) == a + b
    return ok / n


print(f"    add accuracy WITH op-code:    {add_acc_opzeroed(False):.3f}")
print(f"    add accuracy with op-code=0:  {add_acc_opzeroed(True):.3f}  (collapse => op-code is load-bearing)")
print("\nINTERP_UNIFIED DONE")
