"""
interp_unified2.py — sharpened decode of the unified 4-op model.
Fixes from v1: (1) EXCLUDE the shared start state s0 from decode pools (it's one
identical artificial point), (2) use a proper LINEAR PROBE + best-threshold single
dim (not nearest-centroid), (3) decode mult-carry along its 1-D manifold, (4) op-code
selector tested only on the divisors the model was trained on.
"""
import random
import numpy as np
import torch
import core_data as cd
import expA_mealy as E
import expF_unified as F

base = 10; DEV = F.DEVICE; OPS = F.OPS
torch.manual_seed(0)
m = F.UnifiedMealy(base=base, state_dim=8, hidden=96)
m.load_state_dict(torch.load("runs/expF_unified_dynamic.pt", map_location=DEV)); m.to(DEV); m.eval()


def op_input_seq(op, a, b):
    if op in ("add", "sub"):
        w = max(E._ndigits(a, base), E._ndigits(b, base)); L = w + 1
        ad = cd.to_digits(a, L, base); bd = cd.to_digits(b, L, base)
        return [(ad[t], bd[t]) for t in range(L)]
    elif op == "mul":
        w = E._ndigits(a, base); L = w + 1; ad = cd.to_digits(a, L, base)
        return [(ad[t], b % base) for t in range(L)]
    else:
        L = E._ndigits(a, base); am = F.digits_msb(a, L, base)
        return [(am[t], b % base) for t in range(L)]


@torch.no_grad()
def trace(op, a, b):
    seq = op_input_seq(op, a, b); op_idx = OPS.index(op); s = m.s0.unsqueeze(0).to(DEV)
    states = [s.squeeze(0).cpu().numpy().copy()]
    for (da, db) in seq:
        x = torch.zeros(1, 2 * base + F.NOP, device=DEV)
        x[0, da] = 1.0; x[0, base + db] = 1.0; x[0, 2 * base + op_idx] = 1.0
        _, s = m.step(s, x); states.append(s.squeeze(0).cpu().numpy().copy())
    return seq, states


def true_latent(op, seq):
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
    else:
        r = 0
        for (da, db) in seq: lat.append(r); r = (r * base + da) % db
    return lat


def collect(op, n=600, widths=(2, 3, 4, 6), seed=0, div_d=None, skip_start=True):
    """skip_start: drop t=0 (the shared s0). Align states[t] with incoming latent[t]."""
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
        seq, states = trace(op, a, b); lat = true_latent(op, seq)
        t0 = 1 if skip_start else 0
        for t in range(t0, len(seq)):
            S.append(states[t]); Lat.append(lat[t])
    return np.array(S), np.array(Lat)


def best_single_dim(S, y):
    """Best binary-accuracy single dim with OPTIMAL threshold (both polarities)."""
    best = (-1, -1.0, None, None)
    for d in range(S.shape[1]):
        x = S[:, d]; cands = np.unique(np.quantile(x, np.linspace(0, 1, 41)))
        for thr in cands:
            for sgn in (+1, -1):
                pred = (x > thr).astype(int) if sgn > 0 else (x < thr).astype(int)
                acc = (pred == y).mean()
                if acc > best[1]: best = (d, acc, sgn, float(thr))
    return best


def linear_probe(S, y, steps=500, lr=0.05):
    Xt = torch.tensor(S, dtype=torch.float32); yt = torch.tensor(y, dtype=torch.long)
    K = int(y.max()) + 1
    W = torch.zeros(S.shape[1], K, requires_grad=True); b = torch.zeros(K, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr); lossf = torch.nn.CrossEntropyLoss()
    for _ in range(steps):
        loss = lossf(Xt @ W + b, yt); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        acc = (Xt @ W + b).argmax(1).eq(yt).float().mean().item()
        imp = W.detach().abs().sum(1).numpy()  # per-dim importance
    return acc, imp


print("=" * 78)
print("B1. WHERE each op's latent lives (s0 excluded; best single dim + linear probe)")
print("=" * 78)
for op, name in [("add", "CARRY"), ("sub", "BORROW")]:
    S, Lat = collect(op, n=800)
    d, acc, sgn, thr = best_single_dim(S, Lat)
    pacc, imp = linear_probe(S, Lat)
    top = np.argsort(-imp)[:3]
    print(f"  {op} ({name}): best single dim d{d} (sgn{sgn:+d} thr{thr:+.2f}) acc {acc:.4f}  |  "
          f"linear-probe acc {pacc:.4f}")
    print(f"       per-dim probe importance: " + " ".join(f"d{i}:{imp[i]:.1f}" for i in range(8)) +
          f"   -> top dims {list(top)}")

# mult-carry along its manifold
S, Lat = collect("mul", n=1200)
pacc, imp = linear_probe(S, Lat)
Sc = S - S.mean(0); U, sv, Vt = np.linalg.svd(Sc, full_matrices=False)
PC = Sc @ Vt[:3].T
# 1-D nearest-centroid on PC1, and on PC1..3
def nc_decode(P, y):
    classes = sorted(set(y.tolist())); C = np.stack([P[y == c].mean(0) for c in classes]); ck = np.array(classes)
    d2 = ((P[:, None, :] - C[None]) ** 2).sum(-1); return (ck[d2.argmin(1)] == y).mean()
sp = abs(np.corrcoef(np.argsort(np.argsort(PC[:, 0])), Lat)[0, 1])  # Spearman-ish
print(f"  mul (MULT-CARRY 0..8): linear-probe acc {pacc:.4f}  |  "
      f"NC on PC1 {nc_decode(PC[:, :1], Lat):.4f}  NC on PC1-3 {nc_decode(PC[:, :3], Lat):.4f}")
print(f"       monotonicity |corr(rank(PC1), carry)| = {sp:.3f}  (->1 = clean ordered 1-D manifold)")
print(f"       top-dim probe importance: " + " ".join(f"d{i}:{imp[i]:.1f}" for i in range(8)))

for dd in (2, 5):
    S, Lat = collect("div", n=800, div_d=dd)
    pacc, imp = linear_probe(S, Lat); top = np.argsort(-imp)[:3]
    print(f"  div /{dd} (REMAINDER 0..{dd-1}): linear-probe acc {pacc:.4f}   top dims {list(top)}  "
          + " ".join(f"d{i}:{imp[i]:.1f}" for i in range(8)))

print("\n" + "=" * 78)
print("B2. CROSS-OP dim usage — is the carry dim (d1) reused, or are ops on separate dims?")
print("=" * 78)
# decode add-carry from ONLY d1; decode sub-borrow from ONLY d1; etc.
Sa, La = collect("add", n=800); Ss, Ls = collect("sub", n=800)
for label, S_, y_ in [("add-carry", Sa, La), ("sub-borrow", Ss, Ls)]:
    accs = []
    for d in range(8):
        dd_, acc, _, _ = best_single_dim(S_[:, [d]], y_); accs.append(acc)
    order = np.argsort(-np.array(accs))
    print(f"  {label}: single-dim acc per dim = " + " ".join(f"d{i}:{accs[i]:.2f}" for i in range(8)))
    print(f"       best dims: {[f'd{i}({accs[i]:.2f})' for i in order[:3]]}")

print("\n" + "=" * 78)
print("B3. OP-CODE SELECTOR (fixed: div tested only on trained divisors {2,5})")
print("=" * 78)


@torch.no_grad()
def step0_out(op, da, db):
    op_idx = OPS.index(op); x = torch.zeros(1, 2 * base + F.NOP, device=DEV)
    x[0, da] = 1.0; x[0, base + db] = 1.0; x[0, 2 * base + op_idx] = 1.0
    logits, _ = m.step(m.s0.unsqueeze(0).to(DEV), x); return int(logits.argmax(-1))


exp_fns = {"add": lambda a, b: (a + b) % base, "sub": lambda a, b: (a - b) % base,
           "mul": lambda a, b: (a * b) % base, "div": lambda a, b: a // b}
agree = {op: [0, 0] for op in OPS}
for (a, b) in [(7, 3), (4, 5), (9, 2), (6, 2), (8, 5), (3, 5), (1, 2), (9, 5), (5, 2), (7, 5)]:
    for op in OPS:
        bb = b if op != "div" else (2 if b % 2 == 0 else 5)  # ensure a trained divisor
        if op in ("add", "sub", "mul"): bb = b
        got = step0_out(op, a, bb); exp = exp_fns[op](a, bb)
        agree[op][0] += (got == exp); agree[op][1] += 1
print("  step-0 output equals the op-selected function on (a,b) pairs:")
for op in OPS:
    print(f"    {op}: {agree[op][0]}/{agree[op][1]} = {agree[op][0]/agree[op][1]:.3f}")
print("\nINTERP_UNIFIED2 DONE")
