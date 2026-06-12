"""
interp_unified4.py — finish the unified-model account.
  D1  CAUSAL mult-carry injection (the nonlinear case): inject incoming carry=k into the
      scratchpad and verify the output shifts to (a_t*b + k) % 10.
  D2  co-train: are carry and borrow on the SAME axis or DIFFERENT axes?
"""
import random
import numpy as np
import torch
import core_data as cd
import expA_mealy as E
import expF_unified as F

base = 10; DEV = F.DEVICE; OPS = F.OPS
torch.manual_seed(0)
SCRATCH = [1, 5, 6]


def load(ckpt):
    m = F.UnifiedMealy(base=base, state_dim=8, hidden=96)
    m.load_state_dict(torch.load(ckpt, map_location=DEV)); m.to(DEV); m.eval(); return m


def op_input_seq(op, a, b):
    if op in ("add", "sub"):
        w = max(E._ndigits(a, base), E._ndigits(b, base)); L = w + 1
        return [(cd.to_digits(a, L, base)[t], cd.to_digits(b, L, base)[t]) for t in range(L)]
    elif op == "mul":
        w = E._ndigits(a, base); L = w + 1
        return [(cd.to_digits(a, L, base)[t], b % base) for t in range(L)]
    else:
        L = E._ndigits(a, base); am = F.digits_msb(a, L, base)
        return [(am[t], b % base) for t in range(L)]


@torch.no_grad()
def trace(m, op, a, b):
    seq = op_input_seq(op, a, b); op_idx = OPS.index(op); s = m.s0.unsqueeze(0).to(DEV)
    states = [s.squeeze(0).cpu().numpy().copy()]
    for (da, db) in seq:
        x = torch.zeros(1, 2 * base + F.NOP, device=DEV)
        x[0, da] = 1.0; x[0, base + db] = 1.0; x[0, 2 * base + op_idx] = 1.0
        _, s = m.step(s, x); states.append(s.squeeze(0).cpu().numpy().copy())
    return seq, states


def true_latent(op, seq):
    lat = []
    if op == "mul":
        c = 0
        for (da, db) in seq: lat.append(c); c = (da * db + c) // base
    elif op == "add":
        c = 0
        for (da, db) in seq: lat.append(c); c = (da + db + c) // base
    elif op == "sub":
        bw = 0
        for (da, db) in seq: lat.append(bw); bw = 1 if (da - db - bw) < 0 else 0
    return lat


def collect(m, op, n=2000, widths=(2, 3, 4, 6), seed=0):
    rng = random.Random(seed); S = []; Lat = []
    for _ in range(n):
        w = rng.choice(widths); a = rng.randint(0, base ** w - 1)
        if op == "mul":
            b = rng.randint(0, base - 1)
        else:
            b = rng.randint(0, base ** w - 1)
            if op == "sub" and a < b: a, b = b, a
        seq, states = trace(m, op, a, b); lat = true_latent(op, seq)
        for t in range(1, len(seq)):
            S.append(states[t]); Lat.append(lat[t])
    return np.array(S), np.array(Lat)


m = load("runs/expF_unified_dynamic.pt")
print("=" * 78)
print("D1. CAUSAL mult-carry injection (dynamic model). Inject incoming carry=k into the")
print(f"    scratchpad {SCRATCH}; output should become (a_t*b + k) % 10 for every digit/multiplier.")
print("=" * 78)
S, Lat = collect(m, "mul", n=3000)
cents_full = {k: S[Lat == k].mean(0) for k in range(9) if (Lat == k).sum() > 5}
base0 = cents_full[0].copy()


@torch.no_grad()
def mul_out_from_state(state_vec, da, db):
    x = torch.zeros(1, 2 * base + F.NOP, device=DEV)
    x[0, da] = 1.0; x[0, base + db] = 1.0; x[0, 2 * base + 2] = 1.0  # op=mul
    s = torch.tensor(state_vec, dtype=torch.float32, device=DEV).unsqueeze(0)
    logits, _ = m.step(s, x); return int(logits.argmax(-1))


print("    k  | inject-FULL centroid_k acc | inject-SCRATCH-only acc  (target out=(a_t*b+k)%10)")
for k in sorted(cents_full):
    full_state = cents_full[k]
    scr_state = base0.copy(); scr_state[SCRATCH] = cents_full[k][SCRATCH]
    okf = oks = tot = 0
    for da in range(base):
        for db in range(base):
            tot += 1; tgt = (da * db + k) % base
            okf += (mul_out_from_state(full_state, da, db) == tgt)
            oks += (mul_out_from_state(scr_state, da, db) == tgt)
    print(f"    {k}  |        {okf/tot:.3f}            |        {oks/tot:.3f}")

print("\n" + "=" * 78)
print("D2. CO-TRAIN: which dim hosts carry vs borrow (same axis = partition, or shared?)")
print("=" * 78)


def best_single_dim(S, y):
    best = (-1, -1.0, None)
    for d in range(S.shape[1]):
        x = S[:, d]
        for thr in np.unique(np.quantile(x, np.linspace(0, 1, 41))):
            for sgn in (+1, -1):
                pred = (x > thr).astype(int) if sgn > 0 else (x < thr).astype(int)
                acc = (pred == y).mean()
                if acc > best[1]: best = (d, acc, sgn)
    return best


mc = load("runs/expF_unified_cotrain.pt")
for op, name in [("add", "carry"), ("sub", "borrow")]:
    S, Lat = collect(mc, op, n=1500)
    d, acc, sgn = best_single_dim(S, Lat)
    # per-dim single accuracies
    accs = [best_single_dim(S[:, [i]], Lat)[1] for i in range(8)]
    print(f"  cotrain {op} {name:6s}: best dim d{d} (acc {acc:.3f})   "
          f"per-dim: " + " ".join(f"d{i}:{accs[i]:.2f}" for i in range(8)))
print("\nINTERP_UNIFIED4 DONE")
