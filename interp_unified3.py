"""
interp_unified3.py — confirm the "shared scratchpad" mechanism of the unified model.
  C1  decode each op's latent from JUST {d1,d5,d6} vs the complement (shared subspace?)
  C2  curriculum effect: compare DYNAMIC vs CO-TRAIN for the carry-clean/borrow-distributed
      asymmetry (does introducing add first give it the axis-aligned code?)
  C3  CAUSAL carry injection: overwrite only {d1,d5,d6} with the carry=1 centroid and show
      the output digit flips by +1 (proves that subspace IS the carry, causally)
"""
import random
import numpy as np
import torch
import core_data as cd
import expA_mealy as E
import expF_unified as F

base = 10; DEV = F.DEVICE; OPS = F.OPS
torch.manual_seed(0)


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


def collect(m, op, n=800, widths=(2, 3, 4, 6), seed=0, div_d=None):
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
        seq, states = trace(m, op, a, b); lat = true_latent(op, seq)
        for t in range(1, len(seq)):
            S.append(states[t]); Lat.append(lat[t])
    return np.array(S), np.array(Lat)


def linear_probe(S, y, steps=500, lr=0.05):
    if S.shape[1] == 0: return 1.0 / (int(y.max()) + 1)
    Xt = torch.tensor(S, dtype=torch.float32); yt = torch.tensor(y, dtype=torch.long)
    K = int(y.max()) + 1
    W = torch.zeros(S.shape[1], K, requires_grad=True); b = torch.zeros(K, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr); lossf = torch.nn.CrossEntropyLoss()
    for _ in range(steps):
        loss = lossf(Xt @ W + b, yt); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return (Xt @ W + b).argmax(1).eq(yt).float().mean().item()


def best_single(S, y):
    best = -1.0
    for d in range(S.shape[1]):
        x = S[:, d]
        for thr in np.unique(np.quantile(x, np.linspace(0, 1, 41))):
            for sgn in (+1, -1):
                pred = (x > thr).astype(int) if sgn > 0 else (x < thr).astype(int)
                best = max(best, (pred == y).mean())
    return best


m = load("runs/expF_unified_dynamic.pt")
SCRATCH = [1, 5, 6]; COMP = [0, 2, 3, 4, 7]

print("=" * 78)
print(f"C1. Decode each op's latent from the SCRATCHPAD {SCRATCH} vs complement {COMP}")
print("    (dynamic model). If the shared 3-dim subspace holds every op's latent, the")
print("    scratchpad probe ~ full-8d probe and the complement probe ~ chance.")
print("=" * 78)
jobs = [("add", None), ("sub", None), ("mul", None), ("div", 2), ("div", 5)]
for op, dd in jobs:
    S, Lat = collect(m, op, n=900, div_d=dd)
    full = linear_probe(S, Lat); scr = linear_probe(S[:, SCRATCH], Lat); comp = linear_probe(S[:, COMP], Lat)
    chance = max(np.bincount(Lat) / len(Lat))
    name = f"{op}/{dd}" if dd else op
    print(f"  {name:6s}: full-8d {full:.3f} | scratchpad{SCRATCH} {scr:.3f} | complement{COMP} {comp:.3f} "
          f"| chance {chance:.3f}")

print("\n" + "=" * 78)
print("C2. CURRICULUM EFFECT — carry vs borrow axis-alignment, DYNAMIC vs CO-TRAIN")
print("    (single-dim decode high = axis-aligned/clean; low+probe-high = distributed)")
print("=" * 78)
for ckpt in ["runs/expF_unified_dynamic.pt", "runs/expF_unified_cotrain.pt"]:
    mm = load(ckpt); tag = ckpt.split("_")[-1].replace(".pt", "")
    print(f"  [{tag}]")
    for op, name in [("add", "carry"), ("sub", "borrow")]:
        S, Lat = collect(mm, op, n=900)
        bs = best_single(S, Lat); pr = linear_probe(S, Lat)
        print(f"    {op} {name:6s}: best-single-dim {bs:.3f}  linear-probe {pr:.3f}  "
              f"-> {'AXIS-ALIGNED' if bs > 0.95 else 'DISTRIBUTED'}")

print("\n" + "=" * 78)
print("C3. CAUSAL carry injection (dynamic model): overwrite ONLY {1,5,6} with the")
print("    carry=1 centroid; on inputs where a+b<10, the output must jump (a+b) -> (a+b+1)")
print("=" * 78)
S, Lat = collect(m, "add", n=1500)
c0 = S[Lat == 0].mean(0); c1 = S[Lat == 1].mean(0)


@torch.no_grad()
def out_from_state(state_vec, da, db):
    x = torch.zeros(1, 2 * base + F.NOP, device=DEV)
    x[0, da] = 1.0; x[0, base + db] = 1.0; x[0, 2 * base + 0] = 1.0  # op=add
    s = torch.tensor(state_vec, dtype=torch.float32, device=DEV).unsqueeze(0)
    logits, _ = m.step(s, x); return int(logits.argmax(-1))


def inject_test(donor_dims):
    """Start from carry=0 centroid; copy donor_dims from carry=1 centroid; expect +1 on output."""
    base_state = c0.copy(); inj = c0.copy(); inj[donor_dims] = c1[donor_dims]
    ok_base = ok_inj = tot = 0
    for da in range(base):
        for db in range(base):
            if da + db >= base: continue  # need no intrinsic carry so the +1 is visible
            tot += 1
            ok_base += (out_from_state(base_state, da, db) == (da + db) % base)
            ok_inj += (out_from_state(inj, da, db) == (da + db + 1) % base)
    return ok_base / tot, ok_inj / tot


b_all, i_all = inject_test([0, 1, 2, 3, 4, 5, 6, 7])
b_scr, i_scr = inject_test(SCRATCH)
b_cmp, i_cmp = inject_test(COMP)
print(f"  carry=0 centroid  -> output==(a+b)%10 on {b_all:.3f} of no-carry pairs (baseline sanity)")
print(f"  inject FULL carry=1 centroid     -> output==(a+b+1)%10 on {i_all:.3f}")
print(f"  inject ONLY scratch dims {SCRATCH}   -> output==(a+b+1)%10 on {i_scr:.3f}  "
      f"(high => those 3 dims ARE the carry, causally)")
print(f"  inject ONLY complement {COMP} -> output==(a+b+1)%10 on {i_cmp:.3f}  (should stay low)")
print("\nINTERP_UNIFIED3 DONE")
