"""
interp_unified6.py — is the multiplication carry ENTANGLED with the multiplier b?
Hypothesis: a real carry=k state encodes 'carry=k in the context of multiplier b0', not a
context-free 'carry=k'. Test: feed each real state its MATCHING b0 (over all da) vs a
MISMATCHED b. If matched~1.0 and mismatched low, the carry is b-coupled (explains the smooth
manifold / why discreteness hurt mult / why a clean 9-state FSM never extracted).
"""
import random
import numpy as np
import torch
import core_data as cd
import expA_mealy as E
import expF_unified as F

base = 10; DEV = F.DEVICE
torch.manual_seed(0)
m = F.UnifiedMealy(base=base, state_dim=8, hidden=96)
m.load_state_dict(torch.load("runs/expF_unified_dynamic.pt", map_location=DEV)); m.to(DEV); m.eval()


@torch.no_grad()
def trace_mul(a, b):
    w = E._ndigits(a, base); L = w + 1; ad = cd.to_digits(a, L, base)
    s = m.s0.unsqueeze(0).to(DEV); states = []
    for t in range(L):
        states.append(s.squeeze(0).cpu().numpy().copy())
        x = torch.zeros(1, 2 * base + F.NOP, device=DEV)
        x[0, ad[t]] = 1.0; x[0, base + (b % base)] = 1.0; x[0, 2 * base + 2] = 1.0
        _, s = m.step(s, x)
    return ad, states


# collect (state, carry_k, b0) tuples from real runs
samples = {k: [] for k in range(9)}
rng = random.Random(0)
for _ in range(6000):
    w = rng.choice([2, 3, 4, 6]); a = rng.randint(0, base ** w - 1); b = rng.randint(0, base - 1)
    ad, states = trace_mul(a, b); c = 0
    for t in range(len(ad)):
        if len(samples[c]) < 60: samples[c].append((states[t].copy(), b))
        c = (ad[t] * b + c) // base


@torch.no_grad()
def mul_out(state_vec, da, db):
    x = torch.zeros(1, 2 * base + F.NOP, device=DEV)
    x[0, da] = 1.0; x[0, base + db] = 1.0; x[0, 2 * base + 2] = 1.0
    s = torch.tensor(state_vec, dtype=torch.float32, device=DEV).unsqueeze(0)
    logits, _ = m.step(s, x); return int(logits.argmax(-1))


print("=" * 78)
print("Mult-carry: MATCHED multiplier b0 vs MISMATCHED. target out=(da*b + k)%10 over da=0..9")
print("=" * 78)
print("    k  | matched-b0 acc | mismatched-b acc | n")
rng2 = random.Random(1)
for k in range(9):
    if not samples[k]: continue
    macc = []; macc_mis = []
    for (sv, b0) in samples[k]:
        okm = sum(mul_out(sv, da, b0) == (da * b0 + k) % base for da in range(base)) / base
        bmis = rng2.choice([x for x in range(base) if x != b0])
        okx = sum(mul_out(sv, da, bmis) == (da * bmis + k) % base for da in range(base)) / base
        macc.append(okm); macc_mis.append(okx)
    print(f"    {k}  |     {np.mean(macc):.3f}      |      {np.mean(macc_mis):.3f}       | {len(samples[k])}")
print("\n(matched~1.0 & mismatched~chance => the multiplication carry is encoded JOINTLY with")
print(" the multiplier; it is a faithful Mealy state only when the same b is supplied. This is")
print(" the mechanistic reason mult-carry reads as a smooth/entangled manifold, not 9 clean states.)")
print("\nINTERP_UNIFIED6 DONE")
