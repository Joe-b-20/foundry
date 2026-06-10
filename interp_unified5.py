"""
interp_unified5.py — mult-carry causal test done right: inject REAL visited carry=k states
(not centroids, which lie off the curved manifold). Tests two claims:
  (1) a real carry=k state, fed ARBITRARY digits, outputs (a_t*b + k)%10  -> the state is a
      faithful Mealy 'carry=k';  (2) transplanting ONLY the scratchpad {1,5,6} of that real
      state onto a carry=0 base reproduces it -> the carry lives in the scratchpad, causally.
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
m = F.UnifiedMealy(base=base, state_dim=8, hidden=96)
m.load_state_dict(torch.load("runs/expF_unified_dynamic.pt", map_location=DEV)); m.to(DEV); m.eval()


@torch.no_grad()
def trace_mul(a, b):
    w = E._ndigits(a, base); L = w + 1; ad = cd.to_digits(a, L, base)
    s = m.s0.unsqueeze(0).to(DEV); states = []
    for t in range(L):
        states.append(s.squeeze(0).cpu().numpy().copy())  # state BEFORE step t (encodes carry_t)
        x = torch.zeros(1, 2 * base + F.NOP, device=DEV)
        x[0, ad[t]] = 1.0; x[0, base + (b % base)] = 1.0; x[0, 2 * base + 2] = 1.0
        _, s = m.step(s, x)
    return ad, states


# collect REAL states grouped by incoming mult-carry value
bypk = {k: [] for k in range(9)}
rng = random.Random(0)
for _ in range(4000):
    w = rng.choice([2, 3, 4, 6]); a = rng.randint(0, base ** w - 1); b = rng.randint(0, base - 1)
    ad, states = trace_mul(a, b)
    c = 0
    for t in range(len(ad)):
        if len(bypk[c]) < 40: bypk[c].append(states[t].copy())
        c = (ad[t] * b + c) // base


@torch.no_grad()
def mul_out(state_vec, da, db):
    x = torch.zeros(1, 2 * base + F.NOP, device=DEV)
    x[0, da] = 1.0; x[0, base + db] = 1.0; x[0, 2 * base + 2] = 1.0
    s = torch.tensor(state_vec, dtype=torch.float32, device=DEV).unsqueeze(0)
    logits, _ = m.step(s, x); return int(logits.argmax(-1))


base0 = np.mean(bypk[0], 0)  # a representative carry=0 region (for transplant base)
print("=" * 78)
print("REAL-state mult-carry injection (dynamic model). Each carry=k uses actual visited")
print("states; output target = (a_t*b + k) % 10 over the full 10x10 digit/multiplier grid.")
print("=" * 78)
print("    k  | real-state-as-is acc | scratchpad{1,5,6}-transplant acc | n_states")
for k in range(9):
    if not bypk[k]: continue
    accs_real = []; accs_scr = []
    for sv in bypk[k]:
        scr = base0.copy(); scr[SCRATCH] = sv[SCRATCH]
        okr = oks = 0
        for da in range(base):
            for db in range(base):
                tgt = (da * db + k) % base
                okr += (mul_out(sv, da, db) == tgt); oks += (mul_out(scr, da, db) == tgt)
        accs_real.append(okr / 100); accs_scr.append(oks / 100)
    print(f"    {k}  |        {np.mean(accs_real):.3f}         |            {np.mean(accs_scr):.3f}            |   {len(bypk[k])}")
print("\n(High real-state acc => the visited state faithfully encodes carry=k for ALL inputs;")
print(" high transplant acc => that carry lives in the scratchpad {1,5,6}. Centroids failed")
print(" only because the 9-state mult-carry is a CURVED manifold — its mean is off-manifold.)")
print("\nINTERP_UNIFIED5 DONE")
