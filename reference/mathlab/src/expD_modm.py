"""
expD_modm.py — CONFIRM the division diagnosis by isolating STATE MAINTENANCE from output.

Diagnostic (TRACKER): division fails because the per-step state update is rem'=(rem*base+a_t)%d
with d != base — a NON-base-modulus counter the digit-serial net can't maintain over length;
whereas +,x reduce their state via the BASE (a free shift). Pure isolation test: feed the
digits of n MSB-first and ask only for the RUNNING PREFIX VALUE mod m at each step (NO quotient
output). The required state is exactly n_prefix mod m (m states); transition r'=(r*base+a_t)%m.
  - m = base (10): 10≡0, so r'=a_t  -> MEMORYLESS -> must be trivial (control).
  - m = 9: 10≡1, so r'=(r+a_t)%9     -> running digit-sum mod 9 (accumulator, 9 states).
  - m = 7: 10≡3, so r'=(3r+a_t)%7    -> multiply-by-3 then add (7 states), the division case.
Prediction: m=10 length-generalizes; m=7 (and likely 9) do NOT, mirroring division. If m=7
fails, the blocker is non-base modular STATE MAINTENANCE (not the quotient output). If m=7
succeeds, the quotient map was the blocker instead. Either way the diagnosis is pinned down.

Run: python expD_modm.py --mods 10 9 7 --dims 6 --steps 6000 --train_widths 1 2 3 4 5
"""
from __future__ import annotations
import argparse, random
import torch, torch.nn as nn

import core_data as cd
import expA_mealy as E

DEVICE = E.DEVICE


def make_modm_batch(n, width, base, m, seed):
    rng = random.Random(seed)
    N = [rng.randint(0, base ** width - 1) for _ in range(n)]
    L = width
    a_oh = torch.zeros(n, L, base); b_oh = torch.zeros(n, L, base)  # b unused (digit 0)
    tgt = torch.zeros(n, L, dtype=torch.long)
    for i in range(n):
        nm = list(reversed(cd.to_digits(N[i], L, base)))   # MSB-first digits
        r = 0
        for t in range(L):
            a_oh[i, t, nm[t]] = 1.0
            b_oh[i, t, 0] = 1.0
            r = (r * base + nm[t]) % m
            tgt[i, t] = r                                    # running prefix value mod m
    return a_oh, b_oh, tgt


def train_modm(model, base, m, steps, bs=256, lr=1e-2, train_widths=None):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    lossf = nn.CrossEntropyLoss()
    wrng = random.Random(5)
    for step in range(1, steps + 1):
        w = wrng.choice(train_widths) if train_widths else 3
        a_oh, b_oh, tgt = make_modm_batch(bs, w, base, m, seed=step)
        a_oh, b_oh, tgt = a_oh.to(DEVICE), b_oh.to(DEVICE), tgt.to(DEVICE)
        logits = model(a_oh, b_oh)
        loss = lossf(logits.reshape(-1, base), tgt.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 2000 == 0 or step == 1:
            print(f"    step {step:5d} loss {loss.item():.5f}")
    return model


@torch.no_grad()
def modm_lengen(model, base, m, widths, n=800, seed=0):
    model.eval()
    rep = {}
    for w in widths:
        rng = random.Random(seed + w); ok = 0
        L = w
        for _ in range(n):
            x = rng.randint(0, base ** w - 1)
            nm = list(reversed(cd.to_digits(x, L, base)))
            a_oh = torch.zeros(1, L, base, device=DEVICE); b_oh = torch.zeros(1, L, base, device=DEVICE)
            for t in range(L):
                a_oh[0, t, nm[t]] = 1.0; b_oh[0, t, 0] = 1.0
            last = model(a_oh, b_oh)[0, -1].argmax().item()   # final-position output = n mod m
            if last == x % m:
                ok += 1
        rep[w] = ok / n
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mods", type=int, nargs="+", default=[10, 9, 7])
    ap.add_argument("--dims", type=int, default=6)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--train_widths", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    args = ap.parse_args()
    tw = tuple(args.train_widths) if args.train_widths else None
    widths = (1, 2, 3, 4, 6, 8, 12, 16, 20)
    base = 10
    print(f"device={DEVICE}  running-prefix mod m (MSB-first), continuous NeuralMealy")
    for m in args.mods:
        note = "= base, MEMORYLESS" if base % m == 0 else f"coprime-ish (10 mod {m} = {10 % m})"
        print(f"\n=== mod {m} ({note})  d={args.dims} hidden={args.hidden} train_widths={tw} ===")
        torch.manual_seed(0)
        model = E.NeuralMealy(base=base, state_dim=args.dims, hidden=args.hidden)
        train_modm(model, base, m, args.steps, train_widths=tw)
        rep = modm_lengen(model, base, m, widths)
        print("  len-gen (final n mod m exact): " + "  ".join(f"w{w}:{rep[w]:.3f}" for w in widths))
