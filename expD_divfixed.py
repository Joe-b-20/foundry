"""
expD_divfixed.py — DIAGNOSTIC: does division length-generalize when the divisor is FIXED?

Session 1 + the discrete-state run both failed to length-generalize a//d for VARIABLE
single-digit d (1..9). Honest open question: is the blocker (i) dividing by a *variable*
divisor (the per-step map (r,a_t,d)->(q,r') is a 3-input function), or (ii) the MSB-first
division map / remainder automaton itself?

This isolates it: train with a SINGLE fixed divisor (architecture unchanged — d is still
fed each step, just constant). The remainder automaton for fixed d has exactly d states
(r=0..d-1). If THIS length-generalizes, the variable divisor was the blocker. If even this
fails, the division remainder automaton itself is what the Mealy machine can't find.

Run: python expD_divfixed.py --divisor 7 --dims 3 4 --steps 8000 --train_widths 1 2 3 4 5
"""
from __future__ import annotations
import argparse, random
import torch, torch.nn as nn

import core_data as cd
import expA_mealy as E
import expA_div1 as D

DEVICE = E.DEVICE
# Diagnostic uses the CONTINUOUS NeuralMealy (session-1 arch) on purpose: it fits well,
# so any length-gen failure here is about division itself, not the discrete-state confound.


def make_fixed_batch(n, width, base, dfix, seed):
    rng = random.Random(seed)
    A = [rng.randint(0, base ** width - 1) for _ in range(n)]
    L = width
    a_oh = torch.zeros(n, L, base); b_oh = torch.zeros(n, L, base)
    tgt = torch.zeros(n, L, dtype=torch.long)
    for i in range(n):
        am = D.digits_msb(A[i], L, base); qm = D.digits_msb(A[i] // dfix, L, base)
        for t in range(L):
            a_oh[i, t, am[t]] = 1.0; b_oh[i, t, dfix] = 1.0; tgt[i, t] = qm[t]
    return a_oh, b_oh, tgt


@torch.no_grad()
def predict_fixed(model, base, dfix):
    model.eval()
    def predict(a, _ignored=None):
        W = E._ndigits(a, base); L = W
        a_oh = torch.zeros(1, L, base, device=DEVICE); b_oh = torch.zeros(1, L, base, device=DEVICE)
        am = D.digits_msb(a, L, base)
        for t in range(L):
            a_oh[0, t, am[t]] = 1.0; b_oh[0, t, dfix] = 1.0
        qm = model(a_oh, b_oh)[0].argmax(-1).tolist()
        q = 0
        for dig in qm:
            q = q * base + dig
        return q
    return predict


def lengen_fixed(predict, base, dfix, widths, n=800, seed=0):
    rep = {}
    for w in widths:
        rng = random.Random(seed + w); ok = 0
        for _ in range(n):
            a = rng.randint(0, base ** w - 1)
            if predict(a) == a // dfix:
                ok += 1
        rep[w] = ok / n
    return rep


def train_fixed(model, base, dfix, steps, bs=256, lr=1e-2, train_widths=None):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    lossf = nn.CrossEntropyLoss()
    wrng = random.Random(7)
    for step in range(1, steps + 1):
        w = wrng.choice(train_widths) if train_widths else 3
        a_oh, b_oh, tgt = make_fixed_batch(bs, w, base, dfix, seed=step)
        a_oh, b_oh, tgt = a_oh.to(DEVICE), b_oh.to(DEVICE), tgt.to(DEVICE)
        logits = model(a_oh, b_oh)
        loss = lossf(logits.reshape(-1, base), tgt.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 2000 == 0 or step == 1:
            print(f"  step {step:5d} loss {loss.item():.5f} (w={w})")
    return model


if __name__ == "__main__":
    import os
    os.makedirs("runs", exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--divisors", type=int, nargs="+", default=[7])
    ap.add_argument("--base", type=int, default=10)
    ap.add_argument("--dims", type=int, nargs="+", default=[4])
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--train_widths", type=int, nargs="+", default=None)
    args = ap.parse_args()
    tw = tuple(args.train_widths) if args.train_widths else None
    base = args.base
    widths = (1, 2, 3, 4, 6, 8, 12, 16, 20)
    print(f"device={DEVICE}  FIXED-divisor division, base {base}. PREDICTION: divides base "
          f"=> length-generalizes (base-modular remainder); else fails.")
    for dfix in args.divisors:
        divides = "DIVIDES base -> predict PASS" if base % dfix == 0 else "coprime-ish -> predict FAIL"
        for d in args.dims:
            tag = f"train_widths={tw}" if tw else "train_width=3"
            print(f"\n=== /{dfix} (base {base}, {divides})  d={d} hidden={args.hidden} ({tag}) ===")
            torch.manual_seed(0)
            model = E.NeuralMealy(base=base, state_dim=d, hidden=args.hidden)
            train_fixed(model, base, dfix, args.steps, train_widths=tw)
            rep = lengen_fixed(predict_fixed(model, base, dfix), base, dfix, widths)
            print("  NET len-gen exact-acc: " + "  ".join(f"w{w}:{rep[w]:.3f}" for w in widths))
