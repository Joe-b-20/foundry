"""
expA_div1.py — single-digit-divisor division: a (W digits) / d (one digit 1..9) -> a//d.
Completes the +,-,x,/ curriculum and tests a NEW processing direction: division is
naturally MSB-FIRST (long division goes high digit -> low), whereas +,-,x were LSB-first.
This IS a finite-state transduction: process digits MSB-first, state = running remainder
(0..d-1); at each digit  value = state*base + a_t;  q_t = value//d;  state = value % d.
So the Mealy machine should discover the REMAINDER AUTOMATON (state = remainder).

Headline metric: NET exact length-generalization (train width 3, test to width 20).
(Extraction of the >2-state smooth automaton mirrors the single-digit-mult case and is
expected partial; we report net generalization as the primary result.)
Run: python expA_div1.py
"""
from __future__ import annotations
import random
import torch, torch.nn as nn

import expA_mealy as E
import core_data as cd

DEVICE = E.DEVICE


def digits_msb(n, width, base):
    return list(reversed(cd.to_digits(n, width, base)))    # most-significant digit first


def make_div1_batch(n, width, base, seed=None):
    rng = random.Random(seed)
    A = [rng.randint(0, base ** width - 1) for _ in range(n)]
    D = [rng.randint(1, base - 1) for _ in range(n)]        # divisor 1..base-1 (no /0)
    L = width
    a_oh = torch.zeros(n, L, base); b_oh = torch.zeros(n, L, base)
    tgt = torch.zeros(n, L, dtype=torch.long)
    for i in range(n):
        am = digits_msb(A[i], L, base)
        qm = digits_msb(A[i] // D[i], L, base)
        for t in range(L):
            a_oh[i, t, am[t]] = 1.0
            b_oh[i, t, D[i]] = 1.0
            tgt[i, t] = qm[t]
    return a_oh, b_oh, tgt


def train_div1(model, base=10, width=3, steps=8000, bs=256, lr=1e-2, train_widths=None):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)  # decay -> fit precisely
    lossf = nn.CrossEntropyLoss()
    wrng = random.Random(99)
    for step in range(1, steps + 1):
        w = wrng.choice(train_widths) if train_widths else width
        a_oh, b_oh, tgt = make_div1_batch(bs, w, base, seed=step)
        a_oh, b_oh, tgt = a_oh.to(DEVICE), b_oh.to(DEVICE), tgt.to(DEVICE)
        logits = model(a_oh, b_oh)
        loss = lossf(logits.reshape(-1, base), tgt.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 2000 == 0 or step == 1:
            print(f"  step {step:5d} loss {loss.item():.4f}")
    return model


@torch.no_grad()
def net_predict_div1(model, base=10):
    model.eval()
    def predict(a, d):
        W = E._ndigits(a, base); L = W
        a_oh = torch.zeros(1, L, base, device=DEVICE)
        b_oh = torch.zeros(1, L, base, device=DEVICE)
        am = digits_msb(a, L, base)
        for t in range(L):
            a_oh[0, t, am[t]] = 1.0
            b_oh[0, t, d % base] = 1.0
        qm = model(a_oh, b_oh)[0].argmax(-1).tolist()       # MSB-first quotient digits
        q = 0
        for dig in qm:
            q = q * base + dig
        return q
    return predict


def div1_lengen(predict, base=10, widths=(1, 2, 3, 4, 6, 8, 12, 16, 20), n=800, seed=0):
    rep = {}
    for w in widths:
        rng = random.Random(seed + w); ok = 0
        for _ in range(n):
            a = rng.randint(0, base ** w - 1); d = rng.randint(1, base - 1)
            if predict(a, d) == a // d:
                ok += 1
        rep[w] = ok / n
    return rep


if __name__ == "__main__":
    import os, argparse
    os.makedirs("runs", exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=int, nargs="+", default=[4, 6])
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--train_widths", type=int, nargs="+", default=None)
    ap.add_argument("--noise", type=float, default=0.0)
    args = ap.parse_args()
    tw = tuple(args.train_widths) if args.train_widths else None
    widths = (1, 2, 3, 4, 6, 8, 12, 16, 20)
    for d in args.dims:
        tag = f"train_widths={tw}" if tw else "train_width=3"
        print(f"\n=== single-digit division, d={d}, hidden={args.hidden}, noise={args.noise} ({tag}, MSB-first) ===")
        torch.manual_seed(0)
        model = E.NeuralMealy(base=10, state_dim=d, hidden=args.hidden, state_noise=args.noise)
        print(f"  params: {sum(p.numel() for p in model.parameters())}")
        train_div1(model, base=10, width=3, steps=args.steps, train_widths=tw)
        rep = div1_lengen(net_predict_div1(model), base=10, widths=widths, n=800)
        print("  NET len-gen exact-acc: " + "  ".join(f"w{w}:{rep[w]:.3f}" for w in widths))
        if min(rep.values()) > 0.999:
            torch.save(model.state_dict(), f"runs/expA_div1_d{d}_h{args.hidden}.pt")
            print(f"  >>> single-digit division length-generalizes (NET discovers the remainder automaton).")
