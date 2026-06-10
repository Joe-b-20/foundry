"""
expMul_full.py — THE WALL. Full multi-digit x multi-digit multiplication with the
SAME fixed-state neural Mealy machine. Theory: n x n multiplication is NOT a
finite-state transduction (product digit k = sum_{i+j=k} a_i b_j + carry; the column
sum grows without bound as n grows), so any FIXED-state machine must fail to
length-generalize, no matter how it's trained. We confirm this empirically and
contrast with single-digit mult (expA_mul1), which DOES generalize.

Format: feed a,b digit-serial LSB-first for W steps, then W zero steps; the machine
emits 2W output digits (the product). Train width 3, test longer.
Run: python expMul_full.py
"""
from __future__ import annotations
import random
import torch, torch.nn as nn

import expA_mealy as E
import core_data as cd

DEVICE = E.DEVICE


def make_fullmult_batch(n, width, base, seed=None):
    rng = random.Random(seed)
    A = [rng.randint(0, base ** width - 1) for _ in range(n)]
    Bn = [rng.randint(0, base ** width - 1) for _ in range(n)]
    L = 2 * width                       # product has up to 2*width digits
    a_oh = E.onehot_seq(A, L, base)     # a's digits in 0..width-1, zeros after
    b_oh = E.onehot_seq(Bn, L, base)
    tgt = torch.zeros(n, L, dtype=torch.long)
    for i in range(n):
        tgt[i] = torch.tensor(cd.to_digits(A[i] * Bn[i], L, base))
    return a_oh, b_oh, tgt


def train_fullmult(model, base=10, width=3, steps=8000, bs=256, lr=5e-3):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    for step in range(1, steps + 1):
        a_oh, b_oh, tgt = make_fullmult_batch(bs, width, base, seed=step)
        a_oh, b_oh, tgt = a_oh.to(DEVICE), b_oh.to(DEVICE), tgt.to(DEVICE)
        logits = model(a_oh, b_oh)
        loss = lossf(logits.reshape(-1, base), tgt.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 2000 == 0 or step == 1:
            print(f"  step {step:5d} loss {loss.item():.4f}")
    return model


@torch.no_grad()
def net_predict_fullmult(model, base=10):
    model.eval()
    def predict(a, b):
        width = max(E._ndigits(a, base), E._ndigits(b, base))
        L = 2 * width
        a_oh = E.onehot_seq([a], L, base).to(DEVICE)
        b_oh = E.onehot_seq([b], L, base).to(DEVICE)
        logits = model(a_oh, b_oh)[0]
        return cd.from_digits(logits.argmax(-1).tolist(), base)
    return predict


def fullmult_lengen(predict, base=10, widths=(1, 2, 3, 4, 5, 6, 8), n=800, seed=0):
    rep = {}
    for w in widths:
        rng = random.Random(seed + w); ok = 0
        for _ in range(n):
            a = rng.randint(0, base ** w - 1); b = rng.randint(0, base ** w - 1)
            if predict(a, b) == a * b:
                ok += 1
        rep[w] = ok / n
    return rep


if __name__ == "__main__":
    widths = (1, 2, 3, 4, 5, 6, 8)
    # try a few state sizes; even large fixed state should fail to length-generalize
    for d in (4, 16, 64):
        print(f"\n=== full mult, state_dim={d} (base=10, train_width=3) ===")
        torch.manual_seed(0)
        model = E.NeuralMealy(base=10, state_dim=d, hidden=64)
        print(f"  params: {sum(p.numel() for p in model.parameters())}")
        train_fullmult(model, base=10, width=3, steps=8000)
        rep = fullmult_lengen(net_predict_fullmult(model), base=10, widths=widths)
        print("  NET len-gen exact-acc: " + "  ".join(f"w{w}:{rep[w]:.3f}" for w in widths))
        print(f"  (train width=3 is w3={rep[3]:.3f}; watch w4..w8 collapse -> the wall)")
