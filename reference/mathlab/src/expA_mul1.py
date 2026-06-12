"""
expA_mul1.py — single-digit x multi-digit multiplication (a is W digits, b in 0..9).
This IS a finite-state transduction: out_t=(a_t*b+carry)%base, carry'=(a_t*b+carry)//base,
with carry in {0..base-2} (9 states for base 10). So the neural Mealy machine should be
able to discover it — but it needs MORE than 2 states, unlike add/sub. Question: what
minimum state_dim is needed, and does it cleanly find the 9-state multiplicative carry?

Uses the general k-means FSM extractor (sign-bits won't resolve 9 states).
Run: python expA_mul1.py
"""
from __future__ import annotations
import random
import torch, torch.nn as nn

import expA_mealy as E
import core_data as cd
import fsm_extract as FX

DEVICE = E.DEVICE


def make_mul1_batch(n, width, base, seed=None):
    rng = random.Random(seed)
    A = [rng.randint(0, base ** width - 1) for _ in range(n)]
    Bd = [rng.randint(0, base - 1) for _ in range(n)]          # single digit
    L = width + 1
    a_oh = E.onehot_seq(A, L, base)
    b_oh = torch.zeros(n, L, base)                              # b broadcast to all positions
    for i, b in enumerate(Bd):
        b_oh[i, :, b] = 1.0
    tgt = torch.zeros(n, L, dtype=torch.long)
    for i in range(n):
        tgt[i] = torch.tensor(cd.to_digits(A[i] * Bd[i], L, base))
    return a_oh, b_oh, tgt


def train_mul1(model, base=10, width=3, steps=6000, bs=256, lr=1e-2, quiet=False,
               train_widths=None):
    """If train_widths is given (e.g. (1,2,3,4,5)), each step samples a width from it
    => the recurrence is trained at MULTIPLE sequence lengths, which discourages the
    state from acting as a position-counter (the cause of length-overfitting)."""
    import random
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    wrng = random.Random(12345)
    for step in range(1, steps + 1):
        w = wrng.choice(train_widths) if train_widths else width
        a_oh, b_oh, tgt = make_mul1_batch(bs, w, base, seed=step)
        a_oh, b_oh, tgt = a_oh.to(DEVICE), b_oh.to(DEVICE), tgt.to(DEVICE)
        logits = model(a_oh, b_oh)
        loss = lossf(logits.reshape(-1, base), tgt.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if not quiet and (step % 1500 == 0 or step == 1):
            print(f"  step {step:5d} loss {loss.item():.4f} (w={w})")
    return model


@torch.no_grad()
def net_predict_mul1(model, base=10):
    model.eval()
    def predict(a, b):
        w = E._ndigits(a, base); L = w + 1
        a_oh = E.onehot_seq([a], L, base).to(DEVICE)
        b_oh = torch.zeros(1, L, base, device=DEVICE); b_oh[0, :, b % base] = 1.0
        logits = model(a_oh, b_oh)[0]
        return cd.from_digits(logits.argmax(-1).tolist(), base)
    return predict


def mul1_lengen(predict, base=10, widths=(1, 2, 3, 4, 6, 8, 12, 16, 20), n=800, seed=0):
    """predict(a,b)->int; exact match against a*b. b is a single digit."""
    rep = {}
    for w in widths:
        rng = random.Random(seed + w); ok = 0
        for _ in range(n):
            a = rng.randint(0, base ** w - 1); b = rng.randint(0, base - 1)
            if predict(a, b) == a * b:
                ok += 1
        rep[w] = ok / n
    return rep


def extract_best_fsm(model, base=10, width=3, kmax=16):
    """Faithful empirical extraction: collect the net's actual transitions, cluster
    states, read out/next tables by majority vote. Sweep #clusters k; return the
    smallest k whose FSM length-generalizes exactly."""
    import numpy as np
    a_oh, b_oh, _ = make_mul1_batch(6000, width, base, seed=999)
    trans = FX.collect_transitions(model, a_oh.to(DEVICE), b_oh.to(DEVICE), DEVICE)
    Sall = np.concatenate([trans[0], trans[4]], axis=0)   # s_prev and s_next
    best = (None, None, None)
    for k in range(2, kmax + 1):
        C = FX.kmeans(Sall, k, seed=0)
        _, _, _, fsm_predict = FX.build_fsm_empirical(model, base, C, trans, DEVICE, E._ndigits)
        rep = mul1_lengen(lambda a, b: fsm_predict(a, b, True), base,
                          widths=(3, 6, 12, 20), n=300)
        if min(rep.values()) > 0.999:
            return k, fsm_predict, rep
        if best[0] is None or min(rep.values()) > min(best[2].values()):
            best = (k, fsm_predict, rep)
    return None, best[1], best[2]   # none perfect; return best attempt


if __name__ == "__main__":
    import os, argparse
    os.makedirs("runs", exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=int, nargs="+", default=[4, 6])
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--train_widths", type=int, nargs="+", default=None,
                    help="if set, train across these sequence lengths (anti length-overfit)")
    ap.add_argument("--noise", type=float, default=0.0,
                    help="state-noise discreteness bottleneck (e.g. 0.1)")
    args = ap.parse_args()
    widths = (1, 2, 3, 4, 6, 8, 12, 16, 20)
    tw = tuple(args.train_widths) if args.train_widths else None
    for d in args.dims:
        tag = f"train_widths={tw}" if tw else "train_width=3"
        print(f"\n=== single-digit mult, d={d}, hidden={args.hidden}, noise={args.noise} ({tag}) ===")
        torch.manual_seed(0)
        model = E.NeuralMealy(base=10, state_dim=d, hidden=args.hidden, state_noise=args.noise)
        print(f"  params: {sum(p.numel() for p in model.parameters())}")
        train_mul1(model, base=10, width=3, steps=args.steps, train_widths=tw)
        net_rep = mul1_lengen(net_predict_mul1(model), base=10, widths=widths, n=800)
        print("  NET len-gen exact-acc: " + "  ".join(f"w{w}:{net_rep[w]:.3f}" for w in widths))
        if min(net_rep.values()) > 0.999:                       # save when NET generalizes
            torch.save(model.state_dict(), f"runs/expA_mul1_d{d}_h{args.hidden}_n{args.noise}.pt")
        k, fsm_predict, fsm_rep = extract_best_fsm(model, base=10, width=3)
        if k:
            full = mul1_lengen(lambda a, b: fsm_predict(a, b, True), base=10, widths=widths, n=800)
            print(f"  FSM ({k} states) len-gen exact-acc: " +
                  "  ".join(f"w{w}:{full[w]:.3f}" for w in widths))
            if min(full.values()) > 0.999:
                print(f"  >>> {k}-state FSM length-generalizes perfectly (EXTRACTED & verified).")
        else:
            print(f"  FSM extraction best-effort @w20: {fsm_rep.get(20)} "
                  f"(net generalizes={min(net_rep.values())>0.999}).")
