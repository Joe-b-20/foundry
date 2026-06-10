"""
expE_shared.py — does ONE op-conditioned model SHARE a mechanism across + and -?

The PROMPT's seed question: when subtraction is mixed in, does the model find that subtraction
is addition+negation, or borrow, or something else? Session 1 answered "borrow" but trained a
SEPARATE model per op (audit: "no operation vectors exist"). This tests the harder, more
direct version: a SINGLE Mealy machine that takes an OP-CODE input (add|sub) and is trained on
BOTH jointly. Question: does it reuse ONE state mechanism for both ops, or partition into two?

Carry and borrow are mirror images — both are a single "did we cross a base boundary" bit
(carry: a+b>=base; borrow: a-b<0). HYPOTHESIS (stated before running): a single SCALAR state
(d=1) conditioned on the op-code should handle BOTH, i.e. +/- share one boundary-bit machine
whose only op-dependent part is the OUTPUT digit map. If d=1 length-generalizes on both ops,
that's evidence of a shared mechanism (subtraction is NOT add+negate; it reuses the carry
state, sign-flipped). Honestly unsure whether d=1 suffices or it needs separate bits.

Run: python expE_shared.py --dims 1 2 --steps 8000 --train_widths 1 2 3 4 5
"""
from __future__ import annotations
import argparse, random
import torch, torch.nn as nn

import core_data as cd
import expA_mealy as E

DEVICE = E.DEVICE
OPS = ["add", "sub"]


class SharedMealy(nn.Module):
    """Mealy machine with an extra op-code input fed every step: input = [state ; onehot(a) ;
    onehot(b) ; onehot(op)]."""
    def __init__(self, base=10, state_dim=1, hidden=16, n_ops=2):
        super().__init__()
        self.base = base; self.state_dim = state_dim; self.n_ops = n_ops
        in_dim = state_dim + 2 * base + n_ops
        self.f = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh(),
                               nn.Linear(hidden, state_dim), nn.Tanh())
        self.g = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh(),
                               nn.Linear(hidden, base))
        self.s0 = nn.Parameter(torch.zeros(state_dim))

    def step(self, s_prev, x):
        z = torch.cat([s_prev, x], dim=-1)
        return self.g(z), self.f(z)

    def forward(self, a_oh, b_oh, op_oh):
        N, L, _ = a_oh.shape
        s = self.s0.unsqueeze(0).expand(N, -1)
        outs = []
        for t in range(L):
            x = torch.cat([a_oh[:, t], b_oh[:, t], op_oh[:, t]], dim=-1)
            logits, s = self.step(s, x)
            outs.append(logits)
        return torch.stack(outs, dim=1)


def make_batch(n, width, base, op, seed):
    rng = random.Random(seed)
    A = [rng.randint(0, base ** width - 1) for _ in range(n)]
    Bn = [rng.randint(0, base ** width - 1) for _ in range(n)]
    if op == "sub":
        for i in range(n):
            if A[i] < Bn[i]: A[i], Bn[i] = Bn[i], A[i]
    L = width + 1
    a_oh = E.onehot_seq(A, L, base); b_oh = E.onehot_seq(Bn, L, base)
    op_idx = OPS.index(op)
    op_oh = torch.zeros(n, L, len(OPS)); op_oh[:, :, op_idx] = 1.0
    tgt = torch.zeros(n, L, dtype=torch.long)
    for i in range(n):
        r = cd.exact_result(A[i], Bn[i], op)
        tgt[i] = torch.tensor(cd.to_digits(r, L, base))
    return a_oh, b_oh, op_oh, tgt


def train(model, base, steps, bs=256, lr=1e-2, train_widths=None):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    wrng = random.Random(31); orng = random.Random(57)
    for step in range(1, steps + 1):
        w = wrng.choice(train_widths) if train_widths else 3
        op = orng.choice(OPS)                       # joint training: each batch is add OR sub
        a_oh, b_oh, op_oh, tgt = make_batch(bs, w, base, op, seed=step)
        a_oh, b_oh, op_oh, tgt = a_oh.to(DEVICE), b_oh.to(DEVICE), op_oh.to(DEVICE), tgt.to(DEVICE)
        logits = model(a_oh, b_oh, op_oh)
        loss = lossf(logits.reshape(-1, base), tgt.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 2000 == 0 or step == 1:
            print(f"    step {step:5d} loss {loss.item():.5f}")
    return model


@torch.no_grad()
def predict_fn(model, base, op):
    model.eval()
    op_idx = OPS.index(op)
    def predict(a, b):
        width = max(E._ndigits(a, base), E._ndigits(b, base)); L = width + 1
        a_oh = E.onehot_seq([a], L, base).to(DEVICE); b_oh = E.onehot_seq([b], L, base).to(DEVICE)
        op_oh = torch.zeros(1, L, len(OPS), device=DEVICE); op_oh[:, :, op_idx] = 1.0
        digits = model(a_oh, b_oh, op_oh)[0].argmax(-1).tolist()
        return cd.from_digits(digits, base)
    return predict


@torch.no_grad()
def analyze_states(model, base):
    """For d=1: characterize the scalar state per op. Are the SAME state values used for add and
    sub? Read the 2-state machine for each op and compare the boundary-crossing transition."""
    model.eval()
    print("  --- shared-mechanism analysis (per op, holding op-code fixed) ---")
    for op in OPS:
        op_idx = OPS.index(op)
        # probe over all (a,b,state-sign) to read out the 2 states' output maps + transitions
        # start state sign:
        s0 = model.s0.detach()
        def run_state_signs():
            # collect visited state signs on random probes
            rng = random.Random(5)
            A = [rng.randint(0, base**3 - 1) for _ in range(2000)]
            Bn = [rng.randint(0, base**3 - 1) for _ in range(2000)]
            if op == "sub":
                A, Bn = zip(*[(max(a, b), min(a, b)) for a, b in zip(A, Bn)]); A, Bn = list(A), list(Bn)
            L = 4
            a_oh = E.onehot_seq(A, L, base).to(DEVICE); b_oh = E.onehot_seq(Bn, L, base).to(DEVICE)
            op_oh = torch.zeros(len(A), L, len(OPS), device=DEVICE); op_oh[:, :, op_idx] = 1.0
            s = model.s0.unsqueeze(0).expand(len(A), -1).to(DEVICE)
            signs = set()
            signs.add(tuple((s[0] > 0).int().tolist()))
            for t in range(L):
                x = torch.cat([a_oh[:, t], b_oh[:, t], op_oh[:, t]], dim=-1)
                _, s = model.step(s, x)
                for i in range(min(len(A), 500)):
                    signs.add(tuple((s[i] > 0).int().tolist()))
            return signs
        signs = run_state_signs()
        # for each visited sign-state, read output map vs (a +/- b + k) and the transition rule
        desc = []
        comb = (lambda da, db: da + db) if op == "add" else (lambda da, db: da - db)
        for st in sorted(signs):
            sval = torch.tensor([1.0 if b else -1.0 for b in st], device=DEVICE).unsqueeze(0)
            # infer k from (0,0)
            x00 = torch.zeros(1, 2 * base + len(OPS), device=DEVICE); x00[0, 0] = 1; x00[0, base] = 1; x00[0, 2 * base + op_idx] = 1
            k = (int(model.g(torch.cat([sval, x00], -1)).argmax(-1)) - comb(0, 0)) % base
            ok = True
            for da in range(base):
                for db in range(base):
                    x = torch.zeros(1, 2 * base + len(OPS), device=DEVICE); x[0, da] = 1; x[0, base + db] = 1; x[0, 2 * base + op_idx] = 1
                    if int(model.g(torch.cat([sval, x], -1)).argmax(-1)) != (comb(da, db) + k) % base:
                        ok = False; break
                if not ok: break
            sign = "+" if op == "add" else "-"
            desc.append(f"state{st}: out=(a{sign}b+{k})%{base}" if ok else f"state{st}: NONLINEAR")
        print(f"    [{op}] visited state-signs={sorted(signs)}  ->  " + " | ".join(desc))


if __name__ == "__main__":
    import os
    os.makedirs("runs", exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--hidden", type=int, default=16)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--train_widths", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = ap.parse_args()
    tw = tuple(args.train_widths) if args.train_widths else None
    widths = (1, 2, 3, 4, 6, 8, 12, 16, 20)
    print(f"device={DEVICE}  op-conditioned SharedMealy trained JOINTLY on +,-")
    combos = [(d, sd) for d in args.dims for sd in args.seeds]
    for d, sd in combos:
        print(f"\n=== shared +/- d={d} seed={sd} hidden={args.hidden} train_widths={tw} ===")
        torch.manual_seed(sd)
        model = SharedMealy(base=10, state_dim=d, hidden=args.hidden)
        print(f"  params: {sum(p.numel() for p in model.parameters())}")
        train(model, 10, args.steps, train_widths=tw)
        for op in OPS:
            rep = cd.length_gen_report(predict_fn(model, 10, op), op, base=10,
                                       widths=widths, n_per_width=800)
            print(f"  [{op}] len-gen: " + "  ".join(f"w{w}:{rep[w]:.3f}" for w in widths))
        if d == 1:
            analyze_states(model, 10)
