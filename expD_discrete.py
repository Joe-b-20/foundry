"""
expD_discrete.py — DISCRETE-state Mealy machine via a straight-through estimator (STE).

Motivation (TRACKER session-2 plan): session-1's NeuralMealy used a continuous tanh
state. It self-quantizes cleanly only for 2-state problems (carry/borrow). For >2-state
problems the state is a SMOOTH manifold, which (a) makes FSM extraction lossy (mult ~0.94,
needs k-means) and (b) lets division DRIFT over length. Fix: make the state LITERALLY
binary {-1,+1}^d with a sign-STE (hard forward, tanh-gradient backward). Then:
  * the eval state is exactly a hypercube vertex => no drift possible;
  * extraction is EXACT BY CONSTRUCTION: enumerate reachable vertices by BFS over all
    (a,b) digit inputs; the resulting FSM == the net at eval, bit-for-bit (no clustering,
    no centroid replay, no majority vote).

The only open question is whether such a net can LEARN a length-generalizing discrete
machine. Headline metric, as always: exact length generalization (train width 3 or
mixed {1..5}; test to width 20). Combined with the session-1 mixed-width rescue.

Run:  python expD_discrete.py --op div1 --dims 4 --hidden 64 --steps 8000 --train_widths 1 2 3 4 5
"""
from __future__ import annotations
import argparse
import torch, torch.nn as nn

import core_data as cd
import expA_mealy as E
import expA_mul1 as M
import expA_div1 as D

DEVICE = E.DEVICE


# ----------------------------------------------------------------------------
# Discrete Mealy machine: state lives on {-1,+1}^d, enforced by a sign-STE.
# ----------------------------------------------------------------------------
def discretize(pre: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Forward morphs soft->hard as alpha goes 0->1; gradient ALWAYS flows through tanh.
    forward = (1-alpha)*tanh(pre) + alpha*sign(pre);  backward = d tanh(pre).
    alpha=1 => exactly sign(pre) (a hypercube vertex). alpha=0 => continuous tanh.
    Annealing alpha 0->1 lets the net fit in the easy continuous regime, then harden into a
    genuinely discrete (drift-free, exactly-extractable) machine."""
    soft = torch.tanh(pre)
    hard = (pre > 0).float() * 2.0 - 1.0
    return soft + alpha * (hard - soft).detach()


class DiscreteMealy(nn.Module):
    def __init__(self, base=10, state_dim=4, hidden=64):
        super().__init__()
        self.base = base
        self.state_dim = state_dim
        self.alpha = 1.0            # hardness; train() may anneal 0->1, eval forces 1.0
        in_dim = state_dim + 2 * base
        # f produces RAW pre-activations; discretize() replaces the usual final tanh.
        self.f = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh(),
                               nn.Linear(hidden, state_dim))
        self.g = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh(),
                               nn.Linear(hidden, base))
        self.s0 = nn.Parameter(torch.zeros(state_dim))   # discretized to the start vertex

    def _alpha(self):
        return self.alpha if self.training else 1.0   # eval is ALWAYS fully discrete

    def step(self, s_prev, x):
        z = torch.cat([s_prev, x], dim=-1)
        logits = self.g(z)
        s_next = discretize(self.f(z), self._alpha())
        return logits, s_next

    def start(self, n):
        return discretize(self.s0, self._alpha()).unsqueeze(0).expand(n, -1)

    def forward(self, a_oh, b_oh):
        N, L, _ = a_oh.shape
        s = self.start(N)
        outs = []
        for t in range(L):
            x = torch.cat([a_oh[:, t], b_oh[:, t]], dim=-1)
            logits, s = self.step(s, x)
            outs.append(logits)
        return torch.stack(outs, dim=1)


# ----------------------------------------------------------------------------
# Op dispatch: reuse the existing batch makers / predict / len-gen so eval infra
# is identical to session 1 — only the model class changes.
# ----------------------------------------------------------------------------
def make_batch(op, n, w, base, seed):
    if op in ("add", "sub"):
        return E.make_batch(n, w, base, op=op, seed=seed)
    if op == "mul1":
        return M.make_mul1_batch(n, w, base, seed=seed)
    if op == "div1":
        return D.make_div1_batch(n, w, base, seed=seed)
    raise ValueError(op)


def predict_fn(op, model, base):
    if op in ("add", "sub"):
        return E.net_predict_fn(model, base)
    if op == "mul1":
        return M.net_predict_mul1(model, base)
    if op == "div1":
        return D.net_predict_div1(model, base)
    raise ValueError(op)


def lengen(op, predict, base, widths):
    if op in ("add", "sub"):
        return cd.length_gen_report(predict, op, base=base, widths=widths, n_per_width=800)
    if op == "mul1":
        return M.mul1_lengen(predict, base=base, widths=widths, n=800)
    if op == "div1":
        return D.div1_lengen(predict, base=base, widths=widths, n=800)
    raise ValueError(op)


def train(model, op, base=10, width=3, steps=8000, bs=256, lr=1e-2, train_widths=None,
          cosine=True, anneal=False):
    """If anneal: ramp state hardness alpha 0->1 over the first 60% of steps, then hold at
    1.0 for the last 40% so the net adapts to a fully-discrete state before we extract."""
    import random
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps) if cosine else None
    lossf = nn.CrossEntropyLoss()
    wrng = random.Random(2024)
    for step in range(1, steps + 1):
        model.alpha = min(1.0, (step / (0.6 * steps))) if anneal else 1.0
        w = wrng.choice(train_widths) if train_widths else width
        a_oh, b_oh, tgt = make_batch(op, bs, w, base, seed=step)
        a_oh, b_oh, tgt = a_oh.to(DEVICE), b_oh.to(DEVICE), tgt.to(DEVICE)
        logits = model(a_oh, b_oh)
        loss = lossf(logits.reshape(-1, base), tgt.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if sched: sched.step()
        if step % 2000 == 0 or step == 1:
            print(f"  step {step:5d} loss {loss.item():.5f} (w={w}, alpha={model.alpha:.2f})")
    return model


# ----------------------------------------------------------------------------
# EXACT extraction: BFS the reachable hypercube vertices. FSM == net by construction.
# ----------------------------------------------------------------------------
@torch.no_grad()
def extract_discrete_fsm(model, base=10):
    model.eval()
    def vtuple(s):  # (d,) tensor -> tuple of 0/1 bits for readability
        return tuple(int(v > 0) for v in s.tolist())
    start = vtuple(discretize(model.s0))
    def to_vec(vt):
        return torch.tensor([1.0 if b else -1.0 for b in vt], device=DEVICE).unsqueeze(0)
    seen = {start}
    stack = [start]
    out_table, next_table = {}, {}
    while stack:
        st = stack.pop()
        c = to_vec(st)
        for da in range(base):
            for db in range(base):
                x = torch.zeros(1, 2 * base, device=DEVICE)
                x[0, da] = 1.0; x[0, base + db] = 1.0
                logits, s_next = model.step(c, x)
                ns = vtuple(s_next[0])
                out_table[(st, da, db)] = int(logits.argmax(-1).item())
                next_table[(st, da, db)] = ns
                if ns not in seen:
                    seen.add(ns); stack.append(ns)
    return out_table, next_table, start, sorted(seen)


class FSMModel:
    """A drop-in replacement for the net that runs purely on the extracted tables.
    Implements .eval(), .s0, .step(), forward() so the existing predict fns work on it."""
    def __init__(self, out_table, next_table, start, state_dim, base):
        self.out_table, self.next_table, self._start = out_table, next_table, start
        self.state_dim, self.base = state_dim, base
        self.s0 = torch.tensor([1.0 if b else -1.0 for b in start])
    def eval(self): return self
    def to(self, *a, **k): return self
    def __call__(self, a_oh, b_oh): return self.forward(a_oh, b_oh)
    def forward(self, a_oh, b_oh):
        N, L, B = a_oh.shape
        out = torch.zeros(N, L, B)
        for i in range(N):
            st = self._start
            for t in range(L):
                da = int(a_oh[i, t].argmax().item()); db = int(b_oh[i, t].argmax().item())
                o = self.out_table[(st, da, db)]; st = self.next_table[(st, da, db)]
                out[i, t, o] = 1.0
        return out


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    os.makedirs("runs", exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", default="div1", choices=["add", "sub", "mul1", "div1"])
    ap.add_argument("--dims", type=int, nargs="+", default=[4])
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--train_widths", type=int, nargs="+", default=None)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--anneal", action="store_true", help="ramp state hardness 0->1 (soft->hard)")
    args = ap.parse_args()
    tw = tuple(args.train_widths) if args.train_widths else None
    widths = (1, 2, 3, 4, 6, 8, 12, 16, 20)
    print(f"device={DEVICE}  op={args.op}  STE-discrete state")
    for d in args.dims:
        tag = f"train_widths={tw}" if tw else "train_width=3"
        print(f"\n=== DISCRETE op={args.op} d={d} hidden={args.hidden} ({tag}) seed={args.seed} ===")
        torch.manual_seed(args.seed)
        model = DiscreteMealy(base=10, state_dim=d, hidden=args.hidden)
        print(f"  params: {sum(p.numel() for p in model.parameters())}")
        train(model, args.op, base=10, width=3, steps=args.steps, lr=args.lr,
              train_widths=tw, anneal=args.anneal)

        net_rep = lengen(args.op, predict_fn(args.op, model, 10), 10, widths)
        print("  NET len-gen exact-acc: " + "  ".join(f"w{w}:{net_rep[w]:.3f}" for w in widths))

        out_t, nxt_t, start, states = extract_discrete_fsm(model, base=10)
        fsm = FSMModel(out_t, nxt_t, start, d, 10)
        fsm_rep = lengen(args.op, predict_fn(args.op, fsm, 10), 10, widths)
        print(f"  FSM ({len(states)} reachable states) len-gen: " +
              "  ".join(f"w{w}:{fsm_rep[w]:.3f}" for w in widths))
        # confirm FSM == net exactly on a sample (should be identical by construction)
        agree = all(predict_fn(args.op, model, 10)(*p) == predict_fn(args.op, fsm, 10)(*p)
                    for p in [(a, b) for a in (7, 83, 905, 31407)
                              for b in (3, 6, 9)])
        print(f"  FSM==NET on sample: {agree}   #states={len(states)}")
        if min(fsm_rep.values()) > 0.999:
            torch.save(model.state_dict(),
                       f"runs/expD_{args.op}_d{d}_h{args.hidden}.pt")
            print(f"  >>> {len(states)}-state DISCRETE FSM length-generalizes EXACTLY; saved.")
