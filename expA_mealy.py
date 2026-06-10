"""
expA_mealy.py — Exp A: tiny "neural Mealy machine" for digit-serial addition,
then EXTRACT the learned finite-state transducer and verify it exactly.

Why this shape (see TRACKER 2026-06-04):
A Mealy transducer has  out_t = g(state_{t-1}, input_t),  state_t = f(state_{t-1}, input_t).
Addition fits this exactly: carry_in = state_{t-1}; sum_digit = (a+b+carry_in)%B;
carry_out = state_t. We build a net in this form with a TINY continuous state, never
telling it about carry, train it, then discretize the state to read off an actual
finite-state transducer. The headline metric is exact accuracy on LONG numbers
(train width 3, test up to width 20): a real algorithm length-generalizes, a lookup
table does not.

Run:  python expA_mealy.py            # sweeps state dims, trains, extracts, verifies
"""
from __future__ import annotations
import argparse, sys
import torch, torch.nn as nn

import core_data as cd

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------------
# Model: neural Mealy machine
# ----------------------------------------------------------------------------
class NeuralMealy(nn.Module):
    def __init__(self, base=10, state_dim=2, hidden=16, state_noise=0.0):
        super().__init__()
        self.base = base
        self.state_dim = state_dim
        self.state_noise = state_noise   # DISCRETENESS BOTTLENECK: Gaussian noise added
        # to the state at train time. Forces the model to separate distinct states by
        # margins > noise => the learned state self-quantizes into a clean FSM that the
        # extractor can recover (interpretability-by-construction). Off at eval.
        in_dim = state_dim + 2 * base          # [state ; onehot(a) ; onehot(b)]
        # next-state and output share the input but have separate heads
        self.f = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh(),
                               nn.Linear(hidden, state_dim), nn.Tanh())  # bounded state
        self.g = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh(),
                               nn.Linear(hidden, base))                   # output digit logits
        self.s0 = nn.Parameter(torch.zeros(state_dim))

    def step(self, s_prev, x):
        """One Mealy step. s_prev:(N,state_dim) x:(N,2B) -> (logits:(N,B), s_next:(N,state_dim))"""
        if self.training and self.state_noise > 0:
            s_prev = s_prev + torch.randn_like(s_prev) * self.state_noise
        z = torch.cat([s_prev, x], dim=-1)
        logits = self.g(z)
        s_next = self.f(z)
        return logits, s_next

    def forward(self, a_oh, b_oh):
        """a_oh,b_oh: (N, L, B) one-hot digit sequences (LSB-first), already padded.
        Returns logits (N, L, B) — output digit at each position."""
        N, L, B = a_oh.shape
        s = self.s0.unsqueeze(0).expand(N, -1)
        outs = []
        for t in range(L):
            x = torch.cat([a_oh[:, t], b_oh[:, t]], dim=-1)
            logits, s = self.step(s, x)
            outs.append(logits)
        return torch.stack(outs, dim=1)


# ----------------------------------------------------------------------------
# Tensor helpers
# ----------------------------------------------------------------------------
def onehot_seq(nums, width, base):
    """list[int] -> (N, width, base) one-hot, LSB-first."""
    N = len(nums)
    out = torch.zeros(N, width, base)
    for i, n in enumerate(nums):
        for t, d in enumerate(cd.to_digits(n, width, base)):
            out[i, t, d] = 1.0
    return out


def make_batch(n, train_width, base, op="add", seed=None):
    """Batch for a digit-serial transduction op. Sequence length = train_width+1
    (flush step). For sub we enforce A>=B so the result is non-negative."""
    import random
    rng = random.Random(seed)
    A = [rng.randint(0, base ** train_width - 1) for _ in range(n)]
    Bn = [rng.randint(0, base ** train_width - 1) for _ in range(n)]
    if op == "sub":
        for i in range(n):
            if A[i] < Bn[i]:
                A[i], Bn[i] = Bn[i], A[i]
    L = train_width + 1
    a_oh = onehot_seq(A, L, base)          # extra position is digit 0 -> flush step
    b_oh = onehot_seq(Bn, L, base)
    tgt = torch.zeros(n, L, dtype=torch.long)
    for i in range(n):
        r = cd.exact_result(A[i], Bn[i], op)     # >=0 for add and (A>=B) sub
        tgt[i] = torch.tensor(cd.to_digits(r, L, base))
    return a_oh, b_oh, tgt


# ----------------------------------------------------------------------------
# Train
# ----------------------------------------------------------------------------
def train(model, base=10, train_width=3, op="add", steps=4000, bs=256, lr=1e-2,
          log_every=1000, train_widths=None):
    """If train_widths is given (e.g. (1,2,3,4,5)), each step samples a width from it,
    training the recurrence at MULTIPLE sequence lengths (anti length-overfit)."""
    import random
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    wrng = random.Random(777)
    for step in range(1, steps + 1):
        w = wrng.choice(train_widths) if train_widths else train_width
        a_oh, b_oh, tgt = make_batch(bs, w, base, op=op, seed=step)
        a_oh, b_oh, tgt = a_oh.to(DEVICE), b_oh.to(DEVICE), tgt.to(DEVICE)
        logits = model(a_oh, b_oh)
        loss = lossf(logits.reshape(-1, base), tgt.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % log_every == 0 or step == 1:
            print(f"  step {step:5d}  loss {loss.item():.4f}")
    return model


# ----------------------------------------------------------------------------
# Wrap a trained net as predict_fn(a,b)->int  (runs at whatever width is needed)
# ----------------------------------------------------------------------------
@torch.no_grad()
def net_predict_fn(model, base=10):
    model.eval()
    def predict(a, b):
        width = max(len(cd.to_digits(a, 1, base)) if a == 0 else _ndigits(a, base),
                    _ndigits(b, base))
        L = width + 1
        a_oh = onehot_seq([a], L, base).to(DEVICE)
        b_oh = onehot_seq([b], L, base).to(DEVICE)
        logits = model(a_oh, b_oh)[0]            # (L, base)
        digits = logits.argmax(-1).tolist()
        return cd.from_digits(digits, base)
    return predict


def _ndigits(n, base):
    if n == 0:
        return 1
    k = 0
    while n > 0:
        n //= base; k += 1
    return k


# ----------------------------------------------------------------------------
# Extraction: discretize state -> finite-state transducer, then verify exactly
# ----------------------------------------------------------------------------
@torch.no_grad()
def extract_fsm(model, base=10, probe_width=3, n_probe=4000, seed=123):
    """Collect (s_prev -> discrete) statistics, then build canonical transition &
    output tables by re-running the cell from each discrete state's centroid.
    Returns (tables, fsm_predict_fn, info)."""
    model.eval()
    import random
    rng = random.Random(seed)
    # Collect continuous states actually visited.
    A = [rng.randint(0, base ** probe_width - 1) for _ in range(n_probe)]
    Bn = [rng.randint(0, base ** probe_width - 1) for _ in range(n_probe)]
    L = probe_width + 1
    a_oh = onehot_seq(A, L, base).to(DEVICE)
    b_oh = onehot_seq(Bn, L, base).to(DEVICE)
    N = len(A)
    s = model.s0.unsqueeze(0).expand(N, -1).to(DEVICE)
    visited = {}   # discrete_state_bits -> list of continuous s vectors
    def bits(sv):  # sign discretization
        return tuple((sv > 0).int().tolist())
    # include the start state
    for i in range(N):
        visited.setdefault(bits(s[i]), []).append(s[i])
    for t in range(L):
        x = torch.cat([a_oh[:, t], b_oh[:, t]], dim=-1)
        _, s = model.step(s, x)
        for i in range(N):
            visited.setdefault(bits(s[i]), []).append(s[i])
    # centroid per discrete state
    centroids = {k: torch.stack(v).mean(0) for k, v in visited.items()}
    start_state = bits(model.s0.to(DEVICE))
    # Build canonical tables: from each discrete state's centroid, for each (a,b)
    # digit pair, compute output digit and next discrete state.
    states = sorted(centroids.keys())
    out_table = {}     # (state, a, b) -> out_digit
    next_table = {}    # (state, a, b) -> next_state
    for st in states:
        c = centroids[st].unsqueeze(0)
        for da in range(base):
            for db in range(base):
                x = torch.zeros(1, 2 * base, device=DEVICE)
                x[0, da] = 1.0; x[0, base + db] = 1.0
                logits, s_next = model.step(c, x)
                out_table[(st, da, db)] = int(logits.argmax(-1).item())
                ns = tuple((s_next[0] > 0).int().tolist())
                next_table[(st, da, db)] = ns
    # Some next_states might not be in `states` (unvisited corner). Map unknowns to
    # nearest known by hamming distance so the FSM is total.
    known = set(states)
    def closest(ns):
        if ns in known:
            return ns
        best, bd = None, 1e9
        for k in known:
            d = sum(p != q for p, q in zip(ns, k))
            if d < bd:
                bd, best = d, k
        return best
    next_table = {k: closest(v) for k, v in next_table.items()}

    def fsm_predict(a, b):
        width = max(_ndigits(a, base), _ndigits(b, base))
        L = width + 1
        ad = cd.to_digits(a, L, base); bd = cd.to_digits(b, L, base)
        st = start_state
        digits = []
        for t in range(L):
            key = (st, ad[t], bd[t])
            digits.append(out_table[key])
            st = next_table[key]
        return cd.from_digits(digits, base)

    info = {"n_states": len(states), "states": states, "start": start_state}
    return (out_table, next_table), fsm_predict, info


# ----------------------------------------------------------------------------
# Pretty-print the extracted FSM (only feasible for small base; show carry-like view)
# ----------------------------------------------------------------------------
def describe_fsm(tables, info, base=10, op="add"):
    """Human-readable characterization. For add: is out==(a+b+k)%base? For sub:
    out==(a-b+k)%base?  Also reports how many distinct next-states each state has
    (a clean carry/borrow machine has exactly 2 reachable states, each transitioning
    on the overflow/underflow event). The EXACT validation is fsm len-gen, not this."""
    out_table, next_table = tables
    states = info["states"]
    comb = (lambda da, db: da + db) if op == "add" else (lambda da, db: da - db)
    print(f"  Extracted FSM: {len(states)} states, start={info['start']}")
    for st in states:
        k = (out_table[(st, 0, 0)] - comb(0, 0)) % base
        consistent = all(out_table[(st, da, db)] == (comb(da, db) + k) % base
                         for da in range(base) for db in range(base))
        sign = "+" if op == "add" else "-"
        tag = f"out=(a{sign}b+{k})%{base}" if consistent else "out=NONLINEAR"
        # partition (a,b) by the resulting next-state
        from collections import Counter
        nxt_counts = Counter(next_table[(st, da, db)]
                             for da in range(base) for db in range(base))
        print(f"    state {st}: {tag}  next_states={dict(nxt_counts)}")


# ----------------------------------------------------------------------------
# Main: sweep state dims
# ----------------------------------------------------------------------------
def run(state_dims=(1, 2, 3), base=10, train_width=3, op="add", steps=4000, seed=0):
    torch.manual_seed(seed)
    widths = (1, 2, 3, 4, 6, 8, 12, 16, 20)
    results = {}
    for d in state_dims:
        print(f"\n=== op={op} state_dim={d} (base={base}, train_width={train_width}) ===")
        model = NeuralMealy(base=base, state_dim=d)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  params: {n_params}")
        train(model, base=base, train_width=train_width, op=op, steps=steps)
        net_rep = cd.length_gen_report(net_predict_fn(model, base), op,
                                       base=base, widths=widths, n_per_width=1000)
        print(f"  NET  len-gen exact-acc: " +
              "  ".join(f"w{w}:{net_rep[w]:.3f}" for w in widths))
        tables, fsm_predict, info = extract_fsm(model, base=base, probe_width=train_width)
        fsm_rep = cd.length_gen_report(fsm_predict, op,
                                       base=base, widths=widths, n_per_width=1000)
        print(f"  FSM  len-gen exact-acc: " +
              "  ".join(f"w{w}:{fsm_rep[w]:.3f}" for w in widths))
        describe_fsm(tables, info, base=base, op=op)
        results[d] = {"net": net_rep, "fsm": fsm_rep, "n_states": info["n_states"],
                      "params": n_params}
        # save checkpoint if the FSM generalizes
        if min(fsm_rep.values()) > 0.99:
            torch.save(model.state_dict(), f"runs/expA_mealy_{op}_d{d}.pt")
            print(f"  >>> FSM length-generalizes perfectly; saved runs/expA_mealy_{op}_d{d}.pt")
    return results


if __name__ == "__main__":
    import os
    os.makedirs("runs", exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--base", type=int, default=10)
    ap.add_argument("--train_width", type=int, default=3)
    ap.add_argument("--op", type=str, default="add", choices=["add", "sub"])
    ap.add_argument("--steps", type=int, default=4000)
    args = ap.parse_args()
    print(f"device={DEVICE}")
    run(state_dims=tuple(args.dims), base=args.base, op=args.op,
        train_width=args.train_width, steps=args.steps)
