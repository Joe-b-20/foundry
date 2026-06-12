"""
expI_repr.py — SESSION 3: let the model CHOOSE its own number representation.

The whole project feeds numbers as fixed base-10 one-hot digits, LSB-first, and every
algorithm discovered is an algorithm OVER that imposed code. Carry is a strong attractor
because base-10 digits are imposed. Here the intermediate representation is a LEARNED
degree of freedom, and we ask: given a WIDER digit alphabet (room for redundancy), does a
tiny model discover CARRY-FREE addition (carry-save) instead of the sequential carry FSM?

Architecture (all position-wise / systematic, so it length-generalizes by construction):
    operand digits a_i,b_i  (base B, LSB-first, one-hot)
      --[COMBINE]-->  symbol s_i in {0..K-1}     position-wise, NO recurrence  (carry-FREE op)
      --[DECODER (tiny Mealy)]--> output digit o_i in {0..B-1}                  (normalization)
The OP has no state => it cannot carry. Any carrying must be DEFERRED to the recurrent
DECODER. K is the redundancy knob: K=B is no redundancy; K up to 2B-1 can hold a_i+b_i.

Two-stage protocol (joint discrete-code training is hard; this separates the questions):
  STAGE 1 (soft): train with a soft continuous code (anneal tau 1.5->0.5). Measures whether
    the carry-FREE decomposition exists and LENGTH-GENERALIZES at all. Eval = soft.
  STAGE 2 (discreteness diagnostic): freeze COMBINE's argmax -> a fixed discrete code table;
    reinit & retrain ONLY the decoder on that hard code. If it length-generalizes, the chosen
    code is a usable DISCRETE redundant code (read it off); if not, the model needed the
    continuous values. Eval = hard.

Headline metric: exact length-gen (train widths {1..5}, test to w20). Eval is exact.

Run: python expI_repr.py --K 19 --win 0 --dec_dim 2
     python expI_repr.py --sweep
"""
from __future__ import annotations
import argparse, os
import torch, torch.nn as nn, torch.nn.functional as F
import core_data as cd
from expA_mealy import onehot_seq, _ndigits, make_batch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WIDTHS = (1, 2, 3, 4, 6, 8, 12, 16, 20)


class ReprAdder(nn.Module):
    """Learned redundant symbol code + carry-free combine + recurrent normalizing decoder."""
    def __init__(self, base=10, K=19, win=0, dec_dim=2, hidden=64):
        super().__init__()
        self.base, self.K, self.win, self.dec_dim = base, K, win, dec_dim
        comb_in = (2 * win + 1) * 2 * base
        self.combine = nn.Sequential(nn.Linear(comb_in, hidden), nn.Tanh(),
                                     nn.Linear(hidden, K))
        dec_in = max(dec_dim, 0) + K
        self.f = (nn.Sequential(nn.Linear(dec_in, hidden), nn.Tanh(),
                                nn.Linear(hidden, dec_dim), nn.Tanh())
                  if dec_dim > 0 else None)
        self.g = nn.Sequential(nn.Linear(dec_in, hidden), nn.Tanh(),
                               nn.Linear(hidden, base))
        self.s0 = nn.Parameter(torch.zeros(max(dec_dim, 0)))
        self.tau = 1.0

    def combine_logits(self, a_oh, b_oh):
        """(N,L,B),(N,L,B) -> (N,L,K) with a symmetric ±win position window."""
        w = self.win
        if w == 0:
            x = torch.cat([a_oh, b_oh], dim=-1)
        else:
            parts = []
            for off in range(-w, w + 1):
                a_s = torch.roll(a_oh, shifts=off, dims=1)
                b_s = torch.roll(b_oh, shifts=off, dims=1)
                if off > 0:
                    a_s[:, :off] = 0; b_s[:, :off] = 0
                elif off < 0:
                    a_s[:, off:] = 0; b_s[:, off:] = 0
                parts += [a_s, b_s]
            x = torch.cat(parts, dim=-1)
        return self.combine(x)

    def forward(self, a_oh, b_oh, mode="soft"):
        """mode in {soft (training), hard (eval / frozen-code)}."""
        N, L, B = a_oh.shape
        clog = self.combine_logits(a_oh, b_oh)              # (N,L,K)
        idx = clog.argmax(-1)
        sym = F.one_hot(idx, self.K).float() if mode == "hard" else torch.softmax(clog / self.tau, -1)
        outs = []
        if self.dec_dim > 0:
            s = self.s0.unsqueeze(0).expand(N, -1)
            for t in range(L):
                z = torch.cat([s, sym[:, t]], dim=-1)
                outs.append(self.g(z))
                s = self.f(z)
        else:
            for t in range(L):
                outs.append(self.g(sym[:, t]))
        return torch.stack(outs, dim=1), clog, idx


def _train_loop(model, params, steps, bs, lr, widths, base, mode, ent_coef, anneal_frac, seed0, tag, op="add"):
    import random
    opt = torch.optim.Adam(params, lr=lr)
    lossf = nn.CrossEntropyLoss()
    model.train()
    wr = random.Random(777)
    anneal = max(1, int(anneal_frac * steps))
    for step in range(1, steps + 1):
        model.tau = max(0.5, 1.5 - 1.0 * (step / anneal)) if mode == "soft" else 0.5
        w = wr.choice(widths)
        a_oh, b_oh, tgt = make_batch(bs, w, base, op=op, seed=seed0 + step)
        a_oh, b_oh, tgt = a_oh.to(DEVICE), b_oh.to(DEVICE), tgt.to(DEVICE)
        logits, clog, _ = model(a_oh, b_oh, mode=mode)
        loss = lossf(logits.reshape(-1, base), tgt.reshape(-1))
        if ent_coef > 0 and mode == "soft":
            p = torch.softmax(clog / model.tau, -1)
            loss = loss + ent_coef * (-(p * (p + 1e-9).log()).sum(-1).mean())   # peak the code
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 2000 == 0 or step == 1:
            print(f"    [{tag}] step {step:5d}  loss {loss.item():.4f}  tau {model.tau:.2f}")


def train_soft(model, steps=8000, bs=256, lr=3e-3, widths=(1, 2, 3, 4, 5), base=10, ent_coef=0.02, op="add"):
    model.to(DEVICE)
    _train_loop(model, model.parameters(), steps, bs, lr, widths, base,
                mode="soft", ent_coef=ent_coef, anneal_frac=0.6, seed0=0, tag="soft", op=op)


def reset_decoder(model):
    for mod in (model.f, model.g):
        if mod is None:
            continue
        for layer in mod:
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
    with torch.no_grad():
        model.s0.zero_()


def retrain_decoder_frozen(model, steps=4000, bs=256, lr=3e-3, widths=(1, 2, 3, 4, 5), base=10, op="add"):
    """Freeze COMBINE (its argmax = the fixed discrete code); reinit & retrain ONLY the decoder
    on that hard code. Tests whether the chosen discrete code is usable for exact addition."""
    for p in model.combine.parameters():
        p.requires_grad_(False)
    reset_decoder(model)
    dec_params = [p for n, p in model.named_parameters() if not n.startswith("combine")]
    _train_loop(model, dec_params, steps, bs, lr, widths, base,
                mode="hard", ent_coef=0.0, anneal_frac=0.6, seed0=100000, tag="dec", op=op)


@torch.no_grad()
def predict_fn(model, base, mode="hard"):
    model.eval()
    def predict(a, b):
        L = max(_ndigits(a, base), _ndigits(b, base)) + 1
        a_oh = onehot_seq([a], L, base).to(DEVICE)
        b_oh = onehot_seq([b], L, base).to(DEVICE)
        logits, _, _ = model(a_oh, b_oh, mode=mode)
        return cd.from_digits(logits[0].argmax(-1).tolist(), base)
    return predict


def lengen(model, base, mode, op="add", n=1000):
    return cd.length_gen_report(predict_fn(model, base, mode=mode), op,
                                base=base, widths=WIDTHS, n_per_width=n)


def show(rep, label):
    print(f"  {label}: " + "  ".join(f"w{w}:{rep[w]:.3f}" for w in WIDTHS))


@torch.no_grad()
def describe_combine(model, base, op="add"):
    """For win=0: read off the (a,b)->symbol table. Is the symbol a function of the column
    combination (a+b for add, a-b for sub)? Injective in it? (= the carry/borrow-save code.)"""
    model.eval()
    if model.win != 0:
        print("    (combine table printed only for win=0)")
        return
    comb = (lambda da, db: da + db) if op == "add" else (lambda da, db: da - db)
    name = "a+b" if op == "add" else "a-b"
    sym = {}
    for da in range(base):
        for db in range(base):
            a_oh = torch.zeros(1, 1, base, device=DEVICE); a_oh[0, 0, da] = 1
            b_oh = torch.zeros(1, 1, base, device=DEVICE); b_oh[0, 0, db] = 1
            sym[(da, db)] = int(model.combine_logits(a_oh, b_oh).argmax(-1).item())
    by_c = {}
    for (da, db), s in sym.items():
        by_c.setdefault(comb(da, db), set()).add(s)
    is_fn = all(len(v) == 1 for v in by_c.values())
    n_used = len({s for v in by_c.values() for s in v})
    injective = is_fn and len({next(iter(by_c[k])) for k in by_c}) == len(by_c)
    print(f"    COMBINE code: symbol = f({name})? {is_fn}; injective in {name}? {injective}; "
          f"#symbols used={n_used} of K={model.K}")
    if is_fn:
        print(f"    {name}->symbol: {{{', '.join(f'{k}:{next(iter(by_c[k]))}' for k in sorted(by_c))}}}")


@torch.no_grad()
def count_decoder_states(model, base, probe_width=4, n=400):
    if model.dec_dim == 0:
        print("    decoder has no recurrent state (dec_dim=0)")
        return
    import random
    model.eval()
    rng = random.Random(1)
    A = [rng.randint(0, base ** probe_width - 1) for _ in range(n)]
    Bn = [rng.randint(0, base ** probe_width - 1) for _ in range(n)]
    L = probe_width + 1
    a_oh = onehot_seq(A, L, base).to(DEVICE); b_oh = onehot_seq(Bn, L, base).to(DEVICE)
    idx = model.combine_logits(a_oh, b_oh).argmax(-1)
    sym = F.one_hot(idx, model.K).float()
    s = model.s0.unsqueeze(0).expand(len(A), -1)
    seen = set()
    for t in range(L):
        s = model.f(torch.cat([s, sym[:, t]], dim=-1))
        for row in s:
            seen.add(tuple((row > 0).int().tolist()))
    print(f"    decoder visited {len(seen)} distinct sign-states (clean carry = 2)")


def run_one(base=10, K=19, win=0, dec_dim=2, steps=8000, seed=0, op="add"):
    torch.manual_seed(seed)
    model = ReprAdder(base=base, K=K, win=win, dec_dim=dec_dim)
    tag = f"{op}_K{K}_win{win}_dec{dec_dim}"
    print(f"\n=== {tag} (base={base}, params={sum(p.numel() for p in model.parameters())}) ===")
    train_soft(model, steps=steps, base=base, op=op)
    rep_soft = lengen(model, base, "soft", op=op)
    show(rep_soft, f"[{tag}] STAGE1 SOFT (carry-free?) len-gen")
    describe_combine(model, base, op=op)
    count_decoder_states(model, base)
    print("  -- STAGE 2: freeze argmax code, reinit+retrain decoder (is the code DISCRETE-usable?) --")
    retrain_decoder_frozen(model, steps=max(4000, steps // 2), base=base, op=op)
    rep_hard = lengen(model, base, "hard", op=op)
    show(rep_hard, f"[{tag}] STAGE2 HARD (frozen discrete code) len-gen")
    if min(rep_hard.values()) > 0.99 or min(rep_soft.values()) > 0.99:
        os.makedirs("runs", exist_ok=True)
        torch.save(model.state_dict(), f"runs/expI_{tag}.pt")
        print(f"  >>> length-generalizes; saved runs/expI_{tag}.pt")
    return tag, rep_soft, rep_hard


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=int, default=10)
    ap.add_argument("--K", type=int, default=19)
    ap.add_argument("--win", type=int, default=0)
    ap.add_argument("--dec_dim", type=int, default=2)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--op", type=str, default="add", choices=["add", "sub"])
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()
    print(f"device={DEVICE}")
    if args.sweep:
        # K=19 win0 dec2 already run separately (soft 1.000 to w20). Here the informative rest:
        configs = [
            dict(K=10, win=0, dec_dim=2),   # no ALPHABET redundancy: does soft still work?
                                            #   (= is the redundancy in the alphabet or the continuous code?)
            dict(K=19, win=0, dec_dim=0),   # no recurrence ANYWHERE -> predict FAIL (carry can't be eliminated)
            dict(K=11, win=1, dec_dim=2),   # local ±1 lookahead, low K (signed-digit-ish)
        ]
        summ = []
        for c in configs:
            tag, rs, rh = run_one(base=args.base, steps=args.steps, seed=args.seed, **c)
            summ.append((tag, min(rs.values()), rs[20], min(rh.values()), rh[20]))
        print("\n=== SWEEP SUMMARY (addition, base 10) ===")
        print("  config            soft_min soft_w20 | hard_min hard_w20")
        for tag, sm, sw, hm, hw in summ:
            print(f"  {tag:16s}  {sm:.3f}   {sw:.3f}    | {hm:.3f}   {hw:.3f}")
    else:
        run_one(base=args.base, K=args.K, win=args.win, dec_dim=args.dec_dim,
                steps=args.steps, seed=args.seed, op=args.op)
