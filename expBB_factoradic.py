"""
expBB_factoradic.py — ALIEN ARITHMETIC: can the minimal recurrent discoverer learn a POSITION-DEPENDENT carry?

Every operation the project discovered (carry, borrow, mult-carry, ...) has a position-INVARIANT per-digit rule — the
SAME function at every position — which is exactly why the persistent-register-threaded-across-positions architecture makes
length-generalization automatic. FACTORIAL BASE (factoradic) breaks that: at LSB position t the radix is r_t = t+2 (digit
range 0..t+1, place value (t+1)!), so the per-digit rule  out=(a+b+c) mod r_t,  carry=(a+b+c) div r_t  has a modulus that
DEPENDS ON POSITION and GROWS without bound. The carry stays binary (a+b+c <= 2r-1 => div in {0,1}); the hard part is the
variable-modulus OUTPUT — and length-gen to width W needs radix W+1, UNSEEN in short training. This pits two project
findings against each other: (carry is a strong attractor) vs (mod-by-a-variable-divisor is the division WALL; the net
learns mod only when the divisor divides the base).

CONDITIONS (all: tiny Mealy machine, mixed-width training {1..5}, exact decode, length-gen test):
  1 const10            — standard base-10 addition CONTROL (radix 10 everywhere). Must length-generalize (sanity).
  2 factoradic, no-radix — radix schedule r_t=t+2 but the model is NOT told the radix. EXPECT FAILURE (can't know modulus).
  3 factoradic, +radix  — model fed the scalar radix r_t at each step. Does position-dependence become learnable? This
                          REQUIRES extrapolating mod/div to radices never seen in training (width 12 -> radix 13).
  4 randmix,  +radix    — per-position radix drawn from a TRAIN range [2..7]; tested with radices from [2..7] (interp) and
                          [8..16] (UNSEEN extrap). Isolates "learn mod/div by a scalar base, extrapolate to unseen bases".

Honest prediction (flagged, not elaborated): #1 generalizes; #2 fails; #3 and #4-extrap are the real unknowns — if the net
can't learn variable-modulus mod/div (the division wall), factoradic length-gen is impossible here even WITH radix fed.
Run: python expBB_factoradic.py
"""
from __future__ import annotations
import argparse, math, random
import torch, torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------- mixed-radix codec (exact) ----------
def radices(width, kind, rng=None):
    if kind == "const10":     return [10] * width
    if kind == "factoradic":  return [t + 2 for t in range(width)]          # r_t = t+2
    if kind == "randmix":     return [rng.randint(2, 7) for _ in range(width)]
    if kind == "randmix_hi":  return [rng.randint(8, 16) for _ in range(width)]
    raise ValueError(kind)

def to_digits(n, rads):
    """LSB-first mixed-radix digits of n under radix schedule `rads` (len = #positions)."""
    out = []
    for r in rads:
        out.append(n % r); n //= r
    return out                                                              # n must be < prod(rads)

def from_digits(digs, rads):
    n, place = 0, 1
    for d, r in zip(digs, rads):
        n += d * place; place *= r
    return n

def prod(rads):
    p = 1
    for r in rads: p *= r
    return p


# ---------- model: Mealy machine, optionally fed the scalar radix ----------
class MixedRadixMealy(nn.Module):
    def __init__(self, maxd, state_dim=4, hidden=64, feed_radix=True):
        super().__init__()
        self.K = maxd + 1                      # digit alphabet 0..maxd
        self.feed_radix = feed_radix
        self.maxd = maxd
        in_dim = state_dim + 2 * self.K + (1 if feed_radix else 0)
        self.g = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, self.K))
        self.f = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh(), nn.Linear(hidden, state_dim), nn.Tanh())
        self.s0 = nn.Parameter(torch.zeros(state_dim))

    def forward(self, a_oh, b_oh, rad):
        N, L, K = a_oh.shape
        s = self.s0.unsqueeze(0).expand(N, -1)
        outs = []
        for t in range(L):
            parts = [s, a_oh[:, t], b_oh[:, t]]
            if self.feed_radix:
                parts.append(rad[:, t:t+1])
            z = torch.cat(parts, dim=-1)
            outs.append(self.g(z))
            s = self.f(z)
        return torch.stack(outs, dim=1)


def onehot(nums_digits, L, K):
    N = len(nums_digits)
    out = torch.zeros(N, L, K)
    for i, digs in enumerate(nums_digits):
        for t, d in enumerate(digs):
            out[i, t, d] = 1.0
    return out


def make_batch(n, width, kind, maxd, seed):
    rng = random.Random(seed)
    L = width + 1                              # flush position
    A_d, B_d, T_d, R = [], [], [], []
    for _ in range(n):
        rads = radices(width, kind, rng) + [maxd]      # per-example schedule + a high flush radix so carry-out fits
        Pw = prod(rads[:width])                        # product of the `width` real radices
        a = rng.randrange(Pw); b = rng.randrange(Pw)
        A_d.append(pad(to_digits(a, rads), L))
        B_d.append(pad(to_digits(b, rads), L))
        T_d.append(pad(to_digits(a + b, rads), L))     # a+b < 2*Pw <= Pw*maxd so it fits in L digits
        R.append(rads[:L])
    a_oh = onehot(A_d, L, maxd + 1); b_oh = onehot(B_d, L, maxd + 1)
    tgt = torch.tensor(T_d, dtype=torch.long)
    rad = torch.tensor(R, dtype=torch.float32) / maxd  # normalized scalar radix per position
    return a_oh, b_oh, rad, tgt

def pad(digs, L):
    return (digs + [0] * L)[:L]


def train(model, kind, maxd, steps=6000, bs=256, lr=3e-3, widths=(1, 2, 3, 4, 5), log_every=1500):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    wr = random.Random(99)
    for step in range(1, steps + 1):
        w = wr.choice(widths)
        a_oh, b_oh, rad, tgt = make_batch(bs, w, kind, maxd, seed=step)
        a_oh, b_oh, rad, tgt = a_oh.to(DEVICE), b_oh.to(DEVICE), rad.to(DEVICE), tgt.to(DEVICE)
        logits = model(a_oh, b_oh, rad)
        loss = lossf(logits.reshape(-1, maxd + 1), tgt.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % log_every == 0 or step == 1:
            print(f"    step {step:5d}  loss {loss.item():.4f}")
    return model


@torch.no_grad()
def lengen(model, kind, maxd, widths, n=600, seed=4242):
    model.eval()
    accs = {}
    for w in widths:
        a_oh, b_oh, rad, tgt = make_batch(n, w, kind, maxd, seed=seed + w)
        a_oh, b_oh, rad = a_oh.to(DEVICE), b_oh.to(DEVICE), rad.to(DEVICE)
        pred = model(a_oh, b_oh, rad).argmax(-1).cpu()
        # Exact whole-number match: target digits are the exact mixed-radix digits of a+b under each example's
        # schedule (length L), so digit-sequence equality is equivalent to whole-number equality.
        ok = int((pred == tgt).all(dim=1).sum().item())
        accs[w] = ok / n
    return accs


CONFIGS = [
    ("1 const10  (CONTROL, radix fed)", "const10",    True),
    ("2 factoradic NO radix fed",       "factoradic", False),
    ("3 factoradic +radix scalar",      "factoradic", True),
    ("4 randmix[2..7] +radix scalar",   "randmix",    True),
]
TEST_WIDTHS = (1, 2, 3, 5, 8, 12)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--maxd", type=int, default=18)
    ap.add_argument("--config", type=int, required=True, help="0..3 = train that config; 4 = randmix extrapolation (loads config-3 fresh-trained)")
    ap.add_argument("--out", type=str, default="runs/expBB_results.txt")
    args = ap.parse_args()
    print(f"device={DEVICE}  maxd={args.maxd}  config={args.config}", flush=True)
    torch.manual_seed(0)

    if args.config <= 3:
        name, kind, feed = CONFIGS[args.config]
        print(f"=== {name} ===", flush=True)
        m = MixedRadixMealy(args.maxd, state_dim=4, hidden=64, feed_radix=feed)
        print(f"    params {sum(p.numel() for p in m.parameters())}", flush=True)
        train(m, kind, args.maxd, steps=args.steps)
        acc = lengen(m, kind, args.maxd, TEST_WIDTHS)
        line = f"cfg{args.config} | {name} | " + "  ".join(f"w{w}:{acc[w]:.3f}" for w in TEST_WIDTHS)
        if feed and kind == "randmix":                       # config 4: also do the UNSEEN-radix extrapolation
            acc_hi = lengen(m, "randmix_hi", args.maxd, TEST_WIDTHS)
            line += "  ||EXTRAP(radix8-16) " + "  ".join(f"w{w}:{acc_hi[w]:.3f}" for w in TEST_WIDTHS)
            torch.save(m.state_dict(), "runs/expBB_randmix.pt")
        print("    " + line, flush=True)
        with open(args.out, "a") as f:
            f.write(line + "\n")
    print("DONE", flush=True)
