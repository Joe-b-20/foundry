"""
gpu_exp3_memory.py — Exp 3: does EXTERNAL DIFFERENTIABLE MEMORY break the
"representational wall" on a NON-finite-state operation?

The wall (see TRACKER): a flat finite-state recurrent cell can only represent
operations that are themselves finite-state (Mealy/Myhill-Nerode regular). Full
multi-digit MULTIPLICATION  a*b  is NOT finite-state: to emit the product LSB-first
you must accumulate shifted partial products, i.e. hold an unbounded carry/accumulator
that grows with operand length. So every memory-LESS recurrent net memorizes a lookup
table and COLLAPSES on long numbers. The documented fix is "a richer substrate
(loops, stack, recursion / external memory)".

This file tests that fix as a CONTRAST on the SAME op and SAME controller:

  BASELINE  : tiny GRU-style controller, small hidden state, NO external memory.
  MEMORY    : same controller + a small NTM-lite TAPE (a moving read/write head over
              an addressable buffer). The tape can serve as the running accumulator
              the regular substrate cannot hold.

The RESULT is the side-by-side length-generalization gap (train widths 1-4, test up
to ~16), not any single number. Exact whole-number match only — `2*3` must be `6`.

Number rep: digits LSB-first, base 10, via core_data codecs (reused). Product width
~ 2x operand width, so we run the transducer for 2*W (+slack) steps.

Run:
  python gpu_exp3_memory.py --smoke           # tiny, <90s on a 4060, every path
  python gpu_exp3_memory.py                    # scale defaults
  python gpu_exp3_memory.py --arch memory --steps 20000 --hidden 96 --seeds 3
"""
from __future__ import annotations
import argparse, os, time
import torch, torch.nn as nn, torch.nn.functional as F

import core_data as cd

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------------
# Data: TWO-PHASE digit-serial IO (LSB-first).
#
# This is the standard NTM/stack formulation where external memory can beat a
# memory-less cell. The sequence has an INPUT PHASE then an OUTPUT PHASE:
#   * input phase  (in_len steps): present the operand digits a_t,b_t one-hot;
#     the model's emissions here are DON'T-CARE (not scored).
#   * output phase (out_len steps): operand channels are BLANK; the model must
#     EMIT the answer digit at each step. Loss & exact-match are scored ONLY here.
# A phase flag bit tells the controller which phase it is in (0=input, 1=output).
#
# Why two phases: to length-generalize mul/reversal the model must first INGEST the
# whole operand, then PRODUCE the answer — it cannot do that if input and output are
# forced into lockstep. The phase split gives a memory-augmented controller a place
# to write during ingest and read back during produce. A memory-LESS cell has only
# its fixed hidden state to bridge the two phases => it cannot, beyond short widths.
#
# Exact whole-number match is preserved: we just don't penalize the don't-care
# input-phase emissions. `2*3` must still come out `6` on the output positions.
# ----------------------------------------------------------------------------
def io_lens(width: int, op: str):
    """(in_len, out_len). mul: ingest both operands (width steps, digit-aligned),
    produce 2*width product digits. rev: ingest width digits of a, produce width."""
    if op == "mul":
        return width, 2 * width
    if op == "rev":
        return width, width
    raise ValueError(op)


def make_batch(n, width, base=10, seed=None, op="mul"):
    """Returns (inp, tgt, mask):
      inp : (n, T, 2*base+1) — [onehot(a_t) ; onehot(b_t) ; phase_flag], T=in+out.
            During the output phase the two digit blocks are all-zero (blank).
      tgt : (n, T) long — answer digit at output positions, 0 elsewhere.
      mask: (n, T) float — 1.0 on output positions (scored), 0.0 on input positions.
    All LSB-first."""
    import random
    rng = random.Random(seed)
    in_len, out_len = io_lens(width, op)
    T = in_len + out_len
    inp = torch.zeros(n, T, 2 * base + 1)
    tgt = torch.zeros(n, T, dtype=torch.long)
    mask = torch.zeros(n, T)
    for i in range(n):
        a = rng.randint(0, base ** width - 1)
        b = rng.randint(0, base ** width - 1)
        if op == "mul":
            ad = cd.to_digits(a, in_len, base)
            bd = cd.to_digits(b, in_len, base)
            res = cd.to_digits(a * b, out_len, base)
        else:  # rev: reverse a's width digits; b unused. Non-regular look-back.
            ad = cd.to_digits(a, in_len, base)
            bd = [0] * in_len
            res = list(reversed(ad))
        # input phase: digits present, phase flag = 0
        for t in range(in_len):
            inp[i, t, ad[t]] = 1.0
            inp[i, t, base + bd[t]] = 1.0
        # output phase: digit blocks blank, phase flag = 1, target = answer digit
        for t in range(out_len):
            tt = in_len + t
            inp[i, tt, 2 * base] = 1.0          # phase flag
            tgt[i, tt] = res[t]
            mask[i, tt] = 1.0
    return inp, tgt, mask


# ----------------------------------------------------------------------------
# Controller: a tiny GRU-style recurrent cell shared by both architectures.
# ----------------------------------------------------------------------------
class GRUCell(nn.Module):
    def __init__(self, in_dim, hidden):
        super().__init__()
        self.cell = nn.GRUCell(in_dim, hidden)

    def forward(self, x, h):
        return self.cell(x, h)


# ----------------------------------------------------------------------------
# BASELINE: controller only, no external memory. Emits a digit each step.
# This is the memory-LESS flat substrate the wall predicts will fail on mul.
# ----------------------------------------------------------------------------
class Baseline(nn.Module):
    def __init__(self, base=10, hidden=64):
        super().__init__()
        self.base, self.hidden = base, hidden
        self.ctrl = GRUCell(2 * base + 1, hidden)        # +1 phase flag
        self.h0 = nn.Parameter(torch.zeros(hidden))
        self.out = nn.Linear(hidden, base)

    def forward(self, inp):
        N, T, _ = inp.shape
        h = self.h0.unsqueeze(0).expand(N, -1).contiguous()
        outs = []
        for t in range(T):
            h = self.ctrl(inp[:, t], h)
            outs.append(self.out(h))
        return torch.stack(outs, dim=1)


# ----------------------------------------------------------------------------
# MEMORY: same controller + an NTM-lite TAPE with one read/write head.
# The head emits a 3-way softmax shift {left, stay, right} (Gumbel/ST at train time
# for a more discrete, inspectable head), reads under the head, writes an
# erase/add update. The tape is the running accumulator a flat cell cannot hold.
# Kept deliberately small: M cells of width C.
# ----------------------------------------------------------------------------
class MemoryTape(nn.Module):
    def __init__(self, base=10, hidden=64, mem_slots=24, mem_width=8,
                 gumbel=True, tau=1.0):
        super().__init__()
        self.base, self.hidden = base, hidden
        self.M, self.C = mem_slots, mem_width
        self.gumbel, self.tau = gumbel, tau
        # controller sees input pair + phase flag + the vector it just read off tape
        self.ctrl = GRUCell(2 * base + 1 + mem_width, hidden)
        self.h0 = nn.Parameter(torch.zeros(hidden))
        # head: 3-way shift of the read/write address (left / stay / right)
        self.shift = nn.Linear(hidden, 3)
        # write: erase gate + add content (LSTM-write style)
        self.erase = nn.Linear(hidden, mem_width)
        self.add = nn.Linear(hidden, mem_width)
        self.out = nn.Linear(hidden + mem_width, base)
        # initial head position: a one-hot-ish weighting over the M slots (start at 0)
        w0 = torch.zeros(mem_slots); w0[0] = 1.0
        self.register_buffer("w0", w0)

    def _shift_addr(self, w, shift_logits):
        """Circularly convolve address weighting w:(N,M) with a 3-tap kernel over
        {-1,0,+1} given by softmax(shift_logits). Discrete-ish via Gumbel/ST."""
        if self.training and self.gumbel:
            k = F.gumbel_softmax(shift_logits, tau=self.tau, hard=True)  # (N,3)
        else:
            k = F.softmax(shift_logits, dim=-1)
        # roll left (+1 slot), stay, roll right (-1 slot)
        w_left = torch.roll(w, shifts=-1, dims=1)
        w_right = torch.roll(w, shifts=1, dims=1)
        w_new = (k[:, 0:1] * w_left + k[:, 1:2] * w + k[:, 2:3] * w_right)
        # renormalize (rolls preserve mass, but ST kernels can drift a hair)
        return w_new / (w_new.sum(dim=1, keepdim=True) + 1e-8)

    def forward(self, inp, return_trace=False):
        N, T, _ = inp.shape
        h = self.h0.unsqueeze(0).expand(N, -1).contiguous()
        mem = torch.zeros(N, self.M, self.C, device=inp.device)
        w = self.w0.unsqueeze(0).expand(N, -1).contiguous()
        outs, trace = [], []
        for t in range(T):
            r = (w.unsqueeze(-1) * mem).sum(dim=1)            # read: (N,C)
            x = torch.cat([inp[:, t], r], dim=-1)
            h = self.ctrl(x, h)
            # move head, then write at the new location
            w = self._shift_addr(w, self.shift(h))
            e = torch.sigmoid(self.erase(h))                  # (N,C)
            a = torch.tanh(self.add(h))                       # (N,C)
            wexp = w.unsqueeze(-1)                            # (N,M,1)
            mem = mem * (1 - wexp * e.unsqueeze(1)) + wexp * a.unsqueeze(1)
            outs.append(self.out(torch.cat([h, r], dim=-1)))
            if return_trace:
                trace.append(w.detach())
        logits = torch.stack(outs, dim=1)
        if return_trace:
            return logits, torch.stack(trace, dim=1)
        return logits


def build(arch, base, hidden, mem_slots, mem_width, gumbel):
    if arch == "baseline":
        return Baseline(base=base, hidden=hidden)
    if arch == "memory":
        return MemoryTape(base=base, hidden=hidden, mem_slots=mem_slots,
                          mem_width=mem_width, gumbel=gumbel)
    raise ValueError(arch)


# ----------------------------------------------------------------------------
# Train (shared harness). Samples a width per step from train widths (anti
# length-overfit), cross-entropy over all output positions.
# ----------------------------------------------------------------------------
def train(model, train_widths, base=10, op="mul", steps=4000, bs=256, lr=3e-3,
          log_every=500, anneal_tau=True):
    import random
    model.to(DEVICE).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss(reduction="none")          # masked to output phase
    wrng = random.Random(777)
    t0 = time.time()
    for step in range(1, steps + 1):
        if anneal_tau and hasattr(model, "tau"):
            # anneal Gumbel temperature 2.0 -> 0.5 for sharper discrete head late
            model.tau = max(0.5, 2.0 - 1.5 * step / steps)
        w = wrng.choice(train_widths)
        inp, tgt, mask = make_batch(bs, w, base, seed=step, op=op)
        inp, tgt, mask = inp.to(DEVICE), tgt.to(DEVICE), mask.to(DEVICE)
        logits = model(inp)
        per = lossf(logits.reshape(-1, base), tgt.reshape(-1)).reshape_as(mask)
        loss = (per * mask).sum() / mask.sum()             # mean over scored positions
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % log_every == 0 or step == 1:
            print(f"    step {step:6d}  loss {loss.item():.4f}  "
                  f"({time.time()-t0:.0f}s)")
    return model


# ----------------------------------------------------------------------------
# Exact length-gen eval. Wrap the net as predict_fn(a,b)->int at the needed width,
# reuse core_data's exact-match harness. Batched per width for speed.
# ----------------------------------------------------------------------------
@torch.no_grad()
def exact_acc_at_width(model, width, base=10, op="mul", n=512, seed=0):
    """Exact whole-number match: decode the OUTPUT-phase digits to an int and
    compare to exact ground truth. No partial credit."""
    model.eval()
    inp, tgt, mask = make_batch(n, width, base, seed=seed + width, op=op)
    inp = inp.to(DEVICE)
    pred = model(inp).argmax(-1).cpu()                       # (n,T)
    mask = mask.bool()
    correct = 0
    for i in range(n):
        out_digits = pred[i][mask[i]].tolist()               # output-phase digits
        tgt_digits = tgt[i][mask[i]].tolist()
        if cd.from_digits(out_digits, base) == cd.from_digits(tgt_digits, base):
            correct += 1
    return correct / n


@torch.no_grad()
def lengen_table(model, widths, base=10, op="mul", n=512, seed=0):
    return {w: exact_acc_at_width(model, w, base, op, n, seed) for w in widths}


# ----------------------------------------------------------------------------
# Run one arch across seeds; return per-width mean accuracy.
# ----------------------------------------------------------------------------
def run_arch(arch, args, op):
    accs_per_seed = []
    last_model = None
    for s in range(args.seeds):
        torch.manual_seed(s)
        model = build(arch, args.base, args.hidden, args.mem_slots,
                      args.mem_width, gumbel=not args.no_gumbel)
        nparams = sum(p.numel() for p in model.parameters())
        print(f"\n  [{arch}] seed {s}  params={nparams}")
        train(model, tuple(args.train_width), base=args.base, op=op,
              steps=args.steps, bs=args.bs, lr=args.lr)
        rep = lengen_table(model, args.test_widths, base=args.base, op=op,
                           n=args.eval_n, seed=1000 + s)
        print(f"  [{arch}] seed {s} len-gen: " +
              "  ".join(f"w{w}:{rep[w]:.3f}" for w in args.test_widths))
        accs_per_seed.append(rep)
        last_model = model
        torch.save(model.state_dict(),
                   os.path.join("runs", f"exp3_{op}_{arch}_seed{s}.pt"))
    # mean over seeds
    mean = {w: sum(r[w] for r in accs_per_seed) / len(accs_per_seed)
            for w in args.test_widths}
    return mean, last_model


def print_table(widths, results, op, train_widths):
    """results: {arch: {width: acc}}. Side-by-side."""
    archs = list(results.keys())
    print("\n" + "=" * 64)
    print(f"LENGTH-GEN exact-match accuracy  (op={op}, train_widths={tuple(train_widths)})")
    print("-" * 64)
    header = "  width |" + "".join(f" {a:>10} |" for a in archs)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for w in widths:
        intrain = "*" if w <= max(train_widths) else " "
        row = f"  {w:>4}{intrain} |" + "".join(
            f" {results[a][w]:>10.3f} |" for a in archs)
        print(row)
    print("  (* = within train width range; beyond = true length generalization)")
    print("=" * 64)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["baseline", "memory", "both"], default="both")
    ap.add_argument("--op", choices=["mul", "rev"], default="mul")
    ap.add_argument("--base", type=int, default=10)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--mem_slots", type=int, default=24)
    ap.add_argument("--mem_width", type=int, default=8)
    ap.add_argument("--no_gumbel", action="store_true",
                    help="use soft (non-discrete) head instead of straight-through Gumbel")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--train-width", type=int, nargs="+", default=[1, 2, 3, 4],
                    dest="train_width")
    ap.add_argument("--test-widths", type=int, nargs="+",
                    default=[2, 4, 6, 8, 12, 16], dest="test_widths")
    ap.add_argument("--eval-n", type=int, default=512, dest="eval_n")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        # tiny: must finish <90s on a 4060 and exercise every code path
        args.hidden = 32
        args.mem_slots = 16
        args.mem_width = 6
        args.steps = 400
        args.bs = 128
        args.seeds = 1
        args.train_width = [1, 2, 3]
        args.test_widths = [2, 3, 4, 6]
        args.eval_n = 128

    os.makedirs("runs", exist_ok=True)
    print(f"device={DEVICE}  arch={args.arch}  op={args.op}  smoke={args.smoke}")
    print(f"hidden={args.hidden} mem={args.mem_slots}x{args.mem_width} "
          f"steps={args.steps} seeds={args.seeds} "
          f"train_w={args.train_width} test_w={args.test_widths}")

    archs = ["baseline", "memory"] if args.arch == "both" else [args.arch]
    results, models = {}, {}
    for arch in archs:
        mean, model = run_arch(arch, args, args.op)
        results[arch] = mean
        models[arch] = model

    print_table(args.test_widths, results, args.op, args.train_width)

    # short results line to runs/exp3_results.txt
    line = (f"op={args.op} smoke={args.smoke} hidden={args.hidden} "
            f"steps={args.steps} seeds={args.seeds} train_w={args.train_width} | "
            + " || ".join(
                arch + " " + " ".join(f"w{w}:{results[arch][w]:.3f}"
                                      for w in args.test_widths)
                for arch in archs))
    with open(os.path.join("runs", "exp3_results.txt"), "a") as f:
        f.write(line + "\n")
    print("\nwrote runs/exp3_results.txt:\n  " + line)

    # one-line verdict on the contrast (only meaningful when both ran)
    if "baseline" in results and "memory" in results:
        beyond = [w for w in args.test_widths if w > max(args.train_width)]
        if beyond:
            b = sum(results["baseline"][w] for w in beyond) / len(beyond)
            m = sum(results["memory"][w] for w in beyond) / len(beyond)
            print(f"\nCONTRAST (mean exact-acc beyond train width {max(args.train_width)}): "
                  f"baseline={b:.3f}  memory={m:.3f}  gap={m-b:+.3f}")


# ----------------------------------------------------------------------------
# Sanity assertions (codec round-trip + tiny overfit) — run on import-as-main
# only when --selftest is passed so --smoke stays fast.
# ----------------------------------------------------------------------------
def selftest():
    print("== selftest ==")
    # 1) codec round-trip exact (LSB-first) + two-phase mul/rev IO is exact
    import random
    rng = random.Random(0)
    for _ in range(500):
        w = rng.randint(1, 4)
        a = rng.randint(0, 10 ** w - 1)
        b = rng.randint(0, 10 ** w - 1)
        _, out_len = io_lens(w, "mul")
        assert a * b <= 10 ** out_len - 1, (a, b, w)        # product fits in 2w digits
        assert cd.from_digits(cd.to_digits(a, w, 10), 10) == a
    # mul batch: output-phase target decodes to exactly a*b, and input phase carries a,b
    inp, tgt, mask = make_batch(64, 3, 10, seed=1, op="mul")
    in_len, out_len = io_lens(3, "mul")
    for i in range(64):
        a = cd.from_digits(inp[i, :in_len, :10].argmax(-1).tolist(), 10)
        b = cd.from_digits(inp[i, :in_len, 10:20].argmax(-1).tolist(), 10)
        out_digits = tgt[i][mask[i].bool()].tolist()
        assert cd.from_digits(out_digits, 10) == a * b, (a, b, out_digits)
        assert mask[i].sum().item() == out_len                # exactly out_len scored
        assert inp[i, :in_len, 20].sum().item() == 0          # phase flag off in input
        assert inp[i, in_len:, 20].sum().item() == out_len    # phase flag on in output
    # rev batch: output digits are a's digits reversed
    inp, tgt, mask = make_batch(32, 3, 10, seed=2, op="rev")
    for i in range(32):
        ad = inp[i, :3, :10].argmax(-1).tolist()
        out_digits = tgt[i][mask[i].bool()].tolist()
        assert out_digits == list(reversed(ad)), (ad, out_digits)
    print("  codec + two-phase mul/rev IO round-trip: PASS")

    # 2) tiny overfit sanity: memory model drives MASKED loss DOWN on a fixed tiny
    # batch in a few hundred steps (exercises shift/read/write/out + masking).
    torch.manual_seed(0)
    model = MemoryTape(base=10, hidden=32, mem_slots=12, mem_width=6).to(DEVICE).train()
    inp, tgt, mask = make_batch(64, 2, 10, seed=3, op="mul")
    inp, tgt, mask = inp.to(DEVICE), tgt.to(DEVICE), mask.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    lossf = nn.CrossEntropyLoss(reduction="none")
    first = None
    for step in range(300):
        logits = model(inp)
        per = lossf(logits.reshape(-1, 10), tgt.reshape(-1)).reshape_as(mask)
        loss = (per * mask).sum() / mask.sum()
        opt.zero_grad(); loss.backward(); opt.step()
        if first is None:
            first = loss.item()
    assert loss.item() < first - 0.2, f"overfit failed: {first:.3f}->{loss.item():.3f}"
    # trace path executes and aligns with sequence length
    logits, tr = model(inp[:4], return_trace=True)
    assert tr.shape[0] == 4 and tr.shape[1] == inp.shape[1]
    print(f"  memory overfit {first:.3f}->{loss.item():.3f} + trace path: PASS")
    print("== selftest PASS ==")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
