"""
expM_add.py — DISCOVER THE PRIMITIVE: addition = counting (digit-successor), from outcome alone.

Session 5 / rung 2 (foundational). Every prior addition result discovered the CARRY but was GIVEN
single-digit addition (the neural Mealy learned the digit-sum table from one-hot digits; the GP / VM had
a `+` primitive). Here the ONLY arithmetic primitive is TICK: a base-b digit WHEEL successor
(OUT += 1; if OUT == base: OUT = 0 and flag a rollover). There is NO add. The model must DISCOVER that
multi-digit ADDITION is: per column (LSB-first), tick the A-digit wheel forward b_i times, then once more
if a carry came in; the WHEEL ROLLOVER is the carry out; emit the wheel; thread the carry to the next
column; and flush a final carry column. I.e. a + b = succ^b(a), generalized to digits with carry = rollover.

FLOOR (given, honest): the digit-wheel successor TICK (OUT++ with rollover, the irreducible "+1"),
per-digit affordances (loopflag = K<b_i counts the B-digit ticks; cin = carry-in flag; done), digit
addressing (LOADA reads a_i), EMIT (write a digit + thread carry + advance), HALT. The answer is assembled
as a DIGIT STRING (NOT via any whole-number add — that would be circular for an addition experiment).
NOT given: addition, or that rollover means carry, or the column loop. DISCOVERED-from-outcome: all of that.

Recipe = expJ/expM exact-filtered self-imitation, generalized to TWO flag-gated inner loops (TICK gated by
loopflag, CARRYTICK gated by cin). NO traces.

Run: python expM_add.py --iters 250 [--smoke]
"""
from __future__ import annotations
import argparse, random
from collections import deque
import torch, torch.nn as nn

import expA_mealy as E

DEVICE = E.DEVICE

INSTRS = ["HALT", "LOADA", "TICK", "CARRYTICK", "EMIT"]
NI = len(INSTRS)
IID = {n: i for i, n in enumerate(INSTRS)}
N_OBS = 4                         # [loopflag(K<b_i), cin(CIN==1), done, 0]
LOOP_GATES = {IID["TICK"]: 0, IID["CARRYTICK"]: 1}   # op -> obs index that gates its repetition
DELIM = IID["EMIT"]              # per-column delimiter (advance), like INC_J in expM_muldigit
DONE_IDX = 2


# ----------------------------------------------------------------------------
# Addition VM with a MINIMAL ALU: the only arithmetic is the digit-wheel successor TICK. No add.
# ----------------------------------------------------------------------------
class VM:
    def __init__(self, op, A, B, D, base):
        self.A, self.B, self.base = A, B, base
        self.i = 0; self.OUT = 0; self.CIN = 0; self.NEXTC = 0; self.K = 0
        self.out_list = []
        self.N = max(E._ndigits(A, base), E._ndigits(B, base))
        self.halted = False

    def _bdig(self):
        return (self.B // self.base ** self.i) % self.base

    def obs(self):
        loopflag = 1.0 if self.K < self._bdig() else 0.0       # more B-ticks owed this column
        cin = 1.0 if self.CIN == 1 else 0.0
        done = 1.0 if (self.i >= self.N and self.CIN == 0) else 0.0
        return [loopflag, cin, done, 0.0]

    def execute(self, instr):
        b = self.base
        if instr == "LOADA":
            self.OUT = (self.A // b ** self.i) % b; self.K = 0       # set the wheel to A's digit i
        elif instr == "TICK":
            self.OUT += 1                                            # wheel successor (the only +1)
            if self.OUT == b:
                self.OUT = 0; self.NEXTC = 1                         # rollover -> carry out
            self.K += 1
        elif instr == "CARRYTICK":
            self.OUT += 1                                            # consume an incoming carry by ticking
            if self.OUT == b:
                self.OUT = 0; self.NEXTC = 1
            self.CIN = 0
        elif instr == "EMIT":
            self.out_list.append(self.OUT)                          # write the column's digit (LSB-first)
            self.CIN = self.NEXTC; self.NEXTC = 0; self.i += 1      # thread carry, advance
        elif instr == "HALT":
            self.halted = True
        else:
            raise ValueError(instr)

    def answer(self):
        v = 0
        for k, d in enumerate(self.out_list):
            v += d * self.base ** k
        return v


def run_reference(vm, max_steps=200000):
    trace = []
    def step(name):
        trace.append((vm.obs(), IID[name])); vm.execute(name)
    while not vm.halted:
        step("LOADA")
        while vm.K < vm._bdig():
            step("TICK")
        if vm.CIN == 1:
            step("CARRYTICK")
        step("EMIT")
        if vm.i >= vm.N and vm.CIN == 0:
            step("HALT")
        if len(trace) > max_steps:
            break
    return trace


def _digits_nocarry(w, base, rng):
    """Genuine-width A,B whose columns never carry (a_i+b_i<base) -> the core loop ignites without the
    carry machinery (warmup only)."""
    a = [0] * w; b = [0] * w
    for i in range(w):
        if i == w - 1:                          # leading column: both digits nonzero, sum < base
            a[i] = rng.randint(1, base - 2)
            b[i] = rng.randint(1, base - 1 - a[i])
        else:
            a[i] = rng.randint(0, base - 1)
            b[i] = rng.randint(0, base - 1 - a[i])
    A = sum(d * base ** i for i, d in enumerate(a))
    B = sum(d * base ** i for i, d in enumerate(b))
    return A, B, 0


def make_problem(op, width, base, rng):
    """EVAL: full range incl. zeros / carries / leading zeros."""
    return rng.randint(0, base ** width - 1), rng.randint(0, base ** width - 1), 0


def make_problem_g(op, w, base, rng, nocarry=False):
    if nocarry:
        return _digits_nocarry(w, base, rng)
    lo = 1 if w == 1 else base ** (w - 1)
    hi = base ** w - 1
    return rng.randint(lo, hi), rng.randint(lo, hi), 0


# ----------------------------------------------------------------------------
# Controller (NI=5 here).
# ----------------------------------------------------------------------------
class Controller(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.gru = nn.GRU(N_OBS, hidden, batch_first=True)
        self.head = nn.Linear(hidden, NI)

    def forward(self, obs, h=None):
        y, h = self.gru(obs, h)
        return self.head(y), h


@torch.no_grad()
def controller_run(model, op, A, B, D, base, step_cap=None):
    model.eval()
    vm = VM(op, A, B, D, base)
    if step_cap is None:
        step_cap = (base + 3) * (vm.N + 2) + 20
    h = None; steps = 0
    while not vm.halted and steps < step_cap:
        obs = torch.tensor([vm.obs()], dtype=torch.float32, device=DEVICE).unsqueeze(0)
        logits, h = model(obs, h)
        instr = INSTRS[int(logits[0, -1].argmax())]
        vm.execute(instr); steps += 1
    return vm.answer(), steps, vm.halted


def step_cap_for(w, base):
    return (base + 3) * (w + 2) + 12


def len_cap_for(w, base):
    return (base + 4) * (w + 2) + 16


# ----------------------------------------------------------------------------
# Sampling.
# ----------------------------------------------------------------------------
def sample_rollouts(model, problems, base, step_cap, temp=1.0, eps=0.0):
    M = len(problems)
    vms = [VM("add", A, B, D, base) for (op, A, B, D) in problems]
    legal = torch.ones(M, NI, device=DEVICE) / NI
    h = None
    tr_obs = [[] for _ in range(M)]; tr_act = [[] for _ in range(M)]
    active = [True] * M
    for _ in range(step_cap):
        if not any(active):
            break
        obs_rows = [vm.obs() for vm in vms]
        obs = torch.tensor(obs_rows, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        logits, h = model(obs, h)
        probs = torch.softmax(logits[:, -1, :] / temp, dim=-1)
        if eps > 0:
            probs = (1 - eps) * probs + eps * legal
        a = torch.multinomial(probs, 1).squeeze(1).tolist()
        for i in range(M):
            if not active[i]:
                continue
            tr_obs[i].append(obs_rows[i]); tr_act[i].append(a[i])
            vms[i].execute(INSTRS[a[i]])
            if vms[i].halted:
                active[i] = False
    out = []
    for i in range(M):
        op, A, B, D = problems[i]
        ok = vms[i].halted and vms[i].answer() == A + B
        out.append((ok, "add", problems[i][1:], tr_obs[i], tr_act[i]))
    return out


_lossf = nn.CrossEntropyLoss(reduction="none")


def imitate(model, opt, traces):
    M = len(traces)
    L = max(len(a) for _, a in traces)
    obs_b = torch.zeros(M, L, N_OBS); tgt = torch.zeros(M, L, dtype=torch.long); mask = torch.zeros(M, L)
    for i, (o, a) in enumerate(traces):
        k = len(a)
        obs_b[i, :k] = torch.tensor(o, dtype=torch.float32)
        tgt[i, :k] = torch.tensor(a, dtype=torch.long); mask[i, :k] = 1.0
    obs_b, tgt, mask = obs_b.to(DEVICE), tgt.to(DEVICE), mask.to(DEVICE)
    logits, _ = model(obs_b)
    loss = (_lossf(logits.reshape(-1, NI), tgt.reshape(-1)) * mask.reshape(-1)).sum() / mask.sum()
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return loss.item()


# ----------------------------------------------------------------------------
# Loop extraction generalized to per-op flag gates.  body token = (instr_id, is_loop, gate_obs_index)
# ----------------------------------------------------------------------------
def cycle_to_body(cycle):
    body = []
    i = 0
    while i < len(cycle):
        j = i
        while j < len(cycle) and cycle[j] == cycle[i]:
            j += 1
        op = cycle[i]
        is_loop = (op in LOOP_GATES) or ((j - i) >= 2)
        gate = LOOP_GATES.get(op, 0)
        body.append((op, is_loop, gate))
        i = j
    return tuple(body)


def segment_bodies(act):
    halt = IID["HALT"]
    segs, cur = [], []
    for x in act:
        cur.append(x)
        if x == DELIM:
            segs.append(tuple(cur)); cur = []
    if cur != [halt] or not segs:
        return []
    return list(dict.fromkeys(cycle_to_body(s) for s in segs))


def interpret(op, body, A, B, D, base, cap=20000):
    vm = VM(op, A, B, D, base)
    obs, act = [], []
    steps = 0
    while not vm.halted and steps < cap:
        for (iid, is_loop, gate) in body:
            if is_loop:
                while vm.obs()[gate] >= 0.5 and not vm.halted and steps < cap:
                    obs.append(vm.obs()); act.append(iid); vm.execute(INSTRS[iid]); steps += 1
            else:
                obs.append(vm.obs()); act.append(iid); vm.execute(INSTRS[iid]); steps += 1
            if vm.halted:
                break
        if not vm.halted and vm.obs()[DONE_IDX] >= 0.5:
            obs.append(vm.obs()); act.append(IID["HALT"]); vm.execute("HALT"); steps += 1
    return obs, act, vm.answer(), vm.halted


def verify_body(op, body, base, rng, widths=(1, 2, 3, 5), n=24):
    for w in widths:
        for _ in range(n):
            A, B, D = make_problem(op, w, base, rng)        # FULL range (incl. carries) -- a real adder
            _, _, got, halted = interpret(op, body, A, B, D, base, cap=(base + 4) * (w + 2) + 40)
            if not (halted and got == A + B):
                return False
    return True


def minimize_body(op, body, base, rng):
    body = list(body)
    changed = True
    while changed and len(body) > 1:
        changed = False
        for i in range(len(body)):
            cand = tuple(body[:i] + body[i + 1:])
            if cand and cand[-1][0] == DELIM and verify_body(op, cand, base, rng, n=40):
                body = list(cand); changed = True; break
    return tuple(body)


def extract_loop(op, act, base, rng):
    for body in segment_bodies(act):
        if body and body[-1][0] == DELIM and verify_body(op, body, base, rng):
            return minimize_body(op, body, base, rng)
    return None


def clean_loop_trace(op, body, w, base, rng):
    A, B, D = make_problem(op, w, base, rng)                 # full range -> teaches carry + carry column
    obs, act, _, _ = interpret(op, body, A, B, D, base, cap=(base + 4) * (w + 2) + 40)
    return obs, act


@torch.no_grad()
def greedy_acc(model, op, base, widths, n=150, seed=0):
    model.eval()
    rep = {}
    for w in widths:
        rng = random.Random(seed + 7919 * w); ok = 0
        for _ in range(n):
            A, B, D = make_problem(op, w, base, rng)
            got, _, halted = controller_run(model, op, A, B, D, base)
            ok += (halted and got == A + B)
        rep[w] = ok / n
    model.train()
    return rep


# ----------------------------------------------------------------------------
# Expert-iteration driver.
# ----------------------------------------------------------------------------
def selfdiscover(model, base=10, iters=250, M=2048, bs_im=256, grad_steps=8, lr=3e-3,
                 wtrain_max=4, warmup=18, buf_max=12000, seed=0, log_every=10, verbose=True):
    op = "add"
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = random.Random(seed)
    buffer = deque(maxlen=buf_max)
    prog = {}
    body = None
    vrng = random.Random(seed + 12345)
    wmax = 1
    first_hit = None; hit_w = None
    tick = IID["TICK"]; ctick = IID["CARRYTICK"]
    history = []
    for it in range(1, iters + 1):
        model.train()
        widths = [1] if wmax == 1 else list(range(2, wmax + 1))
        # NO-CARRY warmup ignites the core LOADA TICK* EMIT loop; then FULL problems force carry discovery.
        nocarry = (body is None and it <= warmup)
        problems = []
        for _ in range(M):
            w = wmax if rng.random() < 0.55 else rng.choice(widths)
            A, B, D = make_problem_g(op, w, base, rng, nocarry=nocarry)
            problems.append((op, A, B, D, w))
        probs_for_vm = [(o, A, B, D) for (o, A, B, D, w) in problems]
        cap = step_cap_for(wmax, base)
        eps = 0.25 if body is None else 0.06
        results = sample_rollouts(model, probs_for_vm, base, cap, temp=1.0, eps=eps)

        n_ok = 0; n_ok_loop = 0
        cand = {}
        for (ok, _, _, o, a), (_, _, _, _, w) in zip(results, problems):
            if not ok or len(a) > len_cap_for(wmax, base) or IID["LOADA"] not in a:
                continue
            # TICK must be loopflag-guarded; CARRYTICK must be cin-guarded (validity: don't tick past the
            # B-digit count / don't add a carry that isn't there). Not the algorithm, just legality.
            if any(a[k] == tick and o[k][0] < 0.5 for k in range(len(a))):
                continue
            if any(a[k] == ctick and o[k][1] < 0.5 for k in range(len(a))):
                continue
            buffer.append((o, a, w)); n_ok += 1
            if first_hit is None:
                first_hit = it
            has_carry = ctick in a
            if w == 1:
                key = tuple(a)
                rec = prog.get(key)
                if rec is None:
                    prog[key] = [o, a, 1]
                else:
                    rec[2] += 1
            if w >= 2:
                n_ok_loop += 1
                if hit_w is None:
                    hit_w = it
            # only FULL rollouts that exercise a carry can yield the complete body -> prioritize them
            if body is None and len(cand) < 160:
                ck = tuple(a)
                if ck not in cand:
                    cand[ck] = (o, w + (100 if has_carry else 0))
        if len(prog) > 3000:
            prog = dict(sorted(prog.items(), key=lambda kv: -kv[1][2])[:1500])

        # open width-2 once the no-carry warmup has ignited the core loop (deterministic, every iter --
        # NOT gated by the logging cadence). Width>=2 exposes the carry column for body extraction.
        if wmax == 1 and body is None and it >= warmup + 12:
            wmax = 2
            if verbose:
                print(f"      -> open width-2 loop search (it {it})")

        if body is None:
            for (act_t), (o, score) in sorted(cand.items(), key=lambda kv: -kv[1][1]):
                b = extract_loop(op, list(act_t), base, vrng)
                if b is not None:
                    body = b
                    if verbose:
                        ps = " ".join(INSTRS[i] + ("*" if lp else "") for (i, lp, g) in b)
                        print(f"      *** DISCOVERED add body: [{ps}]  (it {it}) ***")
                    break

        loss = float("nan")
        targets = []; tw = []
        if body is not None:
            for w in range(1, wtrain_max + 1):
                for _ in range(3):
                    o, a = clean_loop_trace(op, body, w, base, rng)
                    targets.append((o, a)); tw.append(float((w + 1) ** 2))
        else:
            recs1 = list(prog.values())
            if recs1:
                for (o, a, c) in sorted(recs1, key=lambda v: -v[2])[:32]:
                    targets.append((o, a)); tw.append(float(c))
        buf1 = [(o, a) for (o, a, w) in buffer if w == 1 and body is None]
        if targets:
            for _ in range(grad_steps):
                if body is not None:
                    picks = random.choices(targets, weights=tw, k=bs_im)
                else:
                    k = bs_im // 2
                    picks = random.choices(targets, weights=tw, k=k)
                    picks += random.choices(buf1, k=bs_im - k) if buf1 else random.choices(targets, weights=tw, k=bs_im - k)
                loss = imitate(model, opt, picks)

        if it % log_every == 0 or it == 1:
            rep = greedy_acc(model, op, base, list(range(1, max(wmax, 2) + 1)))
            accs = "  ".join(f"w{w}:{rep[w]:.2f}" for w in rep)
            print(f"  it {it:4d}  wmax {wmax}  hits {n_ok:4d} (loop {n_ok_loop:3d})  body[{'Y' if body else '-'}]  "
                  f"loss {loss:.4f}  eps {eps:.2f}  nc {int(nocarry)}  greedy {accs}")
            history.append((it, wmax, n_ok))
    return model, {"first_hit": first_hit, "first_loop_hit": hit_w, "body": body}, history


def length_gen(model, base, widths=(1, 2, 3, 4, 6, 8, 12, 16, 20), n=200):
    print("  LENGTH-GEN (greedy; model emits AND runs the discovered program, no traces shown):")
    rep = greedy_acc(model, "add", base, widths, n=n)
    print("    add: " + "  ".join(f"w{w}:{rep[w]:.3f}" for w in widths))
    return rep


def show_program(model, base, A, B):
    vm = VM("add", A, B, 0, base); h = None; prog = []
    cap = step_cap_for(max(1, vm.N), base) + 10
    model.eval()
    with torch.no_grad():
        steps = 0
        while not vm.halted and steps < cap:
            obs = torch.tensor([vm.obs()], dtype=torch.float32, device=DEVICE).unsqueeze(0)
            logits, h = model(obs, h)
            instr = INSTRS[int(logits[0, -1].argmax())]
            prog.append(instr); vm.execute(instr); steps += 1
    print(f"    {A}+{B} (={A+B}, got {vm.answer()}): {' '.join(prog)}")


if __name__ == "__main__":
    import os
    os.makedirs("runs", exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=250)
    ap.add_argument("--M", type=int, default=2048)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--wtrain_max", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=18)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default="")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    base = 10

    # sanity: reference computes A+B EXACTLY using ONLY the digit-wheel successor (no add primitive)
    rng = random.Random(0)
    for _ in range(4000):
        w = rng.randint(1, 6)
        A, B, D = make_problem("add", w, base, rng)
        vm = VM("add", A, B, 0, base); run_reference(vm)
        assert vm.halted and vm.answer() == A + B, (A, B, vm.answer(), A + B)
    print("reference VM (digit-wheel TICK, NO add primitive) computes A+B EXACTLY on 4000 checks.")
    vm = VM("add", 7, 8, 0, base); tr = run_reference(vm)
    body = extract_loop("add", [i for _, i in tr], base, random.Random(1))
    print("  extract_loop on a clean rollout ->",
          None if body is None else " ".join(INSTRS[i] + ("*" if lp else "") for i, lp, g in body))

    if args.smoke:
        args.iters = 70; args.M = 1536

    print(f"\ndevice={DEVICE}  SELF-DISCOVERY of ADDITION = counting (digit successor), NO traces")
    torch.manual_seed(args.seed)
    model = Controller(hidden=args.hidden)
    print(f"  params: {sum(p.numel() for p in model.parameters())}")
    model, info, _ = selfdiscover(model, base=base, iters=args.iters, M=args.M,
                                  wtrain_max=args.wtrain_max, warmup=args.warmup, seed=args.seed)
    print(f"\n  first exact-correct self-sample (iteration): {info['first_hit']}  first width>=2: {info['first_loop_hit']}")
    if info["body"] is not None:
        print("  DISCOVERED body: " + " ".join(INSTRS[i] + ("*" if lp else "") for i, lp, g in info["body"]))
    print()
    length_gen(model, base)
    print("\n  example emitted programs (the discovered composition):")
    show_program(model, base, 7, 8)
    show_program(model, base, 99, 1)
    show_program(model, base, 4567, 5678)
    if args.save and not args.smoke and info["body"] is not None:
        torch.save(model.state_dict(), args.save)
        print(f"\nsaved {args.save}")
