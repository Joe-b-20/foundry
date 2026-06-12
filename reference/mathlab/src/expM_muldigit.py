"""
expM_muldigit.py — DISCOVER THE PRIMITIVE: single-digit multiply = repeated addition, from outcome alone.

Session 5 / rung 1. expJ discovered the COMPOSITION of multiplication (the per-digit loop
GETDIGIT MULDIGIT SHL ADD_ACC INC_J) from outcome, but MULDIGIT (VAL = A * single-digit) was a GIVEN
atomic ALU instruction. Here we REMOVE it: the only way to build the partial product A*B_j is an inner
flag-gated loop of whole-number ADD (ADD_STEP: VAL += A, K += 1). So the model must DISCOVER that
single-digit multiply IS repeated addition -- mirroring expJ's division, whose inner repeated-subtraction
loop was itself discovered. After this, x and / are structurally identical: outer per-digit loop + inner
repeated-{add,sub} loop, both discovered from outcome.

FLOOR (given, honest): whole-number ADD (VAL+=A), place-value SHL (VAL*=base^J), accumulate (ACC+=VAL),
digit addressing (GETDIGIT), loop counter (INC_J), the inner-loop flag (K<CUR), HALT. NOT given: any
multiplication primitive. DISCOVERED-from-outcome: that the partial product is built by a bounded inner
repeated-add loop, composed into the shift-accumulate outer loop, length-generalizing.

Recipe = expJ's exact-filtered self-imitation: sample stochastic rollouts -> keep ONLY exactly-correct
(binary, no partial credit) -> extract the repeating loop body from the model's OWN correct samples ->
outcome-verify by interpreting on many inputs across widths -> minimize -> distill back -> the model runs
it greedily. NO traces, no program ever shown.

Run: python expM_muldigit.py --iters 200 [--smoke]
"""
from __future__ import annotations
import argparse, random
from collections import deque
import torch, torch.nn as nn

import expA_mealy as E

DEVICE = E.DEVICE

# Minimal ALU: NO single-digit-multiply primitive. The inner loop op is ADD_STEP (repeated add).
INSTRS = ["HALT", "GETDIGIT", "ADD_STEP", "SHL", "ADD_ACC", "INC_J"]
NI = len(INSTRS)
IID = {n: i for i, n in enumerate(INSTRS)}
N_OBS = 4                       # [op_mul, op_div, loopflag, done] -- keep expG's 4-dim shape
LOOP_OP = "ADD_STEP"           # the flag-gated inner-loop op (mirror of div's SUB_D)


# ----------------------------------------------------------------------------
# Multiplication VM with a MINIMAL ALU (no MULDIGIT). Partial product = repeated ADD.
# ----------------------------------------------------------------------------
class VM:
    def __init__(self, op, A, B, D, base):
        self.op, self.A, self.B, self.base = op, A, B, base
        self.ACC = 0; self.CUR = 0; self.VAL = 0; self.J = 0; self.K = 0
        self.N = E._ndigits(B, base)
        self.halted = False

    def obs(self):
        loopflag = 1.0 if self.K < self.CUR else 0.0    # inner repeated-add still owed
        done = 1.0 if self.J >= self.N else 0.0
        return [1.0, 0.0, loopflag, done]

    def execute(self, instr):
        b = self.base
        if instr == "GETDIGIT":
            self.CUR = (self.B // b ** self.J) % b       # digit J of B, LSB-first
            self.VAL = 0; self.K = 0                      # start a fresh partial product for this digit
        elif instr == "ADD_STEP":
            self.VAL += self.A; self.K += 1               # one repeated-addition step (the inner loop body)
        elif instr == "SHL":
            self.VAL = self.VAL * (b ** self.J)           # place-value shift
        elif instr == "ADD_ACC":
            self.ACC += self.VAL
        elif instr == "INC_J":
            self.J += 1
        elif instr == "HALT":
            self.halted = True
        else:
            raise ValueError(instr)

    def answer(self):
        return self.ACC


def run_reference(vm, max_steps=200000):
    """The CORRECT program (sanity only; never shown to the model). Long mult with the inner
    partial-product built by repeated addition."""
    trace = []
    def step(name):
        trace.append((vm.obs(), IID[name])); vm.execute(name)
    while not vm.halted:
        step("GETDIGIT")
        while vm.K < vm.CUR:
            step("ADD_STEP")
        step("SHL"); step("ADD_ACC"); step("INC_J")
        if vm.J >= vm.N:
            step("HALT")
        if len(trace) > max_steps:
            break
    return trace


def make_problem(op, width, base, rng):
    """EVAL distribution: full range incl. zeros / leading zeros (edge coverage)."""
    A = rng.randint(0, base ** width - 1)
    B = rng.randint(0, base ** width - 1)
    return A, B, 0


def make_problem_g(op, w, base, rng):
    """DISCOVERY distribution: genuine width (nonzero leading digit) + nonzero answer, so a degenerate
    program that skips the loop / ignores A is WRONG (no lucky hits). Does NOT encode the solution."""
    lo = 1 if w == 1 else base ** (w - 1)
    hi = base ** w - 1
    return rng.randint(lo, hi), rng.randint(lo, hi), 0


# ----------------------------------------------------------------------------
# Controller: GRU over the 4-dim observation -> next instruction (NI=6 here).
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
        step_cap = (base + 6) * max(1, vm.N) + 20
    h = None; steps = 0
    while not vm.halted and steps < step_cap:
        obs = torch.tensor([vm.obs()], dtype=torch.float32, device=DEVICE).unsqueeze(0)
        logits, h = model(obs, h)
        instr = INSTRS[int(logits[0, -1].argmax())]
        vm.execute(instr); steps += 1
    return vm.answer(), steps, vm.halted


def step_cap_for(op, w, base):
    return (base + 6) * w + 10


def len_cap_for(op, w, base):
    return (base + 8) * w + 12


# ----------------------------------------------------------------------------
# Batched lockstep sampling.
# ----------------------------------------------------------------------------
def sample_rollouts(model, problems, base, step_cap, temp=1.0, eps=0.0):
    M = len(problems)
    vms = [VM(op, A, B, D, base) for (op, A, B, D) in problems]
    legal = torch.ones(M, NI, device=DEVICE) / NI
    h = None
    tr_obs = [[] for _ in range(M)]
    tr_act = [[] for _ in range(M)]
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
        exp = A * B
        ok = vms[i].halted and vms[i].answer() == exp
        out.append((ok, op, problems[i][1:], tr_obs[i], tr_act[i]))
    return out


_lossf = nn.CrossEntropyLoss(reduction="none")


def imitate(model, opt, traces):
    M = len(traces)
    L = max(len(a) for _, a in traces)
    obs_b = torch.zeros(M, L, N_OBS); tgt = torch.zeros(M, L, dtype=torch.long); mask = torch.zeros(M, L)
    for i, (o, a) in enumerate(traces):
        k = len(a)
        obs_b[i, :k] = torch.tensor(o, dtype=torch.float32)
        tgt[i, :k] = torch.tensor(a, dtype=torch.long)
        mask[i, :k] = 1.0
    obs_b, tgt, mask = obs_b.to(DEVICE), tgt.to(DEVICE), mask.to(DEVICE)
    logits, _ = model(obs_b)
    loss = (_lossf(logits.reshape(-1, NI), tgt.reshape(-1)) * mask.reshape(-1)).sum() / mask.sum()
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return loss.item()


# ----------------------------------------------------------------------------
# Loop extraction + interpretation (ADD_STEP is the flag-gated inner-loop op, mirror of div's SUB_D).
# ----------------------------------------------------------------------------
def cycle_to_body(cycle):
    loop = IID[LOOP_OP]
    body = []
    i = 0
    while i < len(cycle):
        j = i
        while j < len(cycle) and cycle[j] == cycle[i]:
            j += 1
        is_loop = (cycle[i] == loop) or ((j - i) >= 2)   # the inner op (any count) or any repeated instr
        body.append((cycle[i], is_loop))
        i = j
    return tuple(body)


def segment_bodies(act):
    inc, halt = IID["INC_J"], IID["HALT"]
    segs, cur = [], []
    for x in act:
        cur.append(x)
        if x == inc:
            segs.append(tuple(cur)); cur = []
    if cur != [halt] or not segs:
        return []
    return list(dict.fromkeys(cycle_to_body(s) for s in segs))


def interpret(op, body, A, B, D, base, cap=20000):
    vm = VM(op, A, B, D, base)
    obs, act = [], []
    steps = 0
    while not vm.halted and steps < cap:
        for (iid, is_loop) in body:
            if is_loop:
                while vm.obs()[2] >= 0.5 and not vm.halted and steps < cap:   # while loopflag
                    obs.append(vm.obs()); act.append(iid); vm.execute(INSTRS[iid]); steps += 1
            else:
                obs.append(vm.obs()); act.append(iid); vm.execute(INSTRS[iid]); steps += 1
            if vm.halted:
                break
        if not vm.halted and vm.obs()[3] >= 0.5:          # done -> HALT
            obs.append(vm.obs()); act.append(IID["HALT"]); vm.execute("HALT"); steps += 1
    return obs, act, vm.answer(), vm.halted


def verify_body(op, body, base, rng, widths=(2, 3, 5), n=24):
    for w in widths:
        for _ in range(n):
            A, B, D = make_problem_g(op, w, base, rng)
            _, _, got, halted = interpret(op, body, A, B, D, base, cap=(base + 8) * w + 40)
            if not (halted and got == A * B):
                return False
    return True


def minimize_body(op, body, base, rng):
    body = list(body)
    changed = True
    while changed and len(body) > 1:
        changed = False
        for i in range(len(body)):
            cand = tuple(body[:i] + body[i + 1:])
            if cand and cand[-1][0] == IID["INC_J"] and verify_body(op, cand, base, rng, n=40):
                body = list(cand); changed = True; break
    return tuple(body)


def extract_loop(op, act, base, rng):
    for body in segment_bodies(act):
        if body and body[-1][0] == IID["INC_J"] and verify_body(op, body, base, rng):
            return minimize_body(op, body, base, rng)
    return None


def clean_loop_trace(op, body, w, base, rng):
    A, B, D = make_problem_g(op, w, base, rng)
    obs, act, _, _ = interpret(op, body, A, B, D, base, cap=(base + 8) * w + 40)
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
            ok += (halted and got == A * B)
        rep[w] = ok / n
    model.train()
    return rep


# ----------------------------------------------------------------------------
# Expert-iteration driver (single op = mul; adapted from expJ).
# ----------------------------------------------------------------------------
def selfdiscover(model, base=10, iters=200, M=2048, bs_im=256, grad_steps=8, lr=3e-3,
                 wtrain_max=4, buf_max=12000, seed=0, log_every=10, verbose=True):
    op = "mul"
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = random.Random(seed)
    buffer = deque(maxlen=buf_max)
    prog = {}                       # (action-tuple) -> [obs, act, count] : width-1 consensus (ignition)
    body = None                     # DISCOVERED canonical loop body
    vrng = random.Random(seed + 12345)
    wmax = 1
    first_hit = None; hit_w = None
    history = []
    loopid = IID[LOOP_OP]
    for it in range(1, iters + 1):
        model.train()
        widths = [1] if wmax == 1 else list(range(2, wmax + 1))
        problems = []
        for _ in range(M):
            w = wmax if rng.random() < 0.55 else rng.choice(widths)
            A, B, D = make_problem_g(op, w, base, rng)
            problems.append((op, A, B, D, w))
        probs_for_vm = [(o, A, B, D) for (o, A, B, D, w) in problems]
        cap = step_cap_for(op, wmax, base)
        eps = 0.25 if body is None else 0.06
        results = sample_rollouts(model, probs_for_vm, base, cap, temp=1.0, eps=eps)

        n_ok = 0; n_ok_loop = 0
        cand = {}
        for (ok, _, _, o, a), (_, _, _, _, w) in zip(results, problems):
            if not ok or len(a) > len_cap_for(op, wmax, base) or IID["GETDIGIT"] not in a:
                continue
            # the inner op (ADD_STEP) must be FLAG-GUARDED: only added while loopflag=1 (K<CUR).
            # An unguarded ADD_STEP over-adds past the digit count -- a validity constraint (don't run
            # the inner loop past its range), NOT the algorithm.
            if any(a[i] == loopid and o[i][2] < 0.5 for i in range(len(a))):
                continue
            buffer.append((o, a, w)); n_ok += 1
            if first_hit is None:
                first_hit = it
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
            if body is None and len(cand) < 120:
                ck = tuple(a)
                if ck not in cand:
                    cand[ck] = (o, w)
        if len(prog) > 3000:
            prog = dict(sorted(prog.items(), key=lambda kv: -kv[1][2])[:1500])

        # DISCOVER THE LOOP from the model's own correct rollouts (widest first).
        if body is None:
            for (act_t), (o, w) in sorted(cand.items(), key=lambda kv: -kv[1][1]):
                b = extract_loop(op, list(act_t), base, vrng)
                if b is not None:
                    body = b
                    if verbose:
                        ps = " ".join(INSTRS[i] + ("*" if lp else "") for (i, lp) in b)
                        print(f"      *** DISCOVERED mul loop body: [{ps}]  (it {it}) ***")
                    break

        loss = float("nan")
        targets = []; tw = []
        if body is not None:
            for w in range(2, wtrain_max + 1):
                for _ in range(3):
                    o, a = clean_loop_trace(op, body, w, base, rng)
                    targets.append((o, a)); tw.append(float(w * w))
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
                  f"loss {loss:.4f}  eps {eps:.2f}  greedy {accs}")
            history.append((it, wmax, n_ok))
            if wmax == 1 and (rep.get(1, 0.0) > 0.6 or it >= 30) and body is None:
                wmax = 2
                if verbose:
                    print(f"      -> open width-2 loop search (greedy w1 {rep.get(1,0):.2f}, it {it})")
    return model, {"first_hit": first_hit, "first_loop_hit": hit_w, "body": body}, history


def length_gen(model, base, widths=(1, 2, 3, 4, 6, 8, 12, 16, 20), n=200):
    print("  LENGTH-GEN (greedy; model emits AND runs the discovered program, no traces shown):")
    rep = greedy_acc(model, "mul", base, widths, n=n)
    print("    mul: " + "  ".join(f"w{w}:{rep[w]:.3f}" for w in widths))
    return rep


def show_program(model, base, A, B):
    vm = VM("mul", A, B, 0, base); h = None; prog = []
    cap = step_cap_for("mul", max(1, E._ndigits(B, base)), base) + 10
    model.eval()
    with torch.no_grad():
        steps = 0
        while not vm.halted and steps < cap:
            obs = torch.tensor([vm.obs()], dtype=torch.float32, device=DEVICE).unsqueeze(0)
            logits, h = model(obs, h)
            instr = INSTRS[int(logits[0, -1].argmax())]
            prog.append(instr); vm.execute(instr); steps += 1
    print(f"    {A}*{B} (={A*B}, got {vm.answer()}): {' '.join(prog)}")


if __name__ == "__main__":
    import os
    os.makedirs("runs", exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--M", type=int, default=2048)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--wtrain_max", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default="")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    base = 10

    # sanity: reference program computes mul EXACTLY using ONLY repeated-add (no MULDIGIT primitive)
    rng = random.Random(0)
    for _ in range(3000):
        w = rng.randint(1, 6)
        A, B, D = make_problem("mul", w, base, rng)
        vm = VM("mul", A, B, 0, base); run_reference(vm)
        assert vm.halted and vm.answer() == A * B, (A, B, vm.answer(), A * B)
    print("reference VM (repeated-add, NO MULDIGIT) computes mul EXACTLY on 3000 checks.")
    # sanity: extract_loop recovers the canonical body from a clean reference rollout
    vm = VM("mul", 47, 83, 0, base); tr = run_reference(vm)
    acts = [i for _, i in tr]
    body = extract_loop("mul", acts, base, random.Random(1))
    print("  extract_loop on a clean rollout ->",
          None if body is None else " ".join(INSTRS[i] + ("*" if lp else "") for i, lp in body))

    if args.smoke:
        args.iters = 50; args.M = 1024

    print(f"\ndevice={DEVICE}  SELF-DISCOVERY of MULDIGIT=repeated-add (exact filter, NO traces)")
    torch.manual_seed(args.seed)
    model = Controller(hidden=args.hidden)
    print(f"  params: {sum(p.numel() for p in model.parameters())}")
    model, info, _ = selfdiscover(model, base=base, iters=args.iters, M=args.M,
                                  wtrain_max=args.wtrain_max, seed=args.seed)
    print(f"\n  first exact-correct self-sample (iteration): {info['first_hit']}  "
          f"first width>=2: {info['first_loop_hit']}")
    if info["body"] is not None:
        print("  DISCOVERED body: " +
              " ".join(INSTRS[i] + ("*" if lp else "") for i, lp in info["body"]))
    print()
    length_gen(model, base)
    print("\n  example emitted programs (the discovered composition):")
    show_program(model, base, 47, 83)
    show_program(model, base, 9, 9)
    show_program(model, base, 7, 100)
    if args.save and not args.smoke:
        torch.save(model.state_dict(), args.save)
        print(f"\nsaved {args.save}")
