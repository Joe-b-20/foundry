"""
expP_newton.py — isqrt from OUTCOME ALONE with a DIVISION-based VM: does it discover NEWTON/HERON?

Companion to expO_isqrt.py. expO gave a candidate-search VM (square + compare) and outcome-discovery
found BINARY SEARCH. The project's recurring law is that WHICH algorithm is discovered is REPRESENTATION/
PRIMITIVE-dependent (divisor-vs-base for division; mod-vs-sub for gcd). So: change the primitives to a
DIVISION-based update and see whether the SAME recipe discovers a structurally DIFFERENT isqrt algorithm.

VM registers X (estimate), Y (next), over input N. Heron/Newton integer sqrt:
  x = n; repeat y=(x + n//x)//2; if y>=x: stop (answer x) else x=y.
Instructions (general, division-based -- division was discovered in earlier sessions, reused as a primitive):
  NEWTON  Y=(X + N//X)//2   (the Heron averaging update; needs X>0)
  STEP    X=Y ; Y=0          (accept the new estimate;  valid iff fresh-NEWTON and Y<X)
  HALT    output X
obs = [conv = (Y>=X)].  conv is the convergence/termination signal (fresh right after a NEWTON; STEP
zeroes Y so conv reads 0 again -> 'keep going').  The model must, via recurrence, do NEWTON, then HALT if
converged else STEP, looped -- the Newton ITERATION and its floor-CONVERGENCE rule (halt when y>=x) are
what get discovered from outcome.  Same exact-filtered self-imitation + extract+verify+distill as expO/expJ.
Run: python expP_newton.py --iters 150
"""
from __future__ import annotations
import argparse, math, random
import torch, torch.nn as nn

import expA_mealy as E

DEVICE = E.DEVICE
INSTRS = ["HALT", "NEWTON", "STEP"]
NI = len(INSTRS)
IID = {n: i for i, n in enumerate(INSTRS)}
N_OBS = 1                                # [conv = (Y>=X)]


def mask_vec():
    return torch.zeros(NI, device=DEVICE)


class NewtonVM:
    def __init__(self, n):
        self.N = n; self.X = n; self.Y = 0; self.fresh = False
        self.halted = False; self.invalid = False

    def obs(self):
        return [1.0 if self.Y >= self.X else 0.0]

    def execute(self, instr):
        if instr == "NEWTON":
            if self.X <= 0:
                self.invalid = True
            else:
                self.Y = (self.X + self.N // self.X) // 2; self.fresh = True
        elif instr == "STEP":
            if self.fresh and self.Y < self.X:
                self.X = self.Y; self.Y = 0; self.fresh = False
            else:
                self.invalid = True
        elif instr == "HALT":
            self.halted = True

    def answer(self):
        return self.X


class Controller(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.gru = nn.GRU(N_OBS, hidden, batch_first=True)
        self.head = nn.Linear(hidden, NI)

    def forward(self, obs, h=None):
        y, h = self.gru(obs, h)
        return self.head(y), h


def cap_for(w, mult, add):
    return mult * w + add


def make_problem(w, rng):
    lo = 1 if w == 1 else 10 ** (w - 1)
    return rng.randint(lo, 10 ** w - 1)


def consistent(o, a):
    """VALIDITY filter: STEP only when NOT converged (conv=0); HALT only when converged (conv=1)."""
    st, ha = IID["STEP"], IID["HALT"]
    for ob, ac in zip(o, a):
        if (ac == st and ob[0] >= 0.5) or (ac == ha and ob[0] < 0.5):
            return False
    return True


def canon_body(a):
    """Return 'NEWTON' if `a` is a clean (NEWTON STEP)* NEWTON HALT; else None."""
    nw, st, ha = IID["NEWTON"], IID["STEP"], IID["HALT"]
    i = 0
    while i < len(a):
        if a[i] != nw:
            return None
        if i + 1 < len(a) and a[i + 1] == st:
            i += 2; continue
        if i + 1 == len(a) - 1 and a[i + 1] == ha:
            return "NEWTON"
        return None
    return None


def interpret(n, cap):
    """Run 'loop: NEWTON; if conv HALT else STEP' on one problem within the budget."""
    vm = NewtonVM(n); steps = 0
    while not vm.halted and not vm.invalid and steps < cap:
        vm.execute("NEWTON"); steps += 1
        if vm.invalid:
            break
        if vm.obs()[0] >= 0.5:
            vm.execute("HALT"); steps += 1
        else:
            vm.execute("STEP"); steps += 1
    return vm.answer(), (vm.halted and not vm.invalid)


def verify_body(mult, add, rng, widths=(2, 3, 5, 8), n=24):
    for w in widths:
        cap = cap_for(w, mult, add)
        for _ in range(n):
            x = make_problem(w, rng)
            got, halted = interpret(x, cap)
            if not (halted and got == math.isqrt(x)):
                return False
    return True


def body_trace(w, mult, add, rng):
    x = make_problem(w, rng); vm = NewtonVM(x); obs, act = [], []; cap = cap_for(w, mult, add); steps = 0
    while not vm.halted and not vm.invalid and steps < cap:
        obs.append(vm.obs()); act.append(IID["NEWTON"]); vm.execute("NEWTON"); steps += 1
        nm = "HALT" if vm.obs()[0] >= 0.5 else "STEP"
        obs.append(vm.obs()); act.append(IID[nm]); vm.execute(nm); steps += 1
    return obs, act


_lossf = nn.CrossEntropyLoss(reduction="none")


def imitate(model, opt, traces):
    M = len(traces); L = max(len(a) for _, a in traces)
    obs_b = torch.zeros(M, L, N_OBS); tgt = torch.zeros(M, L, dtype=torch.long); mask = torch.zeros(M, L)
    for i, (o, a) in enumerate(traces):
        k = len(a)
        obs_b[i, :k] = torch.tensor(o, dtype=torch.float32)
        tgt[i, :k] = torch.tensor(a, dtype=torch.long); mask[i, :k] = 1.0
    obs_b, tgt, mask = obs_b.to(DEVICE), tgt.to(DEVICE), mask.to(DEVICE)
    logits, _ = model(obs_b)
    loss = (_lossf(logits.reshape(-1, NI), tgt.reshape(-1)) * mask.reshape(-1)).sum() / mask.sum()
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return loss.item()


def sample_rollouts(model, ns, step_cap, temp=1.0, eps=0.0):
    M = len(ns); vms = [NewtonVM(n) for n in ns]; h = None
    tr_obs = [[] for _ in range(M)]; tr_act = [[] for _ in range(M)]; active = [True] * M
    legal = torch.ones(NI, device=DEVICE) / NI
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
            if vms[i].halted or vms[i].invalid:
                active[i] = False
    return [(vms[i].halted and not vms[i].invalid and vms[i].answer() == math.isqrt(ns[i]), tr_obs[i], tr_act[i])
            for i in range(M)]


@torch.no_grad()
def controller_run(model, n, cap):
    vm = NewtonVM(n); h = None; steps = 0
    while not vm.halted and not vm.invalid and steps < cap:
        obs = torch.tensor([vm.obs()], dtype=torch.float32, device=DEVICE).unsqueeze(0)
        logits, h = model(obs, h)
        vm.execute(INSTRS[int(logits[0, -1].argmax())]); steps += 1
    return vm.answer(), (vm.halted and not vm.invalid)


@torch.no_grad()
def greedy_acc(model, w, mult, add, n=200, seed=0):
    model.eval(); rng = random.Random(seed + 7919 * w); ok = 0; cap = cap_for(w, mult, add)
    for _ in range(n):
        x = make_problem(w, rng); got, halted = controller_run(model, x, cap)
        ok += (halted and got == math.isqrt(x))
    model.train(); return ok / n


def emit_program(model, n, cap):
    vm = NewtonVM(n); h = None; prog = []; steps = 0; model.eval()
    with torch.no_grad():
        while not vm.halted and not vm.invalid and steps < cap:
            obs = torch.tensor([vm.obs()], dtype=torch.float32, device=DEVICE).unsqueeze(0)
            logits, h = model(obs, h)
            instr = INSTRS[int(logits[0, -1].argmax())]
            prog.append(instr); vm.execute(instr); steps += 1
    model.train(); return prog, vm.answer(), (vm.halted and not vm.invalid)


def selfdiscover(model, iters=150, M=2048, bs_im=256, grad_steps=6, lr=3e-3, w_ignite=2,
                 wtrain_max=6, mult=12, add=12, seed=0, log_every=10, verbose=True):
    model.to(DEVICE); opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = random.Random(seed); vrng = random.Random(seed + 12345)
    distinct = {}; locked = None; first_hit = None; last_loss = float("nan"); history = []
    for it in range(1, iters + 1):
        model.train()
        if locked is None:
            ns = [make_problem(rng.randint(1, w_ignite), rng) for _ in range(M)]
            results = sample_rollouts(model, ns, cap_for(w_ignite, mult, add), temp=1.0, eps=0.3)
            for ok, o, a in results:
                if not ok or not consistent(o, a) or len(a) < 3:
                    continue
                if first_hit is None:
                    first_hit = it
                distinct.setdefault(tuple(a), (o, a))
            for (o, a) in list(distinct.values()):
                if canon_body(a) is not None and verify_body(mult, add, vrng):
                    locked = "NEWTON"
                    if verbose:
                        print(f"      *** DISCOVERED & LOCKED body: NEWTON loop (it {it}, from {len(distinct)} distinct correct programs) ***")
                    break
            if distinct and not locked:
                items = list(distinct.values())
                picks = [random.choice(items) for _ in range(bs_im)]
                for _ in range(grad_steps):
                    last_loss = imitate(model, opt, picks)
        else:
            targets, tw = [], []
            for w in range(1, wtrain_max + 1):
                for _ in range(3):
                    targets.append(body_trace(w, mult, add, rng)); tw.append(float(w * w))
            for _ in range(grad_steps):
                last_loss = imitate(model, opt, random.choices(targets, weights=tw, k=bs_im))
        if it % log_every == 0 or it == 1:
            accs = {w: greedy_acc(model, w, mult, add, n=120, seed=it) for w in (1, 3, 6, 12)}
            astr = " ".join(f"w{w}:{accs[w]:.2f}" for w in accs)
            print(f"  it {it:4d}  locked[{locked if locked else '-'}]  distinct {len(distinct):4d}  loss {last_loss:.4f}  [{astr}]")
            history.append((it, locked, accs))
    return model, {"first_hit": first_hit, "locked": locked}, history


def newton_step_counts(mult, add, widths=(1, 2, 3, 4, 6, 8, 12, 20, 30)):
    """Show Newton's iteration count is O(log n) (a handful), so it fits the step budget at all widths."""
    print("\n  Newton iteration counts (median over 200 random n per width) -- O(log n), fits the budget:")
    for w in widths:
        rng = random.Random(7 + w); cs = []
        for _ in range(200):
            x = make_problem(w, rng); vm = NewtonVM(x); steps = 0
            while not vm.halted and not vm.invalid and steps < 10 ** 6:
                vm.execute("NEWTON")
                if vm.obs()[0] >= 0.5:
                    vm.execute("HALT")
                else:
                    vm.execute("STEP")
                steps += 1
            cs.append(steps)
        cs.sort()
        print(f"    w{w:2d}: median {cs[len(cs)//2]:3d} NEWTON iters   (budget cap {cap_for(w, mult, add)})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--M", type=int, default=2048)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--mult", type=int, default=12)
    ap.add_argument("--add", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default="")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.iters = 80

    for _ in range(3000):
        x = random.randint(1, 10 ** 7)
        g, h = interpret(x, 100000)
        assert h and g == math.isqrt(x), ("newton ref", x, g, math.isqrt(x))
    print("reference NEWTON body computes isqrt exactly (sanity, n up to 1e7).")

    print(f"\ndevice={DEVICE}  SELF-DISCOVERY of isqrt (DIVISION-based VM) from OUTCOME ALONE  cap={args.mult}*w+{args.add}")
    torch.manual_seed(args.seed)
    model = Controller(hidden=args.hidden)
    print(f"  params: {sum(p.numel() for p in model.parameters())}")
    model, info, _ = selfdiscover(model, iters=args.iters, M=args.M, mult=args.mult, add=args.add, seed=args.seed)
    print(f"\n  first exact-correct self-sample at iter: {info['first_hit']}")
    print(f"  DISCOVERED algorithm: {info['locked']}  =>  {'NEWTON / HERON iteration' if info['locked'] else 'NONE (not discovered)'}")

    print("\n  LENGTH-GEN of the discovered model (greedy; emits AND runs the program, within budget):")
    widths = (1, 2, 3, 4, 6, 8, 12, 16, 20, 25, 30)
    rep = {w: greedy_acc(model, w, args.mult, args.add, n=200, seed=100 + w) for w in widths}
    print("    " + "  ".join(f"w{w}:{rep[w]:.3f}" for w in widths))

    print("\n  example emitted programs (the discovered Newton iteration):")
    for x in (4, 81, 1000, 123456, 9999999999):
        prog, got, halted = emit_program(model, x, cap_for(len(str(x)), args.mult, args.add))
        print(f"    isqrt({x})={math.isqrt(x)} got {got} ok={got == math.isqrt(x)}: {' '.join(prog)}")

    newton_step_counts(args.mult, args.add)
    if args.save and not args.smoke:
        torch.save(model.state_dict(), args.save)
        print(f"\nsaved {args.save}")
