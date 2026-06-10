"""
expO_isqrt.py — discover integer square root isqrt(n)=floor(sqrt(n)) from OUTCOME ALONE; WHICH algorithm?

isqrt has NO single clean digit-serial schoolbook form (flagged as a moonshot-adjacent op in
sessions 2/4/5). Several real algorithms exist: linear scan, repeated odd-number subtraction
(both O(sqrt n)); binary search and Newton/Heron (both O(log n)); schoolbook digit-by-digit (O(digits)).

Sharpest version of the project's recurring question (cf. expK GCD = Euclid): under the project's
EXACT-LENGTH-GEN-UNDER-A-STEP-BUDGET criterion, which does outcome-only self-discovery pick? We give a
whole-number CANDIDATE-SEARCH VM with GENERAL primitives so BOTH a naive scan and an efficient
bisection are expressible with the SAME ops -- the budget gets to select.

VM registers LO,HI,MID,S over input N; invariant LO^2 <= N < HI^2, answer = LO when HI-LO==1.
Instructions (all GENERAL ops, none isqrt-specific):
  AVG     MID=(LO+HI)//2 ; S=MID*MID    (bisection probe: floor-average of two registers, then square)
  NEXT    MID=LO+1       ; S=MID*MID     (linear probe: successor, then square)
  TAKE_LO LO=MID                         (keep lower half;  valid iff fresh-probe and LO<MID<HI)
  TAKE_HI HI=MID                         (keep upper half;  valid iff fresh-probe and LO<MID<HI)
  HALT    output LO
obs = [done=(HI-LO<=1), le=(S<=N)].  (squaring is folded into the probe so `le` always reflects the
CURRENT candidate; a TAKE is invalid unless it follows a fresh probe -- the analog of expJ's ge-guard.)

Recipe = expJ exact-filtered self-imitation: sample stochastic rollouts -> keep ONLY exactly-correct
(within the step budget) -> EXTRACT the per-iteration loop body from the model's own correct samples ->
verify it length-generalizes within budget -> distill it back. The naive (linear/NEXT) body busts the
budget at w>=3 so it never verifies; only the bisection (AVG) body does. So the discovered algorithm =
whatever generalizing body the model's correct samples contain. Trained ONLY on outcomes; no trace shown.
Run: python expO_isqrt.py --iters 200
"""
from __future__ import annotations
import argparse, math, random
from collections import deque
import torch, torch.nn as nn

import expA_mealy as E

DEVICE = E.DEVICE
INSTRS = ["HALT", "AVG", "NEXT", "TAKE_LO", "TAKE_HI"]
NI = len(INSTRS)
IID = {n: i for i, n in enumerate(INSTRS)}
N_OBS = 2                                # [done=(HI-LO<=1), le=(S<=N)]
ALLOWED = set(INSTRS)                    # restrict (e.g. drop AVG) for contrast runs


def mask_vec():
    return torch.tensor([0.0 if INSTRS[i] in ALLOWED else -1e9 for i in range(NI)], device=DEVICE)


class SqrtVM:
    def __init__(self, n):
        self.N = n
        self.LO = 0
        self.HI = n + 1                  # invariant LO^2 <= N < HI^2 holds initially (0, n+1)
        self.MID = 0
        self.S = 0
        self.fresh = False               # True iff a probe happened since the last TAKE (forces probe->take)
        self.halted = False
        self.invalid = False

    def obs(self):
        done = 1.0 if (self.HI - self.LO) <= 1 else 0.0
        le = 1.0 if self.S <= self.N else 0.0
        return [done, le]

    def execute(self, instr):
        if instr == "AVG":
            self.MID = (self.LO + self.HI) // 2; self.S = self.MID * self.MID; self.fresh = True
        elif instr == "NEXT":
            self.MID = self.LO + 1; self.S = self.MID * self.MID; self.fresh = True
        elif instr == "TAKE_LO":
            if self.fresh and self.LO < self.MID < self.HI:
                self.LO = self.MID; self.fresh = False
            else:
                self.invalid = True
        elif instr == "TAKE_HI":
            if self.fresh and self.LO < self.MID < self.HI:
                self.HI = self.MID; self.fresh = False
            else:
                self.invalid = True
        elif instr == "HALT":
            self.halted = True

    def answer(self):
        return self.LO


class Controller(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.gru = nn.GRU(N_OBS, hidden, batch_first=True)
        self.head = nn.Linear(hidden, NI)

    def forward(self, obs, h=None):
        y, h = self.gru(obs, h)
        return self.head(y), h


def cap_for(w, mult, add):
    """Step budget: tight enough that LINEAR scan (O(sqrt n) ~ 10^(w/2) probes) busts it by w=3,
    while BINARY search (O(log n) ~ 3.3*w iters * 2 instrs ~ 7w) comfortably fits."""
    return mult * w + add


def make_problem(w, rng):
    lo = 1 if w == 1 else 10 ** (w - 1)
    return rng.randint(lo, 10 ** w - 1)


def consistent(o, a):
    """VALIDITY filter (analog of expJ's ge-guarded SUB_D): a TAKE must agree with the FRESH comparison
    -- TAKE_LO only when le=1, TAKE_HI only when le=0. Drops 'lucky' rollouts that narrowed against the
    evidence. Does NOT specify AVG-vs-NEXT, iteration count, or halt timing (the algorithm)."""
    lo, hi = IID["TAKE_LO"], IID["TAKE_HI"]
    for ob, ac in zip(o, a):
        if (ac == lo and ob[1] < 0.5) or (ac == hi and ob[1] >= 0.5):
            return False
    return True


# ----------------------------------------------------------------------------
# Loop extraction. Each iteration of a clean rollout is [probe, take] (freshness forces probe->take),
# ending in HALT. The body is canonicalized to the PROBE op used (AVG or NEXT); the take is "narrow per
# the le flag". So extraction = read the probe op the model used; then VERIFY that looping that body
# length-generalizes WITHIN the budget. Only AVG (bisection) verifies; NEXT (linear) busts the cap.
# ----------------------------------------------------------------------------
def canon_body(a):
    """Return the probe-op name if `a` is a clean (probe take)* HALT with ONE probe op; else None."""
    avg, nxt, lo, hi, halt = IID["AVG"], IID["NEXT"], IID["TAKE_LO"], IID["TAKE_HI"], IID["HALT"]
    probes = set()
    i = 0
    while i < len(a):
        if a[i] == halt:
            return None if i != len(a) - 1 else (("AVG" if avg in probes else "NEXT") if len(probes) == 1 else None)
        if a[i] not in (avg, nxt) or i + 1 >= len(a) or a[i + 1] not in (lo, hi):
            return None
        probes.add(a[i]); i += 2
    return None


def interpret(probe, n, cap):
    """Run 'while not done: probe; narrow per le; then HALT' on one problem, within the step budget."""
    vm = SqrtVM(n); steps = 0
    while not vm.halted and not vm.invalid and steps < cap:
        if vm.obs()[0] >= 0.5:
            vm.execute("HALT"); steps += 1; continue
        vm.execute(probe); steps += 1
        if vm.invalid:
            break
        vm.execute("TAKE_LO" if vm.obs()[1] >= 0.5 else "TAKE_HI"); steps += 1
    return vm.answer(), (vm.halted and not vm.invalid)


def verify_body(probe, mult, add, rng, widths=(2, 3, 5, 8), n=24):
    for w in widths:
        cap = cap_for(w, mult, add)
        for _ in range(n):
            x = make_problem(w, rng)
            got, halted = interpret(probe, x, cap)
            if not (halted and got == math.isqrt(x)):
                return False
    return True


def body_trace(probe, w, mult, add, rng):
    """Distillation target: interpret the discovered body on a random width-w problem."""
    x = make_problem(w, rng)
    vm = SqrtVM(x); obs, act = [], []; cap = cap_for(w, mult, add); steps = 0
    while not vm.halted and not vm.invalid and steps < cap:
        if vm.obs()[0] >= 0.5:
            obs.append(vm.obs()); act.append(IID["HALT"]); vm.execute("HALT"); steps += 1; continue
        obs.append(vm.obs()); act.append(IID[probe]); vm.execute(probe); steps += 1
        nm = "TAKE_LO" if vm.obs()[1] >= 0.5 else "TAKE_HI"
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
    M = len(ns); vms = [SqrtVM(n) for n in ns]; h = None
    tr_obs = [[] for _ in range(M)]; tr_act = [[] for _ in range(M)]; active = [True] * M
    legal = (mask_vec() > -1e8).float(); legal = legal / legal.sum()
    for _ in range(step_cap):
        if not any(active):
            break
        obs_rows = [vm.obs() for vm in vms]
        obs = torch.tensor(obs_rows, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        logits, h = model(obs, h)
        probs = torch.softmax((logits[:, -1, :] + mask_vec()) / temp, dim=-1)
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
    vm = SqrtVM(n); h = None; steps = 0
    while not vm.halted and not vm.invalid and steps < cap:
        obs = torch.tensor([vm.obs()], dtype=torch.float32, device=DEVICE).unsqueeze(0)
        logits, h = model(obs, h)
        vm.execute(INSTRS[int((logits[0, -1] + mask_vec()).argmax())]); steps += 1
    return vm.answer(), (vm.halted and not vm.invalid)


@torch.no_grad()
def greedy_acc(model, w, mult, add, n=200, seed=0):
    model.eval(); rng = random.Random(seed + 7919 * w); ok = 0; cap = cap_for(w, mult, add)
    for _ in range(n):
        x = make_problem(w, rng); got, halted = controller_run(model, x, cap)
        ok += (halted and got == math.isqrt(x))
    model.train(); return ok / n


def emit_program(model, n, cap):
    vm = SqrtVM(n); h = None; prog = []; steps = 0; model.eval()
    with torch.no_grad():
        while not vm.halted and not vm.invalid and steps < cap:
            obs = torch.tensor([vm.obs()], dtype=torch.float32, device=DEVICE).unsqueeze(0)
            logits, h = model(obs, h)
            instr = INSTRS[int((logits[0, -1] + mask_vec()).argmax())]
            prog.append(instr); vm.execute(instr); steps += 1
    model.train(); return prog, vm.answer(), (vm.halted and not vm.invalid)


# ----------------------------------------------------------------------------
# Expert-iteration driver. Phase 1: explore at small width, imitate a DIVERSE set of correct programs
# (keep entropy so rare bisection rollouts survive), and each iter try to EXTRACT a generalizing body.
# Phase 2: once a body verifies length-gen within budget, LOCK + distill it across widths.
# ----------------------------------------------------------------------------
def selfdiscover(model, iters=200, M=2048, bs_im=256, grad_steps=6, lr=3e-3, w_ignite=2,
                 wtrain_max=6, mult=12, add=12, seed=0, log_every=10, verbose=True):
    model.to(DEVICE); opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = random.Random(seed); vrng = random.Random(seed + 12345)
    distinct = {}                                  # action-tuple -> (obs, act) : DIVERSE correct programs
    locked = None; first_hit = None; last_loss = float("nan"); history = []
    for it in range(1, iters + 1):
        model.train()
        if locked is None:
            ns = [make_problem(rng.randint(1, w_ignite), rng) for _ in range(M)]
            cap = cap_for(w_ignite, mult, add)
            results = sample_rollouts(model, ns, cap, temp=1.0, eps=0.3)
            for ok, o, a in results:
                if not ok or not consistent(o, a) or len(a) < 3:
                    continue
                if first_hit is None:
                    first_hit = it
                distinct.setdefault(tuple(a), (o, a))
            # EXTRACT: try the probe-op of each distinct clean correct program; lock the one that
            # length-generalizes within the budget (only AVG/bisection will).
            for (o, a) in list(distinct.values()):
                probe = canon_body(a)
                if probe is not None and verify_body(probe, mult, add, vrng):
                    locked = probe
                    if verbose:
                        print(f"      *** DISCOVERED & LOCKED body: probe={probe} (it {it}, from {len(distinct)} distinct correct programs) ***")
                    break
            if distinct and not locked:
                items = list(distinct.values())
                picks = [random.choice(items) for _ in range(bs_im)]   # uniform over DISTINCT programs (entropy)
                for _ in range(grad_steps):
                    last_loss = imitate(model, opt, picks)
        else:
            targets, tw = [], []
            for w in range(1, wtrain_max + 1):                 # include w1 (tiny brackets need direct distill)
                for _ in range(3):
                    targets.append(body_trace(locked, w, mult, add, rng)); tw.append(float(w * w))
            for _ in range(grad_steps):
                last_loss = imitate(model, opt, random.choices(targets, weights=tw, k=bs_im))
        if it % log_every == 0 or it == 1:
            accs = {w: greedy_acc(model, w, mult, add, n=120, seed=it) for w in (1, 3, 6, 12)}
            astr = " ".join(f"w{w}:{accs[w]:.2f}" for w in accs)
            lk = locked if locked else "-"
            print(f"  it {it:4d}  locked[{lk}]  distinct {len(distinct):4d}  loss {last_loss:.4f}  [{astr}]")
            history.append((it, locked, accs))
    return model, {"first_hit": first_hit, "locked": locked}, history


# ----------------------------------------------------------------------------
# Reference policies (hand-coded, NOT used in training) -- the airtight "why bisection" contrast.
# ----------------------------------------------------------------------------
def reference_contrast(mult, add, widths=(1, 2, 3, 4, 6, 8, 12, 20), n=200, seed=1):
    print("\n  CONTRAST -- exact-isqrt accuracy of the two reference strategies WITHIN the step budget:")
    for kind, probe in (("BINARY (AVG probe)", "AVG"), ("LINEAR (NEXT probe)", "NEXT")):
        rep = {}
        for w in widths:
            rng = random.Random(seed + w); ok = 0; cap = cap_for(w, mult, add)
            for _ in range(n):
                x = make_problem(w, rng); got, halted = interpret(probe, x, cap)
                ok += (halted and got == math.isqrt(x))
            rep[w] = ok / n
        print(f"    {kind:20s}: " + "  ".join(f"w{w}:{rep[w]:.3f}" for w in widths))
    print("    => linear scan is exact but EXPLODES past the step budget by w=3 (O(sqrt n) probes),")
    print("       so only BINARY SEARCH survives the exact-length-gen-under-budget criterion.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
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

    for _ in range(2000):
        x = random.randint(1, 10 ** 6)
        g, h = interpret("AVG", x, 100000)
        assert h and g == math.isqrt(x), ("binary ref", x, g, math.isqrt(x))
    print("reference BINARY-SEARCH body computes isqrt exactly (sanity, n up to 1e6).")
    gl, hl = interpret("NEXT", 10 ** 8, cap_for(4, args.mult, args.add))
    print(f"linear on isqrt(1e8) within w4 budget(cap={cap_for(4, args.mult, args.add)}): halted={hl} (expect False).")

    print(f"\ndevice={DEVICE}  SELF-DISCOVERY of isqrt from OUTCOME ALONE (no traces)  cap={args.mult}*w+{args.add}")
    torch.manual_seed(args.seed)
    model = Controller(hidden=args.hidden)
    print(f"  params: {sum(p.numel() for p in model.parameters())}")
    model, info, _ = selfdiscover(model, iters=args.iters, M=args.M, mult=args.mult, add=args.add, seed=args.seed)
    print(f"\n  first exact-correct self-sample at iter: {info['first_hit']}")
    algo = {"AVG": "BINARY SEARCH (bisection)", "NEXT": "LINEAR SCAN", None: "NONE (not discovered)"}[info["locked"]]
    print(f"  DISCOVERED algorithm (locked body probe): {info['locked']}  =>  {algo}")

    print("\n  LENGTH-GEN of the discovered model (greedy; emits AND runs the program, within budget):")
    widths = (1, 2, 3, 4, 6, 8, 12, 16, 20, 25, 30)
    rep = {w: greedy_acc(model, w, args.mult, args.add, n=200, seed=100 + w) for w in widths}
    print("    " + "  ".join(f"w{w}:{rep[w]:.3f}" for w in widths))

    print("\n  example emitted programs (the discovered algorithm):")
    for x in (4, 81, 1000, 123456, 9999999999):
        prog, got, halted = emit_program(model, x, cap_for(len(str(x)), args.mult, args.add))
        print(f"    isqrt({x})={math.isqrt(x)} got {got} ok={got == math.isqrt(x)}: {' '.join(prog)}")

    reference_contrast(args.mult, args.add)
    if args.save and not args.smoke:
        torch.save(model.state_dict(), args.save)
        print(f"\nsaved {args.save}")
