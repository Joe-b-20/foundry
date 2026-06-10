"""
expK_gcd.py — can the self-discovery model discover GCD from OUTCOME ALONE, and WHICH algorithm?

GCD is not digit-serial, so it uses a NEW whole-number register VM. Registers A,B; instructions
{MOD: A=A%B, SUB: A=A-B (invalid if A<B), SWAP, HALT (output A)}; observation = [done=(B==0), A>=B].
With this obs, BOTH classic algorithms are tiny MEMORYLESS reactive policies differing in ONE entry:
    done -> HALT ; (not done, A<B) -> SWAP ; (not done, A>=B) -> {MOD = Euclidean | SUB = subtractive}.
Both compute gcd EXACTLY on small inputs; the discriminator is EFFICIENCY UNDER A STEP BUDGET (the
project's exact-eval lever): Euclidean is O(log) steps; subtractive explodes on high-ratio pairs
(gcd(N,1) = N subtractions) and busts the cap. So exact-filtered self-imitation across sizes should
SELECT Euclidean. The "discovered algorithm" = the greedy obs->action policy table (read it off).

Trained ONLY on outcomes (exact gcd of the final answer); no algorithm/trace ever shown.
Run: python expK_gcd.py --iters 120
"""
from __future__ import annotations
import argparse, math, random
from collections import deque, Counter
import torch, torch.nn as nn

import expA_mealy as E

DEVICE = E.DEVICE
INSTRS = ["HALT", "MOD", "SUB", "SWAP"]
NI = len(INSTRS)
IID = {n: i for i, n in enumerate(INSTRS)}
N_OBS = 2                                   # [done=(B==0), A>=B]
OBS_STATES = [(0, 0), (0, 1), (1, 1)]       # reachable obs (done=1 => A>=B always true)
ALLOWED = set(INSTRS)                        # restrict the instruction set (e.g. drop MOD) for contrasts


def mask_vec():
    return torch.tensor([0.0 if INSTRS[i] in ALLOWED else -1e9 for i in range(NI)], device=DEVICE)


class GCDVM:
    def __init__(self, a, b):
        self.A, self.B = a, b
        self.halted = False
        self.invalid = False
    def obs(self):
        return [1.0 if self.B == 0 else 0.0, 1.0 if self.A >= self.B else 0.0]
    def execute(self, instr):
        if instr == "MOD":
            if self.B == 0:
                self.invalid = True
            else:
                self.A = self.A % self.B
        elif instr == "SUB":
            if self.A < self.B:
                self.invalid = True          # keep registers non-negative: SUB needs A>=B
            else:
                self.A = self.A - self.B
        elif instr == "SWAP":
            self.A, self.B = self.B, self.A
        elif instr == "HALT":
            self.halted = True
    def answer(self):
        return self.A


class Controller(nn.Module):
    def __init__(self, hidden=48):
        super().__init__()
        self.gru = nn.GRU(N_OBS, hidden, batch_first=True)
        self.head = nn.Linear(hidden, NI)
    def forward(self, obs, h=None):
        y, h = self.gru(obs, h)
        return self.head(y), h


def make_problem(wmax, rng):
    """Sample a,b with INDEPENDENT widths in 1..wmax, so high-ratio pairs (big a, small b) are common
    -- those are exactly where subtractive GCD explodes past the step cap."""
    wa = rng.randint(1, wmax); wb = rng.randint(1, wmax)
    a = rng.randint(0, 10 ** wa - 1); b = rng.randint(0, 10 ** wb - 1)
    if a == 0 and b == 0:
        a = 1
    return a, b


# ----------------------------------------------------------------------------
# Batched lockstep sampling of M rollouts.
# ----------------------------------------------------------------------------
def sample_rollouts(model, probs, cap, temp=1.0, eps=0.0):
    M = len(probs)
    vms = [GCDVM(a, b) for (a, b) in probs]
    h = None
    tr_obs = [[] for _ in range(M)]
    tr_act = [[] for _ in range(M)]
    active = [True] * M
    legal = (mask_vec() > -1e8).float()
    legal = legal / legal.sum()
    for _ in range(cap):
        if not any(active):
            break
        obs_rows = [vm.obs() for vm in vms]
        obs = torch.tensor(obs_rows, dtype=torch.float32, device=DEVICE).unsqueeze(1)
        logits, h = model(obs, h)
        probs_t = torch.softmax((logits[:, -1, :] + mask_vec()) / temp, dim=-1)
        if eps > 0:
            probs_t = (1 - eps) * probs_t + eps * legal
        a = torch.multinomial(probs_t, 1).squeeze(1).tolist()
        for i in range(M):
            if not active[i]:
                continue
            tr_obs[i].append(obs_rows[i]); tr_act[i].append(a[i])
            vms[i].execute(INSTRS[a[i]])
            if vms[i].halted or vms[i].invalid:
                active[i] = False
    out = []
    for i in range(M):
        a, b = probs[i]
        ok = vms[i].halted and (not vms[i].invalid) and vms[i].answer() == math.gcd(a, b)
        out.append((ok, tr_obs[i], tr_act[i]))
    return out


_lossf = nn.CrossEntropyLoss()


def imitate(model, opt, pairs):
    obs = torch.tensor([o for (o, _) in pairs], dtype=torch.float32, device=DEVICE).unsqueeze(1)
    tgt = torch.tensor([a for (_, a) in pairs], dtype=torch.long, device=DEVICE)
    logits, _ = model(obs)
    loss = _lossf(logits[:, -1, :], tgt)
    opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()


@torch.no_grad()
def greedy_policy_table(model):
    """Read the greedy action the model takes from each reachable obs state with a FRESH hidden state
    (the GCD policy is memoryless if discovery succeeded). Returns {obs_state: instr_name}."""
    model.eval()
    table = {}
    for s in OBS_STATES:
        obs = torch.tensor([list(map(float, s))], dtype=torch.float32, device=DEVICE).unsqueeze(0)
        logits, _ = model(obs)
        table[s] = INSTRS[int((logits[0, -1] + mask_vec()).argmax())]
    model.train()
    return table


@torch.no_grad()
def greedy_acc(model, wmax, cap, n=300, seed=0):
    model.eval()
    rng = random.Random(seed)
    ok = 0
    for _ in range(n):
        a, b = make_problem(wmax, rng)
        vm = GCDVM(a, b); h = None; steps = 0
        while not vm.halted and not vm.invalid and steps < cap:
            obs = torch.tensor([vm.obs()], dtype=torch.float32, device=DEVICE).unsqueeze(0)
            logits, h = model(obs, h)
            vm.execute(INSTRS[int((logits[0, -1] + mask_vec()).argmax())]); steps += 1
        ok += (vm.halted and not vm.invalid and vm.answer() == math.gcd(a, b))
    model.train()
    return ok / n


def run_table(table, a, b, cap):
    """Run a memoryless obs->action policy table; return (answer, halted, steps)."""
    vm = GCDVM(a, b); steps = 0
    while not vm.halted and not vm.invalid and steps < cap:
        s = tuple(int(x) for x in vm.obs())
        instr = table.get(s)
        if instr is None:
            return None, False, steps
        vm.execute(instr); steps += 1
    return vm.answer(), (vm.halted and not vm.invalid), steps


def verify_table(table, cap, widths=(1, 2, 4, 8, 12, 20), n=300, seed=1):
    """Run the extracted policy across SIZES within the step cap; report exact-gcd accuracy per width.
    A subtractive table passes small widths but FAILS large (step explosion); Euclidean passes all."""
    rep = {}
    for w in widths:
        rng = random.Random(seed + w); ok = 0
        for _ in range(n):
            a, b = make_problem(w, rng)
            got, halted, _ = run_table(table, a, b, cap)
            ok += (halted and got == math.gcd(a, b))
        rep[w] = ok / n
    return rep


def selfdiscover(model, iters=120, M=1536, bs_im=256, grad_steps=6, lr=3e-3,
                 wmax=5, cap=150, seed=0, log_every=10, verbose=True,
                 lock_widths=(1, 2, 4, 8, 12, 20)):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = random.Random(seed)
    buffer = deque(maxlen=6000)              # correct (obs, action) pairs (recency-bounded)
    first_hit = None
    locked = None                            # the DISCOVERED policy table once it verifies across sizes
    for it in range(1, iters + 1):
        model.train()
        if locked is None:
            probs = [make_problem(wmax, rng) for _ in range(M)]
            eps = 0.25
            results = sample_rollouts(model, probs, cap, temp=1.0, eps=eps)
            n_ok = 0
            for ok, o, a in results:
                if not ok:
                    continue
                n_ok += 1
                if first_hit is None:
                    first_hit = it
                for ob, ac in zip(o, a):
                    buffer.append((ob, ac))
            # CANDIDATE CHECK: does the current greedy table VERIFY across sizes (incl. large) within the
            # cap? Only an efficient algorithm (Euclidean) passes large widths; subtractive busts the cap.
            tbl = greedy_policy_table(model)
            rep = verify_table(tbl, cap=300, widths=lock_widths, n=120, seed=it)
            if min(rep.values()) > 0.97:
                locked = tbl
                if verbose:
                    tstr = " ; ".join(f"{s}->{tbl[s]}" for s in OBS_STATES)
                    print(f"      *** DISCOVERED & LOCKED policy (it {it}): {tstr} ***")
            if len(buffer) >= bs_im:
                for _ in range(grad_steps):
                    imitate(model, opt, random.sample(buffer, bs_im))
            n_log = n_ok
        else:
            # DISTILL the locked table: imitate ONLY its (obs->action) pairs -> freeze the policy.
            pairs = [(list(map(float, s)), IID[locked[s]]) for s in OBS_STATES] * (bs_im // len(OBS_STATES))
            for _ in range(grad_steps):
                imitate(model, opt, pairs)
            n_log = -1
        if it % log_every == 0 or it == 1:
            tbl = greedy_policy_table(model)
            acc = greedy_acc(model, wmax, cap)
            tstr = " ".join(f"{s}->{tbl[s]}" for s in OBS_STATES)
            lk = "LOCKED" if locked else "      "
            print(f"  it {it:4d} {lk} ok {n_log:5d}/{M}  buf {len(buffer):5d}  greedy_acc(w{wmax}) {acc:.3f}  [{tstr}]")
    return model, first_hit, locked


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--wmax", type=int, default=5, help="max #digits per operand during discovery")
    ap.add_argument("--cap", type=int, default=150, help="VM step budget (subtractive busts it on high-ratio pairs)")
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default="")
    ap.add_argument("--no_mod", action="store_true", help="contrast: drop MOD so only subtractive is expressible")
    ap.add_argument("--lock_max_w", type=int, default=20, help="max width in the lock-verify (small => subtractive can lock)")
    args = ap.parse_args()
    if args.no_mod:
        ALLOWED = {"HALT", "SUB", "SWAP"}        # remove the modulo primitive -> only subtractive GCD exists
    lock_widths = tuple(w for w in (1, 2, 4, 8, 12, 20) if w <= args.lock_max_w)

    # sanity: reference Euclidean & subtractive tables compute gcd; subtractive busts the cap on big ratios
    eucl = {(0, 0): "SWAP", (0, 1): "MOD", (1, 1): "HALT"}
    subt = {(0, 0): "SWAP", (0, 1): "SUB", (1, 1): "HALT"}
    for _ in range(500):
        a = random.randint(0, 9999); b = random.randint(0, 9999)
        if a == 0 and b == 0:
            continue
        g, h, _ = run_table(eucl, a, b, 100000)
        assert h and g == math.gcd(a, b), ("eucl", a, b, g)
    print("reference EUCLIDEAN table computes gcd exactly (sanity).")
    gg, hh, st = run_table(subt, 10 ** 6, 1, args.cap)   # subtractive on a high-ratio pair
    print(f"subtractive on gcd(10^6,1) within cap={args.cap}: halted={hh} (expect False -- it explodes).")

    print(f"\ndevice={DEVICE}  SELF-DISCOVERY of GCD from OUTCOME ALONE (no traces)  wmax={args.wmax} cap={args.cap}")
    torch.manual_seed(args.seed)
    model = Controller(hidden=args.hidden)
    print(f"  params: {sum(p.numel() for p in model.parameters())}")
    model, first_hit, locked = selfdiscover(model, iters=args.iters, wmax=args.wmax, cap=args.cap,
                                            seed=args.seed, lock_widths=lock_widths)

    print(f"\n  first exact-correct self-sample at iter: {first_hit}")
    table = greedy_policy_table(model)
    print(f"  DISCOVERED greedy policy table: " + " ; ".join(f"{s}->{table[s]}" for s in OBS_STATES))
    choice = table[(0, 1)]
    algo = {"MOD": "EUCLIDEAN (gcd(a,b)=gcd(b, a mod b))",
            "SUB": "SUBTRACTIVE (gcd(a,b)=gcd(a-b,b))"}.get(choice, f"OTHER ({choice})")
    print(f"  => for state (not-done, A>=B) it chose {choice}  =>  {algo}")
    print("\n  LENGTH-GEN of the discovered policy (exact gcd across sizes, step cap=300):")
    rep = verify_table(table, cap=300, widths=(1, 2, 4, 8, 12, 20, 30))
    print("    " + "  ".join(f"w{w}:{rep[w]:.3f}" for w in rep))
    # also the neural model run greedily (not just the extracted table)
    print("  LENGTH-GEN of the NEURAL model run greedily (emits+runs):")
    for w in (1, 4, 12, 20):
        print(f"    w{w}: {greedy_acc(model, w, 300, n=300, seed=100+w):.3f}", end="")
    print()

    # WHY this algorithm and not the other: compare the two reference policy tables across sizes.
    print("\n  CONTRAST -- exact-gcd accuracy of the two reference algorithms across sizes (cap=300):")
    re = verify_table(eucl, cap=300, widths=(1, 2, 4, 8, 12, 20, 30))
    rs = verify_table(subt, cap=300, widths=(1, 2, 4, 8, 12, 20, 30))
    print("    EUCLIDEAN  ((A>=B)->MOD): " + "  ".join(f"w{w}:{re[w]:.3f}" for w in re))
    print("    SUBTRACTIVE((A>=B)->SUB): " + "  ".join(f"w{w}:{rs[w]:.3f}" for w in rs))
    print("    => subtractive computes gcd but EXPLODES past the step budget on large/high-ratio pairs,")
    print("       so only Euclidean survives the exact-length-gen criterion -> that is what gets discovered.")
    if args.save:
        torch.save(model.state_dict(), args.save)
        print(f"\nsaved {args.save}")
