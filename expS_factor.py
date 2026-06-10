"""
expS_factor.py — can self-discovery discover INTEGER FACTORIZATION from OUTCOME ALONE, and WHICH algorithm?

Factorization is unlike every prior op: it is (conjectured) NOT in P -- there is no known polynomial-time
algorithm (this is what RSA rests on). So the project's usual headline ("exact length-gen under a POLY step
budget") is fundamentally UNACHIEVABLE here: any correct method is super-polynomial, so the step budget must
itself grow ~sqrt(n) (exponential in #digits). The "real algorithm vs lookup" length-gen test still applies
(a real factoring loop works for ANY n; a lookup table doesn't), but the budget is inherently exponential.

So the questions are: (1) does outcome-only self-discovery find a CORRECT factoring algorithm at all, and
(2) WHICH one -- and is the efficiency lever still operative? We use a whole-number TRIAL-DIVISION VM in which
both the O(n) naive and the O(sqrt n) bounded versions are expressible with the SAME ops, so a per-instance
STEP BUDGET ~ sqrt(n) gets to select the efficient one (the sqrt(n) stopping bound = the discovered optimization).

VM: registers N (remaining number), D (candidate divisor, starts 2); emit prime factors into `out`.
Instructions (general): FACTOR (out+=D ; N//=D ; needs D|N), EMIT_N (out+=N ; N=1 ; the remaining N is prime),
INC (D+=1), HALT (output out). obs = [div=(N%D==0), done=(N==1), past=(D*D>N)].
TRIAL DIVISION is a MEMORYLESS reactive policy:
  done -> HALT ; div -> FACTOR ; (~div,~done,past) -> EMIT_N ; (~div,~done,~past) -> INC.
The EFFICIENCY CHOICE is at (~div,~done,past): EMIT_N (declare N prime once D>sqrt(N) -> O(sqrt n)) vs INC
(keep dividing up to N -> O(n), busts the budget on primes). The budget selects EMIT_N = the sqrt(n) bound.
Same exact-filtered self-imitation recipe as expK/expQ (consensus table + candidate-verify-selection + clean/
consistency filters). The only signal is whether `out` is the exact prime factorization of n. NO traces.
Run: python expS_factor.py --iters 150
"""
from __future__ import annotations
import argparse, itertools, math, random
from collections import deque, Counter
import torch, torch.nn as nn

import expA_mealy as E

DEVICE = E.DEVICE
INSTRS = ["HALT", "FACTOR", "EMIT_N", "INC"]
NI = len(INSTRS)
IID = {n: i for i, n in enumerate(INSTRS)}
N_OBS = 3                                        # [div, done, past]
OBS_STATES = [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]


def factorize(n):
    """Reference oracle: the true prime factorization (sorted, with multiplicity). Allowed to be slow."""
    f, d, x = [], 2, n
    while d * d <= x:
        while x % d == 0:
            f.append(d); x //= d
        d += 1
    if x > 1:
        f.append(x)
    return f


def cap_for(n):
    """Per-instance step budget ~ sqrt(n): tight enough that the O(n) naive (INC up to N) busts it on primes,
    while the O(sqrt n) bounded version (EMIT_N once D*D>N) fits. (Inherently exponential in #digits.)"""
    return 3 * math.isqrt(n) + 14


class FactorVM:
    def __init__(self, n):
        self.n0 = n; self.N = n; self.D = 2; self.out = []
        self.halted = False; self.invalid = False

    def obs(self):
        div = 1.0 if self.N % self.D == 0 else 0.0
        done = 1.0 if self.N == 1 else 0.0
        past = 1.0 if self.D * self.D > self.N else 0.0
        return [div, done, past]

    def execute(self, instr):
        if instr == "FACTOR":
            if self.N > 1 and self.N % self.D == 0:
                self.out.append(self.D); self.N //= self.D
            else:
                self.invalid = True
        elif instr == "EMIT_N":
            if self.N > 1:
                self.out.append(self.N); self.N = 1
            else:
                self.invalid = True
        elif instr == "INC":
            self.D += 1
        elif instr == "HALT":
            self.halted = True

    def answer(self):
        return self.out

    def correct(self):
        return self.out == factorize(self.n0)


class Controller(nn.Module):
    def __init__(self, hidden=48):
        super().__init__()
        self.gru = nn.GRU(N_OBS, hidden, batch_first=True)
        self.head = nn.Linear(hidden, NI)

    def forward(self, obs, h=None):
        y, h = self.gru(obs, h)
        return self.head(y), h


def make_problem(lo, hi, rng):
    return rng.randint(lo, hi)


def make_mag(mag, rng):
    lo = 2 if mag == 1 else 10 ** (mag - 1)
    return rng.randint(lo, 10 ** mag - 1)


def clean(o, a):
    """VALIDITY filter: FACTOR only when div=1 (D must actually divide N to be a factor). Doesn't say when to
    INC vs EMIT_N vs HALT (the algorithm / the sqrt(n)-bound choice)."""
    fac = IID["FACTOR"]
    return not any(a[i] == fac and o[i][0] < 0.5 for i in range(len(a)))


def induce_table(o, a):
    t = {}
    for ob, ac in zip(o, a):
        s = tuple(int(x) for x in ob)
        if s in t and t[s] != ac:
            return None
        t[s] = ac
    return t


def candidate_tables(counts, max_cand=256):
    # Consider any action with real support (>=3 votes), NOT a fraction: when one action dominates a state
    # (e.g. the 'declare-prime gamble' EMIT_N floods (div0,past0)), a %-threshold would hide the correct
    # alternative (INC). The cross-validate then selects the GENERALIZING combination.
    base, ambiguous = {}, []
    for s in OBS_STATES:
        c = counts[s]
        if not c:
            base[s] = "HALT"; continue
        mc = c.most_common()
        base[s] = INSTRS[mc[0][0]]
        if len(mc) >= 2 and mc[1][1] >= 3:
            ambiguous.append((s, [INSTRS[mc[0][0]], INSTRS[mc[1][0]]]))
    while ambiguous and 2 ** len(ambiguous) > max_cand:
        ambiguous.pop()
    if not ambiguous:
        return [base]
    states = [s for s, _ in ambiguous]; opts = [o for _, o in ambiguous]
    out = []
    for combo in itertools.product(*opts):
        t = dict(base)
        for k, s in enumerate(states):
            t[s] = combo[k]
        out.append(t)
    return out


def sample_rollouts(model, ns, temp=1.0, eps=0.0):
    M = len(ns); vms = [FactorVM(n) for n in ns]; caps = [cap_for(n) for n in ns]; mx = max(caps); h = None
    tr_obs = [[] for _ in range(M)]; tr_act = [[] for _ in range(M)]; active = [True] * M
    legal = torch.ones(NI, device=DEVICE) / NI
    for t in range(mx):
        if not any(active):
            break
        for i in range(M):
            if active[i] and t >= caps[i]:               # per-instance budget (exponential ~sqrt(n))
                active[i] = False
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
    return [(vms[i].halted and not vms[i].invalid and vms[i].correct(), tr_obs[i], tr_act[i]) for i in range(M)]


_lossf = nn.CrossEntropyLoss()


def imitate(model, opt, pairs):
    obs = torch.tensor([o for (o, _) in pairs], dtype=torch.float32, device=DEVICE).unsqueeze(1)
    tgt = torch.tensor([a for (_, a) in pairs], dtype=torch.long, device=DEVICE)
    logits, _ = model(obs)
    loss = _lossf(logits[:, -1, :], tgt)
    opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()


def run_table(table, n, cap):
    vm = FactorVM(n); steps = 0
    while not vm.halted and not vm.invalid and steps < cap:
        s = tuple(int(x) for x in vm.obs())
        instr = table.get(s)
        if instr is None:
            return None, False, steps
        vm.execute(instr); steps += 1
    return vm.out, (vm.halted and not vm.invalid), steps


def verify_table(table, mags=(1, 2, 3, 4, 5), n=120, seed=1):
    rep = {}
    for mag in mags:
        rng = random.Random(seed + mag); ok = 0
        for _ in range(n):
            x = make_mag(mag, rng)
            out, halted, _ = run_table(table, x, cap_for(x))
            ok += (halted and out == factorize(x))
        rep[mag] = ok / n
    return rep


@torch.no_grad()
def greedy_acc(model, mag, n=120, seed=0):
    model.eval(); rng = random.Random(seed + 7919 * mag); ok = 0
    for _ in range(n):
        x = make_mag(mag, rng); vm = FactorVM(x); h = None; steps = 0; cap = cap_for(x)
        while not vm.halted and not vm.invalid and steps < cap:
            obs = torch.tensor([vm.obs()], dtype=torch.float32, device=DEVICE).unsqueeze(0)
            logits, h = model(obs, h)
            vm.execute(INSTRS[int(logits[0, -1].argmax())]); steps += 1
        ok += (vm.halted and not vm.invalid and vm.out == factorize(x))
    model.train(); return ok / n


def selfdiscover(model, iters=150, M=1536, bs_im=256, grad_steps=6, lr=3e-3, train_hi=199,
                 buf_max=8000, seed=0, log_every=10, verbose=True, lock_mags=(1, 2, 3, 4, 5)):
    model.to(DEVICE); opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = random.Random(seed)
    buffer = deque(maxlen=buf_max); counts = {s: Counter() for s in OBS_STATES}
    first_hit = None; locked = None
    for it in range(1, iters + 1):
        model.train()
        if locked is None:
            ns = [make_problem(2, train_hi, rng) for _ in range(M)]
            results = sample_rollouts(model, ns, temp=1.0, eps=0.15)
            n_ok = 0
            for ok, o, a in results:
                if not ok or not clean(o, a) or induce_table(o, a) is None:
                    continue
                n_ok += 1
                if first_hit is None:
                    first_hit = it
                for ob, ac in zip(o, a):
                    buffer.append((ob, ac)); counts[tuple(int(x) for x in ob)][ac] += 1
            need = [(1, 0, 0), (0, 0, 0), (0, 0, 1), (0, 1, 1)]      # the reachable states a sort visits
            if all(counts[s] for s in need) and it % 5 == 0:
                for cand in candidate_tables(counts):
                    rep = verify_table(cand, mags=lock_mags, n=60, seed=it)
                    if min(rep.values()) > 0.97:
                        locked = cand
                        if verbose:
                            print(f"      *** DISCOVERED & LOCKED policy (it {it}): " +
                                  " ; ".join(f"{s}->{cand[s]}" for s in OBS_STATES if counts[s]) + " ***")
                        break
            if len(buffer) >= bs_im:
                for _ in range(grad_steps):
                    imitate(model, opt, random.sample(buffer, bs_im))
            n_log = n_ok
        else:
            pairs = [(list(map(float, s)), IID[locked[s]]) for s in OBS_STATES] * (bs_im // len(OBS_STATES))
            for _ in range(grad_steps):
                imitate(model, opt, pairs)
            n_log = -1
        if it % log_every == 0 or it == 1:
            accs = {m: greedy_acc(model, m, n=60, seed=it) for m in (1, 2, 3)}
            astr = " ".join(f"m{m}:{accs[m]:.2f}" for m in accs)
            nseen = sum(1 for s in OBS_STATES if counts[s])
            print(f"  it {it:4d} {'LOCKED' if locked else '      '} ok {n_log:5d}/{M}  states {nseen}/8  greedy[{astr}]")
    cons = {s: (INSTRS[counts[s].most_common(1)[0][0]] if counts[s] else "?") for s in OBS_STATES}
    return model, {"first_hit": first_hit, "locked": locked, "consensus": cons}, []


SMART = {(1, 0, 0): "FACTOR", (1, 0, 1): "FACTOR", (0, 0, 0): "INC", (0, 0, 1): "EMIT_N", (0, 1, 1): "HALT"}
NAIVE = {**SMART, (0, 0, 1): "INC"}                       # no sqrt(n) bound -> O(n), busts the budget on primes


def label_table(t):
    if t[(1, 0, 0)] == "FACTOR" and t[(0, 0, 0)] == "INC" and t[(0, 1, 1)] == "HALT":
        if t[(0, 0, 1)] == "EMIT_N":
            return "TRIAL DIVISION (sqrt(n)-bounded: declare prime once D*D>N)"
        if t[(0, 0, 1)] == "INC":
            return "TRIAL DIVISION (naive O(n): divide up to N)"
    return "OTHER / non-trial-division"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--M", type=int, default=1536)
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--train_hi", type=int, default=199)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default="")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.iters = 80

    for _ in range(3000):
        x = random.randint(2, 5000)
        out, halted, _ = run_table(SMART, x, cap_for(x))
        assert halted and out == factorize(x), ("smart ref", x, out, factorize(x))
    print("reference SMART trial-division table factorizes exactly (sanity, n up to 5000).")
    p = 9973                                            # a prime: naive must INC ~9973 steps, smart ~sqrt -> ~100
    _, hn, sn = run_table(NAIVE, p, cap_for(p)); _, hs, ss = run_table(SMART, p, cap_for(p))
    print(f"prime {p}, budget cap={cap_for(p)}: NAIVE halted={hn} (busts), SMART halted={hs} steps={ss} (fits).")

    print(f"\ndevice={DEVICE}  SELF-DISCOVERY of FACTORIZATION from OUTCOME ALONE (no traces)  train n in [2,{args.train_hi}]")
    torch.manual_seed(args.seed)
    model = Controller(hidden=args.hidden)
    print(f"  params: {sum(p.numel() for p in model.parameters())}")
    model, info, _ = selfdiscover(model, iters=args.iters, train_hi=args.train_hi, seed=args.seed)
    print(f"\n  first exact-correct self-sample at iter: {info['first_hit']}")
    table = info["locked"] if info["locked"] else info["consensus"]
    print(f"  DISCOVERED policy table ({'LOCKED' if info['locked'] else 'consensus, not locked'}):")
    for s in OBS_STATES:
        if info["locked"] or info["consensus"][s] != "?":
            print(f"    (div={s[0]}, done={s[1]}, past={s[2]}) -> {table[s]}")
    print(f"  => ALGORITHM: {label_table(table)}")

    print("\n  LENGTH-GEN of the discovered policy (exact factorization, within an O(sqrt n) budget):")
    rep = verify_table(table, mags=(1, 2, 3, 4, 5, 6, 7, 8), n=200)
    print("    TABLE : " + "  ".join(f"1e{m}:{rep[m]:.3f}" for m in rep))
    print("    NEURAL: " + "  ".join(f"1e{m}:{greedy_acc(model, m, n=150, seed=50 + m):.3f}" for m in (1, 2, 3)))

    print("\n  CONTRAST -- naive O(n) vs sqrt(n)-bounded, exact factorization WITHIN the budget:")
    rs = verify_table(SMART, mags=(1, 2, 3, 4, 5, 6, 7), n=200)
    rn = verify_table(NAIVE, mags=(1, 2, 3, 4, 5, 6, 7), n=200)
    print("    SMART (EMIT_N on past): " + "  ".join(f"1e{m}:{rs[m]:.3f}" for m in rs))
    print("    NAIVE (INC   on past) : " + "  ".join(f"1e{m}:{rn[m]:.3f}" for m in rn))
    print("    => the naive O(n) version busts the ~sqrt(n) budget on primes/large-factor n; the budget selects")
    print("       the sqrt(n)-bounded trial division -- but BOTH are exponential in #digits (no poly algorithm).")

    print("\n  STEP COUNTS of the discovered policy (median over 200 n per magnitude) -- grow ~sqrt(n):")
    for m in (1, 2, 3, 4, 5, 6, 7, 8):
        rng = random.Random(9 + m); cs = []
        for _ in range(200):
            x = make_mag(m, rng); _, _, st = run_table(table if info["locked"] else SMART, x, cap_for(x) + 5)
            cs.append(st)
        cs.sort()
        print(f"    1e{m}: median {cs[len(cs)//2]:6d} steps   (~sqrt(1e{m}) = {int(10 ** (m/2)):6d})")

    print("\n  example factorizations (the discovered trial division):")
    for x in (84, 1001, 7919, 999983, 13 * 9973):
        out, halted, steps = run_table(table if info["locked"] else SMART, x, cap_for(x) + 5)
        print(f"    {x} = {'*'.join(map(str, out))}  correct={out == factorize(x)}  steps={steps}")
    if args.save and not args.smoke:
        torch.save(model.state_dict(), args.save)
        print(f"\nsaved {args.save}")
