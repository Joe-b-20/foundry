"""
expR_selection.py — SORTING from OUTCOME ALONE with MIN-SELECT primitives: does it discover SELECTION SORT?

Companion to expQ_sort.py. expQ gave an adjacent-swap VM and outcome-discovery found BUBBLE SORT. The
project's law is that WHICH algorithm is discovered is PRIMITIVE/REPRESENTATION-dependent (cf. isqrt:
square+compare -> binary search; divide+average -> Newton). So change the primitives to min-selection and
see if the SAME recipe discovers a structurally different sort.

Whole-LIST VM: array A, pointers i (slot to fill), j (scan over the unsorted suffix), m (index of the
running min). Controller sees ONLY flags (never values) -> a real comparison sort, length-generalizing.
Instructions (general min-select ops, none sort-specific):
  SETM   m = j                              (mark current scan position as the running-min candidate)
  INCJ   j += 1                             (advance the scan; invalid at the suffix end)
  PLACE  swap A[i],A[m] ; i+=1 ; j=i+1 ; m=i (place the found min at the boundary, advance, restart scan)
  HALT   output A
obs = [lt = A[j]<A[m], j_end = (j==n-1), done = (i>=n-1)].  SELECTION SORT is a MEMORYLESS reactive policy:
  done -> HALT ; (lt,~je) -> SETM ; (lt,je) -> SETM ; (~lt,~je) -> INCJ ; (~lt,je) -> PLACE.
Same exact-filtered self-imitation + clean/consistency filters + candidate-verify-selection as expQ; the only
signal is whether the output list is non-decreasing. Run: python expR_selection.py --iters 120
"""
from __future__ import annotations
import argparse, itertools, random
from collections import deque, Counter
import torch, torch.nn as nn

import expA_mealy as E

DEVICE = E.DEVICE
INSTRS = ["HALT", "SETM", "INCJ", "PLACE"]
NI = len(INSTRS)
IID = {n: i for i, n in enumerate(INSTRS)}
N_OBS = 3                                        # [lt, j_end, done]
OBS_STATES = [(l, j, d) for l in (0, 1) for j in (0, 1) for d in (0, 1)]


def is_sorted(a):
    return all(a[i] <= a[i + 1] for i in range(len(a) - 1))


class SelectVM:
    def __init__(self, arr):
        self.A = list(arr); self.n = len(arr)
        self.i = 0; self.j = 1; self.m = 0
        self.halted = False; self.invalid = False

    def obs(self):
        lt = 1.0 if (self.j < self.n and self.A[self.j] < self.A[self.m]) else 0.0
        je = 1.0 if self.j >= self.n - 1 else 0.0
        done = 1.0 if self.i >= self.n - 1 else 0.0
        return [lt, je, done]

    def execute(self, instr):
        if instr == "SETM":
            self.m = self.j
        elif instr == "INCJ":
            if self.j < self.n - 1:
                self.j += 1
            else:
                self.invalid = True
        elif instr == "PLACE":
            if self.i < self.n - 1:
                self.A[self.i], self.A[self.m] = self.A[self.m], self.A[self.i]
                self.i += 1; self.j = self.i + 1; self.m = self.i
            else:
                self.invalid = True
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


def cap_for(L):
    return 3 * L * L + 12


def make_problem(L, rng):
    while True:
        a = [rng.randint(0, 999) for _ in range(L)]
        if L < 2 or not is_sorted(a):
            return a


def clean(o, a):
    """VALIDITY filter: SETM only when lt=1 (don't mark a non-smaller element as the min); PLACE only at the
    scan end je=1 (don't place a partial-min). Drops lucky/noisy rollouts; doesn't specify the algorithm."""
    setm, place = IID["SETM"], IID["PLACE"]
    for ob, ac in zip(o, a):
        if (ac == setm and ob[0] < 0.5) or (ac == place and ob[1] < 0.5):
            return False
    return True


def induce_table(o, a):
    t = {}
    for ob, ac in zip(o, a):
        s = tuple(int(x) for x in ob)
        if s in t and t[s] != ac:
            return None
        t[s] = ac
    return t


def candidate_tables(counts, max_cand=64):
    base, ambiguous = {}, []
    for s in OBS_STATES:
        c = counts[s]
        if not c:
            base[s] = "HALT"; continue
        mc = c.most_common(); tot = sum(c.values())
        base[s] = INSTRS[mc[0][0]]
        if len(mc) >= 2 and mc[1][1] >= max(3, 0.05 * tot):
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


def sample_rollouts(model, arrs, step_cap, temp=1.0, eps=0.0):
    M = len(arrs); vms = [SelectVM(a) for a in arrs]; h = None
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
    return [(vms[i].halted and not vms[i].invalid and is_sorted(vms[i].A), tr_obs[i], tr_act[i]) for i in range(M)]


_lossf = nn.CrossEntropyLoss()


def imitate(model, opt, pairs):
    obs = torch.tensor([o for (o, _) in pairs], dtype=torch.float32, device=DEVICE).unsqueeze(1)
    tgt = torch.tensor([a for (_, a) in pairs], dtype=torch.long, device=DEVICE)
    logits, _ = model(obs)
    loss = _lossf(logits[:, -1, :], tgt)
    opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()


def run_table(table, arr, cap):
    vm = SelectVM(arr); steps = 0
    while not vm.halted and not vm.invalid and steps < cap:
        s = tuple(int(x) for x in vm.obs())
        instr = table.get(s)
        if instr is None:
            return None, False, steps
        vm.execute(instr); steps += 1
    return vm.A, (vm.halted and not vm.invalid), steps


def verify_table(table, lengths=(2, 3, 5, 8, 12), n=80, seed=1):
    rep = {}
    for L in lengths:
        rng = random.Random(seed + L); ok = 0
        for _ in range(n):
            a = make_problem(L, rng)
            out, halted, _ = run_table(table, a, cap_for(L))
            ok += (halted and is_sorted(out))
        rep[L] = ok / n
    return rep


@torch.no_grad()
def greedy_acc(model, L, n=120, seed=0):
    model.eval(); rng = random.Random(seed + 7919 * L); ok = 0; cap = cap_for(L)
    for _ in range(n):
        a = make_problem(L, rng); vm = SelectVM(a); h = None; steps = 0
        while not vm.halted and not vm.invalid and steps < cap:
            obs = torch.tensor([vm.obs()], dtype=torch.float32, device=DEVICE).unsqueeze(0)
            logits, h = model(obs, h)
            vm.execute(INSTRS[int(logits[0, -1].argmax())]); steps += 1
        ok += (vm.halted and not vm.invalid and is_sorted(vm.A))
    model.train(); return ok / n


def selfdiscover(model, iters=120, M=1536, bs_im=256, grad_steps=6, lr=3e-3, wmax=5,
                 buf_max=8000, seed=0, log_every=10, verbose=True, lock_lengths=(2, 3, 5, 8, 12)):
    model.to(DEVICE); opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = random.Random(seed)
    buffer = deque(maxlen=buf_max); counts = {s: Counter() for s in OBS_STATES}
    first_hit = None; locked = None; history = []
    for it in range(1, iters + 1):
        model.train()
        if locked is None:
            arrs = [make_problem(rng.choice((2, 3, 3, 3, 4, 4, 5)), rng) for _ in range(M)]
            results = sample_rollouts(model, arrs, cap_for(wmax), temp=1.0, eps=0.15)
            n_ok = 0
            for ok, o, a in results:
                if not ok or not clean(o, a) or induce_table(o, a) is None:
                    continue
                n_ok += 1
                if first_hit is None:
                    first_hit = it
                for ob, ac in zip(o, a):
                    buffer.append((ob, ac)); counts[tuple(int(x) for x in ob)][ac] += 1
            if all(counts[s] for s in OBS_STATES if s[2] == 0) and it % 5 == 0:   # all non-done states seen
                for cand in candidate_tables(counts):
                    rep = verify_table(cand, lengths=lock_lengths, n=40, seed=it)
                    if min(rep.values()) > 0.97:
                        locked = cand
                        if verbose:
                            print(f"      *** DISCOVERED & LOCKED policy (it {it}): " +
                                  " ; ".join(f"{s}->{cand[s]}" for s in OBS_STATES) + " ***")
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
            accs = {L: greedy_acc(model, L, n=60, seed=it) for L in (3, 8, 16)}
            astr = " ".join(f"L{L}:{accs[L]:.2f}" for L in accs)
            nseen = sum(1 for s in OBS_STATES if counts[s])
            print(f"  it {it:4d} {'LOCKED' if locked else '      '} ok {n_log:5d}/{M}  states {nseen}/8  greedy[{astr}]")
    cons = {s: (INSTRS[counts[s].most_common(1)[0][0]] if counts[s] else "?") for s in OBS_STATES}
    return model, {"first_hit": first_hit, "locked": locked, "consensus": cons}, history


SELECTION = {(0, 0, 0): "INCJ", (0, 1, 0): "PLACE", (1, 0, 0): "SETM", (1, 1, 0): "SETM",
             (0, 0, 1): "HALT", (0, 1, 1): "HALT", (1, 0, 1): "HALT", (1, 1, 1): "HALT"}


def label_table(t):
    core = (t[(1, 0, 0)] == "SETM" and t[(0, 0, 0)] == "INCJ" and t[(0, 1, 0)] == "PLACE" and t[(0, 1, 1)] == "HALT")
    if all(t[s] == SELECTION[s] for s in [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0), (0, 1, 1)]):
        return "SELECTION SORT (exact canonical table)"
    return "SELECTION SORT (core mark/scan/place/halt rule)" if core else "OTHER / non-selection"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--M", type=int, default=1536)
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--wmax", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default="")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.iters = 80

    for L in (2, 3, 5, 9, 16):
        rng = random.Random(L)
        for _ in range(200):
            a = make_problem(L, rng)
            out, halted, _ = run_table(SELECTION, a, cap_for(L))
            assert halted and is_sorted(out), ("selection ref", a, out)
    print("reference SELECTION table sorts exactly (sanity, L up to 16).")

    print(f"\ndevice={DEVICE}  SELF-DISCOVERY of SORTING (min-select VM) from OUTCOME ALONE  wmax={args.wmax}")
    torch.manual_seed(args.seed)
    model = Controller(hidden=args.hidden)
    print(f"  params: {sum(p.numel() for p in model.parameters())}")
    model, info, _ = selfdiscover(model, iters=args.iters, M=args.M, wmax=args.wmax, seed=args.seed)
    print(f"\n  first exact-correct self-sample at iter: {info['first_hit']}")
    table = info["locked"] if info["locked"] else info["consensus"]
    print(f"  DISCOVERED policy table ({'LOCKED' if info['locked'] else 'consensus, not locked'}):")
    for s in OBS_STATES:
        print(f"    (lt={s[0]}, j_end={s[1]}, done={s[2]}) -> {table[s]}")
    print(f"  => ALGORITHM: {label_table(table)}")

    print("\n  LENGTH-GEN of the discovered policy (exact sort across lengths; trained on len<=%d):" % args.wmax)
    rep = verify_table(table, lengths=(2, 3, 5, 8, 12, 20, 30, 50), n=200)
    print("    TABLE : " + "  ".join(f"L{L}:{rep[L]:.3f}" for L in rep))
    print("    NEURAL: " + "  ".join(f"L{L}:{greedy_acc(model, L, n=200, seed=100 + L):.3f}" for L in (3, 8, 16, 30)))

    rng = random.Random(7); a = make_problem(12, rng)
    out, halted, steps = run_table(table, a, cap_for(12))
    print(f"\n  example sort (len 12): in {a}\n    out: {out}  sorted={is_sorted(out)}  steps={steps}")
    if args.save and not args.smoke:
        torch.save(model.state_dict(), args.save)
        print(f"\nsaved {args.save}")
