"""
expQ_sort.py — can self-discovery discover SORTING from OUTCOME ALONE, and WHICH algorithm?

FIRST non-arithmetic op in the project: sort a LIST. The project's signature lever (length-gen = "real
algorithm vs lookup") fits perfectly: train on SHORT lists, test on LONG ones -- a memorized sort fails
long lists, a real comparison-sort generalizes. To force a genuine comparison algorithm (not value
memorization), the controller sees ONLY comparison/structure FLAGS, never the element values.

Whole-LIST register VM: array A[0..n-1], a single scan pointer P (0..n-2), a dirty bit (a swap happened
since the last RESET). Instructions (general list ops, none sort-specific):
  SWAP   swap A[P],A[P+1] ; set dirty     (adjacent transposition)
  ADV    P += 1                           (advance the scan; invalid at the last pair)
  RESET  P = 0 ; clear dirty              (start a new pass)
  HALT   output A
obs = [gt = A[P]>A[P+1], end = (P==n-2), dirty].  With this obs BUBBLE SORT is a MEMORYLESS reactive policy:
  gt -> SWAP ; (~gt,~end) -> ADV ; (~gt,end,dirty) -> RESET ; (~gt,end,~dirty) -> HALT.
So "which algorithm" = read the greedy obs->action table; the headline test = exact LENGTH-GEN (sort lists
far longer than trained). Same exact-filtered self-imitation recipe as expK GCD (sample -> keep only
exactly-sorted within a step budget -> self-imitate -> LOCK the greedy table once it sorts across LENGTHS ->
distill). NO traces; the only signal is the VM's yes/no on whether the output list is non-decreasing.
Run: python expQ_sort.py --iters 150
"""
from __future__ import annotations
import argparse, itertools, random
from collections import deque, Counter
import torch, torch.nn as nn

import expA_mealy as E

DEVICE = E.DEVICE
INSTRS = ["HALT", "SWAP", "ADV", "RESET"]
NI = len(INSTRS)
IID = {n: i for i, n in enumerate(INSTRS)}
N_OBS = 3                                        # [gt, end, dirty]
OBS_STATES = [(g, e, d) for g in (0, 1) for e in (0, 1) for d in (0, 1)]
ALLOWED = set(INSTRS)


def mask_vec():
    return torch.tensor([0.0 if INSTRS[i] in ALLOWED else -1e9 for i in range(NI)], device=DEVICE)


def is_sorted(a):
    return all(a[i] <= a[i + 1] for i in range(len(a) - 1))


class SortVM:
    def __init__(self, arr):
        self.A = list(arr)
        self.n = len(arr)
        self.P = 0
        self.dirty = False
        self.halted = False
        self.invalid = False

    def obs(self):
        gt = 1.0 if (self.P + 1 < self.n and self.A[self.P] > self.A[self.P + 1]) else 0.0
        end = 1.0 if self.P >= self.n - 2 else 0.0
        return [gt, end, 1.0 if self.dirty else 0.0]

    def execute(self, instr):
        if instr == "SWAP":
            if self.P + 1 < self.n:
                self.A[self.P], self.A[self.P + 1] = self.A[self.P + 1], self.A[self.P]
                self.dirty = True
            else:
                self.invalid = True
        elif instr == "ADV":
            if self.P < self.n - 2:
                self.P += 1
            else:
                self.invalid = True                      # must RESET/HALT at the last pair, not ADV past it
        elif instr == "RESET":
            self.P = 0; self.dirty = False
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
    return 3 * L * L + 12                              # generous O(n^2) budget for an adjacent-swap sort


def make_problem(L, rng):
    """Random list of L ints; ensure NOT already sorted (so a correct program must do real work -> the
    trivial 'HALT now' never passes; no lucky hits). Controller never sees these values, only gt flags."""
    while True:
        a = [rng.randint(0, 999) for _ in range(L)]
        if L < 2 or not is_sorted(a):
            return a


def sample_rollouts(model, arrs, step_cap, temp=1.0, eps=0.0):
    M = len(arrs)
    vms = [SortVM(a) for a in arrs]
    h = None
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
    out = []
    for i in range(M):
        ok = vms[i].halted and (not vms[i].invalid) and is_sorted(vms[i].A)
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
    model.eval()
    table = {}
    for s in OBS_STATES:
        obs = torch.tensor([list(map(float, s))], dtype=torch.float32, device=DEVICE).unsqueeze(0)
        logits, _ = model(obs)
        table[s] = INSTRS[int((logits[0, -1] + mask_vec()).argmax())]
    model.train()
    return table


def run_table(table, arr, cap):
    vm = SortVM(arr); steps = 0
    while not vm.halted and not vm.invalid and steps < cap:
        s = tuple(int(x) for x in vm.obs())
        instr = table.get(s)
        if instr is None:
            return None, False, steps
        vm.execute(instr); steps += 1
    return vm.A, (vm.halted and not vm.invalid), steps


def candidate_tables(counts, max_cand=64):
    """Propose candidate tables from the model's votes. For AMBIGUOUS obs-states (a 2nd action also has
    real support -- e.g. (gt0,end,dirty1) gets both HALT [array happened to be sorted] and RESET [do another
    pass]), try BOTH; the cross-length verify then selects the GENERALIZING action (RESET). This is the
    isqrt lesson: don't trust the raw majority -- let exact-length-gen pick among the model's own actions."""
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
    cands = []
    for combo in itertools.product(*opts):
        t = dict(base)
        for k, s in enumerate(states):
            t[s] = combo[k]
        cands.append(t)
    return cands


def verify_table(table, lengths=(2, 3, 5, 8, 12, 20), n=80, seed=1):
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
        a = make_problem(L, rng); vm = SortVM(a); h = None; steps = 0
        while not vm.halted and not vm.invalid and steps < cap:
            obs = torch.tensor([vm.obs()], dtype=torch.float32, device=DEVICE).unsqueeze(0)
            logits, h = model(obs, h)
            vm.execute(INSTRS[int((logits[0, -1] + mask_vec()).argmax())]); steps += 1
        ok += (vm.halted and not vm.invalid and is_sorted(vm.A))
    model.train(); return ok / n


def clean(o, a):
    """VALIDITY filter (analog of expK's A>=B guard / isqrt's narrow-per-le): never SWAP an in-order pair
    (gt=0). Drops the lucky/noisy correct rollouts (redundant swap-and-swap-back) that otherwise poison
    self-imitation into 'always SWAP'. Does NOT specify when to ADV/RESET/HALT (the algorithm)."""
    sw = IID["SWAP"]
    return not any(a[i] == sw and o[i][0] < 0.5 for i in range(len(a)))


def induce_table(o, a):
    """Read the obs->action map a rollout USED. Return it iff INTERNALLY CONSISTENT (each obs-state maps to
    exactly one action) -- i.e. the rollout is a genuine MEMORYLESS policy, not an exploration-noised one.
    This keeps only clean memoryless sorts so the consensus is the real algorithm, not averaged noise."""
    t = {}
    for ob, ac in zip(o, a):
        s = tuple(int(x) for x in ob)
        if s in t and t[s] != ac:
            return None
        t[s] = ac
    return t


def selfdiscover(model, iters=150, M=1536, bs_im=256, grad_steps=6, lr=3e-3, wmax=5,
                 buf_max=8000, seed=0, log_every=10, verbose=True, lock_lengths=(2, 3, 5, 8, 12)):
    model.to(DEVICE); opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = random.Random(seed)
    buffer = deque(maxlen=buf_max)
    counts = {s: Counter() for s in OBS_STATES}          # consensus: majority action per obs-state
    first_hit = None; locked = None; history = []
    for it in range(1, iters + 1):
        model.train()
        if locked is None:
            # bias to SMALL lengths: all 8 obs-states already occur by L=3, and clean memoryless sorts are
            # easy to sample there -> the revealed table then generalizes to any length (verified below).
            arrs = [make_problem(rng.choice((2, 3, 3, 3, 4, 4, 5)), rng) for _ in range(M)]
            results = sample_rollouts(model, arrs, cap_for(wmax), temp=1.0, eps=0.15)
            n_ok = 0
            for ok, o, a in results:
                if not ok or not clean(o, a):            # exact-sorted AND clean (no in-order swaps)
                    continue
                if induce_table(o, a) is None:           # AND internally consistent (a real memoryless policy)
                    continue
                n_ok += 1
                if first_hit is None:
                    first_hit = it
                for ob, ac in zip(o, a):
                    buffer.append((ob, ac))
                    counts[tuple(int(x) for x in ob)][ac] += 1
            # propose candidate tables from the votes; the cross-length verify selects the generalizing one
            if all(counts[s] for s in OBS_STATES) and it % 5 == 0:
                for cand in candidate_tables(counts):
                    rep = verify_table(cand, lengths=lock_lengths, n=40, seed=it)
                    if min(rep.values()) > 0.97:
                        locked = cand
                        if verbose:
                            tstr = " ; ".join(f"{s}->{cand[s]}" for s in OBS_STATES)
                            print(f"      *** DISCOVERED & LOCKED policy (it {it}): {tstr} ***")
                        break
            if len(buffer) >= bs_im:                     # imitate CLEAN pairs to sharpen the model's rollouts
                for _ in range(grad_steps):
                    imitate(model, opt, random.sample(buffer, bs_im))
            n_log = n_ok
        else:
            pairs = [(list(map(float, s)), IID[locked[s]]) for s in OBS_STATES] * (bs_im // len(OBS_STATES))
            for _ in range(grad_steps):
                imitate(model, opt, pairs)
            n_log = -1
        if it % log_every == 0 or it == 1:
            accs = {L: greedy_acc(model, L, n=60, seed=it) for L in (2, 5, 12)}
            astr = " ".join(f"L{L}:{accs[L]:.2f}" for L in accs)
            cand = {s: (INSTRS[counts[s].most_common(1)[0][0]] if counts[s] else "?") for s in OBS_STATES}
            nseen = sum(1 for s in OBS_STATES if counts[s])
            lk = "LOCKED" if locked else "      "
            print(f"  it {it:4d} {lk} ok {n_log:5d}/{M}  states {nseen}/8  greedy[{astr}]")
            history.append((it, locked is not None, accs))
    return model, {"first_hit": first_hit, "locked": locked, "consensus":
                   {s: (INSTRS[counts[s].most_common(1)[0][0]] if counts[s] else "?") for s in OBS_STATES}}, history


# bubble sort as the canonical memoryless table (for labeling the discovered table)
BUBBLE = {(1, 0, 0): "SWAP", (1, 0, 1): "SWAP", (0, 0, 0): "ADV", (0, 0, 1): "ADV",
          (1, 1, 0): "SWAP", (1, 1, 1): "SWAP", (0, 1, 0): "HALT", (0, 1, 1): "RESET"}


def label_table(table):
    """Identify which algorithm the discovered obs->action table implements (on the REACHABLE states)."""
    reachable = [(1, 0, 0), (0, 0, 0), (0, 0, 1), (1, 0, 1), (1, 1, 0), (1, 1, 1), (0, 1, 0), (0, 1, 1)]
    if all(table[s] == BUBBLE[s] for s in reachable):
        return "BUBBLE SORT (exact canonical table)"
    # core decisions that define bubble sort:
    core = (table[(1, 0, 0)] == "SWAP" and table[(0, 0, 0)] == "ADV" and
            table[(0, 1, 1)] == "RESET" and table[(0, 1, 0)] == "HALT")
    return "BUBBLE SORT (core swap/advance/reset/halt rule)" if core else "OTHER / non-bubble"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--M", type=int, default=1536)
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--wmax", type=int, default=5, help="max list length during discovery")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default="")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.iters = 80

    # sanity: the canonical bubble table sorts exactly across lengths
    for L in (2, 3, 5, 9, 16):
        rng = random.Random(L)
        for _ in range(200):
            a = make_problem(L, rng)
            out, halted, _ = run_table(BUBBLE, a, cap_for(L))
            assert halted and is_sorted(out), ("bubble ref", a, out)
    print("reference BUBBLE table sorts exactly (sanity, L up to 16).")

    print(f"\ndevice={DEVICE}  SELF-DISCOVERY of SORTING from OUTCOME ALONE (no traces)  wmax={args.wmax}")
    torch.manual_seed(args.seed)
    model = Controller(hidden=args.hidden)
    print(f"  params: {sum(p.numel() for p in model.parameters())}")
    model, info, _ = selfdiscover(model, iters=args.iters, M=args.M, wmax=args.wmax, seed=args.seed)
    print(f"\n  first exact-correct self-sample at iter: {info['first_hit']}")
    table = info["locked"] if info["locked"] else info["consensus"]
    print(f"  DISCOVERED policy table ({'LOCKED' if info['locked'] else 'consensus, not locked'}):")
    for s in OBS_STATES:
        print(f"    (gt={s[0]}, end={s[1]}, dirty={s[2]}) -> {table[s]}")
    print(f"  => ALGORITHM: {label_table(table)}")

    print("\n  LENGTH-GEN of the discovered policy (exact sort across lengths; trained on len<=%d):" % args.wmax)
    rep = verify_table(table, lengths=(2, 3, 5, 8, 12, 20, 30, 50), n=200)
    print("    TABLE : " + "  ".join(f"L{L}:{rep[L]:.3f}" for L in rep))
    print("    NEURAL: " + "  ".join(f"L{L}:{greedy_acc(model, L, n=200, seed=100 + L):.3f}" for L in (2, 5, 12, 20, 30)))

    print("\n  example sort (len 12), showing it is bubble sort:")
    rng = random.Random(7); a = make_problem(12, rng)
    out, halted, steps = run_table(table, a, cap_for(12))
    print(f"    in : {a}")
    print(f"    out: {out}   sorted={is_sorted(out)}  steps={steps}")
    if args.save and not args.smoke:
        torch.save(model.state_dict(), args.save)
        print(f"\nsaved {args.save}")
