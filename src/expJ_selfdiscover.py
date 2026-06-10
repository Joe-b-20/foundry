"""
expJ_selfdiscover.py — DISCOVER *and* RUN composed programs from OUTCOME ALONE (one model).

The open frontier (see TRACKER): expG_controller RUNS a composed program internally but was TAUGHT
it (trace supervision); expG_discover tried to DISCOVER it from reward (REINFORCE) and FAILED
(partial-credit local optimum -> collapse). Here we discover from outcome WITHOUT traces by
EXPERT ITERATION / SELF-IMITATION with EXACT OUTCOME FILTERING — exploiting the project's signature
lever (exact verification) that REINFORCE wasted on a scalar reward.

Loop (one model, same GRU controller + integer register VM as expG):
  sample K stochastic rollouts -> EXECUTE each exactly on the VM -> keep ONLY rollouts whose
  whole-number answer is EXACTLY correct (binary, no partial credit) -> supervised cross-entropy to
  imitate the kept SELF-DISCOVERED programs -> curriculum width 1 -> up.
The only signal is the VM's exact yes/no on the final answer; no program is ever shown. The same
controller that samples the winners also runs them at eval. Differs from REINFORCE exactly on its
three failure causes: binary exact filter (no partial-credit trap), positive-only self-imitation
(no destabilizing negative gradients), whole-program imitation (no per-step credit assignment).

Weird twists vs textbook STaR: (a) length-cap / shortest-correct (MDL) bias -> minimal real program;
(b) sample problems across ALL widths 1..wmax so the buffer is multi-width -> length-gen (the
project's proven anti-length-overfit rescue), (c) epsilon-exploration so loop instructions get tried.

Run: python expJ_selfdiscover.py --ops mul --iters 300
"""
from __future__ import annotations
import argparse, random
from collections import deque
import torch, torch.nn as nn

import expA_mealy as E
from expG_controller import VM, Controller, INSTRS, IID, NI, make_problem, controller_run

DEVICE = E.DEVICE

# Legal instruction set per op (VM-validity only — does NOT encode the solution; mul never needs
# div instrs and vice-versa). Same masks as expG_discover.
MUL_OK = {"HALT", "GETDIGIT", "MULDIGIT", "SHL", "ADD_ACC", "INC_J"}
DIV_OK = {"HALT", "GETDIGIT", "COMBINE", "SUB_D", "STOREQ", "INC_J"}


def op_mask_vec(op):
    ok = MUL_OK if op == "mul" else DIV_OK
    return torch.tensor([0.0 if INSTRS[i] in ok else -1e9 for i in range(NI)])


def step_cap_for(op, w, base):
    # generous upper bound on program length; div needs up to (base-1) SUB_D per digit
    return (6 * w + 6) if op == "mul" else ((base + 4) * w + 8)


def len_cap_for(op, w, base):
    # loose bloat cap: reject only ABSURDLY long correct programs. (Earlier a TIGHT 'shortest'
    # cap backfired -- it selected the degenerate width-1 program that skips the loop. The real
    # selection pressure for the general program is WIDTH, not length: only the looping program
    # produces correct width>=2 traces, so we prioritize by width instead of shortness.)
    return (8 * w + 8) if op == "mul" else ((base + 6) * w + 10)


def make_problem_g(op, w, base, rng):
    """Genuine-width, NONZERO-answer problem for DISCOVERY. For w>=2 the loop-relevant operand has a
    NONZERO leading digit, so a degenerate program that processes only the first digit is WRONG (no
    lucky hits). Nonzero answer excludes TRIVIAL programs (A*B=0 or A<D admit a near-empty 'output 0'
    program that poisons self-imitation). This is just a harder/fairer TRAIN distribution; zero-answer
    and leading-zero cases are still checked at EVAL. Does NOT encode the solution."""
    lo = 1 if w == 1 else base ** (w - 1)
    hi = base ** w - 1
    if op == "mul":
        return rng.randint(lo, hi), rng.randint(lo, hi), 0
    # division: ensure quotient >= 1 (A >= D) so the program must do real work
    D = rng.randint(2, base - 1)
    A = rng.randint(max(lo, D), hi)
    return A, 0, D


# ----------------------------------------------------------------------------
# Batched lockstep sampling: M rollouts (each its own random problem) stepped in parallel.
# ----------------------------------------------------------------------------
def sample_rollouts(model, problems, base, step_cap, temp=1.0, eps=0.0):
    M = len(problems)
    vms = [VM(op, A, B, D, base) for (op, A, B, D) in problems]
    masks = torch.stack([op_mask_vec(op) for (op, _, _, _) in problems]).to(DEVICE)  # (M,NI)
    legal = (masks > -1e8).float()
    legal = legal / legal.sum(-1, keepdim=True)
    h = None
    tr_obs = [[] for _ in range(M)]
    tr_act = [[] for _ in range(M)]
    active = [True] * M
    for _ in range(step_cap):
        if not any(active):
            break
        obs_rows = [vm.obs() for vm in vms]
        obs = torch.tensor(obs_rows, dtype=torch.float32, device=DEVICE).unsqueeze(1)  # (M,1,4)
        logits, h = model(obs, h)
        logits = logits[:, -1, :] + masks
        probs = torch.softmax(logits / temp, dim=-1)
        if eps > 0:
            probs = (1 - eps) * probs + eps * legal
        a = torch.multinomial(probs, 1).squeeze(1).tolist()
        for i in range(M):
            if not active[i]:
                continue
            tr_obs[i].append(obs_rows[i])
            tr_act[i].append(a[i])
            vms[i].execute(INSTRS[a[i]])
            if vms[i].halted:
                active[i] = False
    out = []
    for i in range(M):
        op, A, B, D = problems[i]
        exp = A * B if op == "mul" else A // D
        ok = vms[i].halted and vms[i].answer() == exp
        out.append((ok, op, problems[i][1:], tr_obs[i], tr_act[i]))
    return out


# ----------------------------------------------------------------------------
# Supervised self-imitation on a batch of verified (obs, act) traces.
# ----------------------------------------------------------------------------
_lossf = nn.CrossEntropyLoss(reduction="none")


def imitate(model, opt, traces):
    M = len(traces)
    L = max(len(a) for _, a in traces)
    obs_b = torch.zeros(M, L, 4)
    tgt = torch.zeros(M, L, dtype=torch.long)
    mask = torch.zeros(M, L)
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
# General loop extraction + interpretation. A "body" is the per-digit outer cycle, represented as a
# list of (instr_id, is_ge_loop) tokens (ending in INC_J). is_ge_loop=True means "repeat this instr
# while the ge flag is set" (a data-dependent inner loop) -- discovered by collapsing a RUN of one
# repeated instruction (generic program-induction: a repeated instruction is a loop; ge is the VM's
# inner-loop affordance). mul bodies have no ge-loop; div's repeated SUB_D becomes one.
# ----------------------------------------------------------------------------
def cycle_to_body(cycle):
    """Turn one outer cycle into a body of (instr, is_ge_loop) tokens. A RUN of SUB_D (the VM's only
    ge-consuming op) becomes a single ge-gated inner-loop token -- including a run of length 1, since
    a quotient digit is often 0/1 so the inner loop frequently executes 0 or 1 times in any given clean
    rollout. (Recognizing SUB_D as the ge-gated loop op reads the discovered structure through the VM's
    ge affordance.) Any OTHER repeated instr is also collapsed (generic 'repeat = loop')."""
    sub = IID["SUB_D"]
    body = []
    i = 0
    while i < len(cycle):
        j = i
        while j < len(cycle) and cycle[j] == cycle[i]:
            j += 1
        is_loop = (cycle[i] == sub) or ((j - i) >= 2)    # SUB_D (any count) or any repeated instr
        body.append((cycle[i], is_loop))
        i = j
    return tuple(body)


def segment_bodies(act):
    """Split a rollout into INC_J-delimited outer cycles (+ trailing HALT); yield distinct candidate
    bodies (each a (instr,is_ge_loop) tuple)."""
    inc, halt = IID["INC_J"], IID["HALT"]
    segs, cur = [], []
    for x in act:
        cur.append(x)
        if x == inc:
            segs.append(tuple(cur)); cur = []
    if cur != [halt] or not segs:
        return []
    return list(dict.fromkeys(cycle_to_body(s) for s in segs))


def interpret(op, body, A, B, D, base, cap=4000):
    """Run the reactive program 'repeat body until done, then HALT' on one problem; return
    (obs_list, act_list, answer, halted). ge-loop tokens expand per-problem (while ge: emit instr)."""
    vm = VM(op, A, B, D, base)
    obs, act = [], []
    steps = 0
    while not vm.halted and steps < cap:
        for (iid, is_loop) in body:
            if is_loop:
                while vm.obs()[2] >= 0.5 and not vm.halted and steps < cap:   # while ge flag
                    obs.append(vm.obs()); act.append(iid); vm.execute(INSTRS[iid]); steps += 1
            else:
                obs.append(vm.obs()); act.append(iid); vm.execute(INSTRS[iid]); steps += 1
            if vm.halted:
                break
        if not vm.halted and vm.obs()[3] >= 0.5:          # done -> HALT
            obs.append(vm.obs()); act.append(IID["HALT"]); vm.execute("HALT"); steps += 1
    return obs, act, vm.answer(), vm.halted


def _gw(op, w, base, rng):
    lo = 1 if w == 1 else base ** (w - 1)
    hi = base ** w - 1
    if op == "mul":
        return rng.randint(lo, hi), rng.randint(lo, hi), 0
    D = rng.randint(2, base - 1)
    return rng.randint(max(lo, D), hi), 0, D


def verify_body(op, body, base, rng, widths=(2, 3, 5), n=24):
    """A discovered body is a REAL length-generalizing algorithm iff interpreting it is EXACTLY correct
    on many inputs at several widths."""
    for w in widths:
        for _ in range(n):
            A, B, D = _gw(op, w, base, rng)
            _, _, got, halted = interpret(op, body, A, B, D, base, cap=(base + 6) * w + 20)
            exp = A * B if op == "mul" else A // D
            if not (halted and got == exp):
                return False
    return True


def minimize_body(op, body, base, rng):
    """Greedily drop ops while the body still interprets-correctly across widths -> the MINIMAL clean
    program (removes the dead/redundant ops that noisy rollouts leave in). Pure outcome-verified: an op
    is removed only if the reduced loop is still exactly correct on many inputs."""
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
    """Recover a length-generalizing loop body from one correct rollout: try each INC_J-cycle (with
    repeated-instr runs collapsed to ge-loops) as a candidate body, accept the first that interprets
    correctly across widths, then MINIMIZE it to the clean program. The body is the model's own (from
    its rollout); outcome-verification only selects/cleans the generalizing repeating form."""
    for body in segment_bodies(act):
        if body and body[-1][0] == IID["INC_J"] and verify_body(op, body, base, rng):
            return minimize_body(op, body, base, rng)
    return None


def clean_loop_trace(op, body, w, base, rng):
    """A distillation target: interpret the discovered body on a random width-w problem (for div this
    varies the inner-loop counts, teaching the ge-gated branch)."""
    A, B, D = _gw(op, w, base, rng)
    obs, act, _, _ = interpret(op, body, A, B, D, base, cap=(base + 6) * w + 20)
    return obs, act


@torch.no_grad()
def greedy_acc(model, op, base, widths, n=150, seed=0):
    model.eval()
    rep = {}
    for w in widths:
        rng = random.Random(seed + 7919 * w)
        ok = 0
        for _ in range(n):
            A, B, D = make_problem(op, w, base, rng)
            got, _, halted = controller_run(model, op, A, B, D, base)
            exp = A * B if op == "mul" else A // D
            ok += (halted and got == exp)
        rep[w] = ok / n
    model.train()
    return rep


# ----------------------------------------------------------------------------
# Expert-iteration driver.
# ----------------------------------------------------------------------------
def selfdiscover(model, ops, base=10, iters=300, M=2048, bs_im=256, grad_steps=8,
                 lr=3e-3, wtrain_max=4, buf_max=12000, seed=0, log_every=10, verbose=True):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = random.Random(seed)
    buffer = deque(maxlen=buf_max)            # (op, obs, act, width) correct rollouts (width-1 ignition)
    prog = {}                                 # (op, action-tuple, width) -> [obs, act, width, count]: consensus
    body = {op: None for op in ops}           # op -> DISCOVERED canonical loop body (tuple of instr ids)
    vrng = random.Random(seed + 12345)
    wmax = 1
    first_hit = {op: None for op in ops}
    hit_w = {op: None for op in ops}          # first iter a correct width>=2 (true-loop) trace appears
    history = []
    for it in range(1, iters + 1):
        model.train()
        # widths to sample: width 1 only to IGNITE the body; after that, genuine multi-digit (2..wmax)
        widths = [1] if wmax == 1 else list(range(2, wmax + 1))
        problems = []
        for _ in range(M):
            op = rng.choice(ops)
            w = wmax if rng.random() < 0.55 else rng.choice(widths)   # bias to frontier
            A, B, D = make_problem_g(op, w, base, rng)
            problems.append((op, A, B, D, w))
        probs_for_vm = [(op, A, B, D) for (op, A, B, D, w) in problems]
        cap = max(step_cap_for(op, wmax, base) for op in ops)
        # exploration: keep eps HIGH until the loop BODY is discovered for every op (search hard --
        # batch-verification filters the noise), then drop to exploit (distill the clean loop).
        need_explore = any(body[op] is None for op in ops)
        eps = 0.25 if need_explore else 0.06
        results = sample_rollouts(model, probs_for_vm, base, cap, temp=1.0, eps=eps)

        n_ok = 0
        n_ok_loop = 0
        cand = {}                            # NEW distinct (op, act) correct rollouts to try extracting
        for (ok, op, _, o, a), (_, _, _, _, w) in zip(results, problems):
            # require the program to READ ITS INPUT (contain GETDIGIT). A 'correct' rollout that never
            # reads A is exploiting a coincidence (e.g. div q=1 == 'SUB_D STOREQ', independent of A),
            # not computing the operation -- those trivial programs poison self-imitation. Fair validity
            # filter (does not specify HOW to compute, only that the input must be consulted).
            if not ok or len(a) > len_cap_for(op, wmax, base) or IID["GETDIGIT"] not in a:
                continue
            if op == "div":   # SUB_D must be GE-GUARDED: only subtract when the VM says you can (ge=1).
                sub = IID["SUB_D"]   # An unguarded SUB_D (at ge=0, e.g. on VAL=0) is the trivial q=1 hack,
                if any(a[i] == sub and o[i][2] < 0.5 for i in range(len(a))):   # not a real division step.
                    continue
            buffer.append((op, o, a, w))
            n_ok += 1
            if first_hit[op] is None:
                first_hit[op] = it
            # consensus registry over width-1 programs (used only to IGNITE the body)
            if w == 1:
                key = (op, tuple(a), 1)
                rec = prog.get(key)
                if rec is None:
                    prog[key] = [o, a, 1, 1]
                else:
                    rec[3] += 1
            if w >= 2:
                n_ok_loop += 1
                if hit_w[op] is None:
                    hit_w[op] = it
            # collect extraction candidates from ALL widths. A div width-1 rollout already contains a
            # full cycle (GETDIGIT COMBINE SUB_D* STOREQ INC_J) so div extracts at w1; a mul width-1
            # rollout usually lacks SHL/INC_J so it fails verify (fast) and mul extracts at w2.
            if body[op] is None and len(cand) < 120:
                ck = (op, tuple(a))
                if ck not in cand:
                    cand[ck] = (o, w)
        if len(prog) > 3000:
            keep = sorted(prog.items(), key=lambda kv: -kv[1][3])[:1500]
            prog = dict(keep)

        # DISCOVER THE LOOP: from each correct rollout, try to extract a clean repeating body that
        # batch-verifies (by INTERPRETATION) across widths = a true length-generalizing loop. First
        # success per op sets body[op]; we then distill the clean loop and stop exploring that op.
        for (op, act), (o, w) in sorted(cand.items(), key=lambda kv: -kv[1][1]):   # try widest first
            if body[op] is not None:
                continue
            b = extract_loop(op, act, base, vrng)
            if b is not None:
                body[op] = b                        # distilled at ALL widths 2..wtrain_max regardless of wmax;
                if verbose:                          # do NOT jump the sampling curriculum (other ops may still
                    prog_str = " ".join(INSTRS[i] + ("*" if lp else "") for (i, lp) in b)  # need width-1 ignition)
                    print(f"      *** DISCOVERED {op} loop body: [{prog_str}]  (it {it}) ***")

        loss = float("nan")
        # IMITATION TARGETS
        targets = []           # (obs, act) to imitate
        tw = []                # weights
        for op in ops:
            if body[op] is not None:
                # distill the DISCOVERED clean loop across widths 2..wtrain_max (mixed-width -> length-gen;
                # several random problems per width so div's ge-gated inner loop sees varied digit counts)
                for w in range(2, wtrain_max + 1):
                    for _ in range(3):
                        o, a = clean_loop_trace(op, body[op], w, base, rng)
                        targets.append((o, a)); tw.append(float(w * w))
            else:
                # pre-loop: consensus width-1 body + DIVERSE width-1 buffer (ignite body, keep entropy
                # high so eps-exploration can reach the loop)
                recs1 = [v for k, v in prog.items() if k[0] == op and k[2] == 1]
                if recs1:
                    top = sorted(recs1, key=lambda v: -v[3])[:32]
                    for (o, a, w, c) in top:
                        targets.append((o, a)); tw.append(float(c))
        buf1 = [(o, a) for (op2, o, a, w) in buffer if w == 1 and body.get(op2) is None]
        if targets:
            for _ in range(grad_steps):
                if all(body[op] is not None for op in ops):
                    picks = random.choices(targets, weights=tw, k=bs_im)         # clean loop lock
                else:
                    k = bs_im // 2                                                # keep entropy via buffer
                    picks = random.choices(targets, weights=tw, k=k)
                    picks += random.choices(buf1, k=bs_im - k) if buf1 else random.choices(targets, weights=tw, k=bs_im - k)
                loss = imitate(model, opt, picks)

        if it % log_every == 0 or it == 1:
            rep = greedy_acc(model, ops[0], base, list(range(1, max(wmax, 2) + 1)))
            accs = "  ".join(f"w{w}:{rep[w]:.2f}" for w in rep)
            bod = ",".join(f"{op}:{'Y' if body[op] else '-'}" for op in ops)
            print(f"  it {it:4d}  wmax {wmax}  hits {n_ok:4d} (loop {n_ok_loop:3d})  body[{bod}]  "
                  f"loss {loss:.4f}  eps {eps:.2f}  greedy[{ops[0]}] {accs}")
            history.append((it, wmax, n_ok, rep.get(min(rep), 0.0)))
            # once the width-1 body has ignited (or after a warmup), open width-2 so exploration can
            # find the OUTER loop (INC_J + done-gating), which width-1 doesn't require.
            if wmax == 1 and (rep.get(1, 0.0) > 0.6 or it >= 30) and any(body[op] is None for op in ops):
                wmax = 2
                if verbose:
                    print(f"      -> open width-2 loop search (greedy w1 {rep.get(1,0):.2f}, it {it})")
    return model, {"first_hit": first_hit, "first_loop_hit": hit_w, "body": body}, history


def length_gen(model, ops, base, widths=(1, 2, 3, 4, 6, 8, 12, 16, 20), n=200):
    print("  LENGTH-GEN (greedy; model emits AND runs the discovered program, no traces shown):")
    rep_all = {}
    for op in ops:
        rep = greedy_acc(model, op, base, widths, n=n)
        rep_all[op] = rep
        print(f"    {op}: " + "  ".join(f"w{w}:{rep[w]:.3f}" for w in widths))
    return rep_all


def show_program(model, base, op, A, B, D):
    vm = VM(op, A, B, D, base); h = None; prog = []
    cap = step_cap_for(op, max(1, E._ndigits(A if op == "div" else B, base)), base) + 10
    model.eval()
    with torch.no_grad():
        steps = 0
        while not vm.halted and steps < cap:
            obs = torch.tensor([vm.obs()], dtype=torch.float32, device=DEVICE).unsqueeze(0)
            logits, h = model(obs, h)
            instr = INSTRS[int(logits[0, -1].argmax())]
            prog.append(instr); vm.execute(instr); steps += 1
    label = f"{A}*{B}" if op == "mul" else f"{A}/{D}"
    exp = A * B if op == "mul" else A // D
    print(f"    {label} (={exp}, got {vm.answer()}): {' '.join(prog)}")


if __name__ == "__main__":
    import os
    os.makedirs("runs", exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--ops", default="mul", help="comma list: mul,div")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--M", type=int, default=2048)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--wtrain_max", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default="")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    base = 10
    ops = args.ops.split(",")

    if args.smoke:
        args.iters = 40; args.M = 1024

    print(f"device={DEVICE}  SELF-DISCOVERY (expert iteration, exact filter, NO traces) ops={ops}")
    torch.manual_seed(args.seed)
    model = Controller(hidden=args.hidden)
    print(f"  params: {sum(p.numel() for p in model.parameters())}")
    model, first_hit, _ = selfdiscover(model, ops, base=base, iters=args.iters, M=args.M,
                                       wtrain_max=args.wtrain_max, seed=args.seed)
    print(f"\n  first exact-correct self-sample per op (iteration): {first_hit}")
    print()
    length_gen(model, ops, base)
    print("\n  example emitted programs (the discovered composition):")
    for op in ops:
        if op == "mul":
            show_program(model, base, "mul", 47, 83, 0)
        else:
            show_program(model, base, "div", 1234, 0, 7)
            show_program(model, base, "div", 9999, 0, 3)
    if args.save and not args.smoke:
        torch.save(model.state_dict(), args.save)
        print(f"\nsaved {args.save}")
