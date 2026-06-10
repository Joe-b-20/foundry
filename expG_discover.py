"""
expG_discover.py — can the controller DISCOVER the composition on its OWN (no program traces)?

expG_controller.py put composition inside the model but via TRACE SUPERVISION (imitating the known
program). The stronger claim is the user's "develop compositional solutions on its own": discover
the program from OUTCOME alone. Here the controller is trained by REINFORCE — reward = digit
accuracy of the final answer (with an exact-match bonus) — and NEVER sees a correct instruction.
It must discover, from reward, that "to multiply, loop: getdigit, muldigit, shift, add" etc.

Weird/from-scratch knobs (not default PPO): plain REINFORCE + moving-average baseline + entropy
bonus + a length curriculum (start width 1) + invalid-action masking by op (mul can't pick div
instrs and vice-versa) to shrink the search. Honest pre-registration: discovery from sparse-ish
reward over a ~6-15 step program is hard; this may only partially work. Reported either way.

Run: python expG_discover.py --op mul --steps 8000
"""
from __future__ import annotations
import argparse, random
import torch, torch.nn as nn

import core_data as cd
import expA_mealy as E
from expG_controller import VM, Controller, INSTRS, IID, NI, make_problem

DEVICE = E.DEVICE

# valid instruction sets per op (mask out the others to shrink the search space)
MUL_OK = {"HALT", "GETDIGIT", "MULDIGIT", "SHL", "ADD_ACC", "INC_J"}
DIV_OK = {"HALT", "GETDIGIT", "COMBINE", "SUB_D", "STOREQ", "INC_J"}
def op_mask(op):
    ok = MUL_OK if op == "mul" else DIV_OK
    return torch.tensor([0.0 if INSTRS[i] in ok else -1e9 for i in range(NI)])


def digits(n, base):
    if n == 0: return [0]
    d = []
    while n > 0:
        d.append(n % base); n //= base
    return d


def reward(op, A, B, D, got, base, halted):
    exp = A * B if op == "mul" else A // D
    if got == exp:
        return 1.0
    if not halted:
        return -0.2
    te, tg = digits(exp, base), digits(got, base)
    L = max(len(te), len(tg))
    te += [0] * (L - len(te)); tg += [0] * (L - len(tg))
    match = sum(1 for x, y in zip(te, tg) if x == y) / L      # LSB-aligned digit accuracy
    return 0.6 * match                                         # partial credit < exact bonus


@torch.no_grad()
def _greedy_eval(model, op, base, widths, n=200, seed=0):
    from expG_controller import controller_run
    rep = {}
    for w in widths:
        rng = random.Random(seed + w); ok = 0
        for _ in range(n):
            A, B, D = make_problem(op, w, base, rng)
            got, _, halted = controller_run(model, op, A, B, D, base)
            exp = A * B if op == "mul" else A // D
            ok += (halted and got == exp)
        rep[w] = ok / n
    return rep


def rollout(model, op, A, B, D, base, mask, step_cap, sample=True):
    """One episode; return (logps, entropies, answer, halted)."""
    vm = VM(op, A, B, D, base); h = None
    logps, ents = [], []
    steps = 0
    while not vm.halted and steps < step_cap:
        obs = torch.tensor([vm.obs()], dtype=torch.float32, device=DEVICE).unsqueeze(0)
        logits, h = model(obs, h)
        logits = logits[0, -1] + mask.to(DEVICE)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample() if sample else logits.argmax()
        logps.append(dist.log_prob(a)); ents.append(dist.entropy())
        vm.execute(INSTRS[int(a)]); steps += 1
    return logps, ents, vm.answer(), vm.halted


def discover(model, op, base=10, steps=8000, bs=32, lr=3e-3, beta=0.02):
    model.to(DEVICE); opt = torch.optim.Adam(model.parameters(), lr=lr)
    mask = op_mask(op)
    baseline = 0.0; rng = random.Random(1)
    wmax = 1                                   # length curriculum
    solved_run = 0
    for step in range(1, steps + 1):
        model.train()                              # _greedy_eval() flips to eval; restore for backward
        widths = list(range(1, wmax + 1))
        batch_loss = 0.0; rewards = []
        for _ in range(bs):
            w = rng.choice(widths); A, B, D = make_problem(op, w, base, rng)
            cap = 6 * (w + 1) + (w + 1) * base + 10
            logps, ents, got, halted = rollout(model, op, A, B, D, base, mask, cap)
            R = reward(op, A, B, D, got, base, halted); rewards.append(R)
            adv = R - baseline
            lp = torch.stack(logps).sum(); ent = torch.stack(ents).sum()
            batch_loss = batch_loss + (-adv * lp - beta * ent)
        batch_loss = batch_loss / bs
        opt.zero_grad(); batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        mR = sum(rewards) / len(rewards); baseline = 0.9 * baseline + 0.1 * mR
        if step % 100 == 0:
            print(f"    step {step:5d}  wmax {wmax}  meanR {mR:.3f}  baseline {baseline:.3f}")
        # advance curriculum when current width is essentially solved (greedy)
        if step % 500 == 0:
            acc = _greedy_eval(model, op, base, [wmax], n=80)[wmax]
            print(f"      greedy@w{wmax} {acc:.3f}")
            if acc > 0.9 and wmax < 5:
                wmax += 1; print(f"      -> curriculum advance to width {wmax}")
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", default="mul", choices=["mul", "div"])
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=24)
    args = ap.parse_args()
    base = 10
    print(f"device={DEVICE}  REINFORCE discovery of {args.op} composition (reward=digit acc, NO traces)")
    torch.manual_seed(0)
    model = Controller(hidden=args.hidden)
    discover(model, args.op, base=base, steps=args.steps, bs=args.bs)
    print("\n  greedy length-gen after discovery:")
    rep = _greedy_eval(model, args.op, base, (1, 2, 3, 4, 6, 8, 12), n=200)
    print("    " + "  ".join(f"w{w}:{rep[w]:.3f}" for w in rep))
    from expG_controller import show_program
    if args.op == "mul":
        show_program(model, base, "mul", 47, 83, 0)
    else:
        show_program(model, base, "div", 1234, 0, 7)
