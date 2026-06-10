"""
expG_controller.py — INTERNAL composition: a learned recurrent controller that GENERATES and runs
the multi-step program, replacing the Python glue.

So far composition was external: Python loops called the model's primitives (long-mult loop,
division's repeated-subtraction loop). Here a tiny GRU CONTROLLER emits, step by step, the
instruction sequence (the PROGRAM) over a small register VM; a minimal ALU (the discovered
primitives — here exact integer ops, which the extracted carry/borrow FSMs reproduce exactly)
executes each instruction. The control flow / iteration / accumulation is now a LEARNED model
output, not Python.

Two composite tasks, ONE controller:
  * full n x n MULTIPLICATION (NOT finite-state -> needs an internal loop + growing accumulator),
  * DIVISION by ANY single digit incl. base-coprime (needs an internal repeated-subtraction loop;
    the remainder lives in an EXACT integer register, sidestepping the continuous-drift wall).

Crucial design: the per-step observation is only [op_mul, op_div, ge_flag, done_flag] (4 dims).
It is CONSTANT across the distinct instructions of a cycle, so a memoryless policy provably cannot
solve this — the controller MUST track the program phase in its recurrent state. Length-gen
(train widths 1..4, test to 20) tests whether the learned control FSM generalizes.

Run: python expG_controller.py
"""
from __future__ import annotations
import argparse, random
import torch, torch.nn as nn

import core_data as cd
import expA_mealy as E

DEVICE = E.DEVICE

INSTRS = ["HALT", "GETDIGIT", "MULDIGIT", "SHL", "ADD_ACC", "INC_J", "COMBINE", "SUB_D", "STOREQ"]
NI = len(INSTRS)
IID = {n: i for i, n in enumerate(INSTRS)}
OPS = ["mul", "div"]
N_OBS = 4   # [op_mul, op_div, ge, done]


# ----------------------------------------------------------------------------
# Register VM. Integer registers; ALU = exact integer ops (== the extracted carry/borrow FSMs).
# ----------------------------------------------------------------------------
class VM:
    def __init__(self, op, A, B, D, base):
        self.op, self.A, self.B, self.D, self.base = op, A, B, D, base
        self.ACC = 0; self.REM = 0; self.VAL = 0; self.CUR = 0; self.Q = 0; self.J = 0
        self.out = []                      # emitted quotient digits (div), MSB-first
        self.N = E._ndigits(B, base) if op == "mul" else E._ndigits(A, base)
        self.halted = False

    def obs(self):
        ge = 1.0 if (self.op == "div" and self.VAL >= self.D) else 0.0
        done = 1.0 if self.J >= self.N else 0.0
        return [1.0 if self.op == "mul" else 0.0, 1.0 if self.op == "div" else 0.0, ge, done]

    def execute(self, instr):
        b = self.base
        if instr == "GETDIGIT":
            if self.op == "mul":
                self.CUR = (self.B // b ** self.J) % b                 # B digit J, LSB-first
            else:
                self.CUR = (self.A // b ** (self.N - 1 - self.J)) % b  # A digit J, MSB-first
        elif instr == "MULDIGIT": self.VAL = self.A * self.CUR
        elif instr == "SHL":      self.VAL = self.VAL * (b ** self.J)
        elif instr == "ADD_ACC":  self.ACC = self.ACC + self.VAL
        elif instr == "INC_J":    self.J = self.J + 1
        elif instr == "COMBINE":  self.VAL = self.REM * b + self.CUR
        elif instr == "SUB_D":    self.VAL = self.VAL - self.D; self.Q = self.Q + 1
        elif instr == "STOREQ":   self.out.append(self.Q); self.REM = self.VAL; self.Q = 0
        elif instr == "HALT":     self.halted = True
        else: raise ValueError(instr)

    def answer(self):
        if self.op == "mul":
            return self.ACC
        q = 0
        for dig in self.out:
            q = q * self.base + dig
        return q


# ----------------------------------------------------------------------------
# Reference policy (the CORRECT program) — used to generate training traces.
# ----------------------------------------------------------------------------
def run_reference(vm, max_steps=100000):
    """Run the known program, recording (obs_before, instr) at each step."""
    trace = []
    def step(name):
        trace.append((vm.obs(), IID[name])); vm.execute(name)
    if vm.op == "mul":                      # acc += SHL(MULDIGIT(A,B_j), j) for each j
        while not vm.halted:
            step("GETDIGIT"); step("MULDIGIT"); step("SHL"); step("ADD_ACC"); step("INC_J")
            step("HALT") if vm.J >= vm.N else None
    else:                                   # long division, inner loop = repeated subtraction
        while not vm.halted:
            step("GETDIGIT"); step("COMBINE")
            while vm.VAL >= vm.D:
                step("SUB_D")
            step("STOREQ"); step("INC_J")
            step("HALT") if vm.J >= vm.N else None
            if len(trace) > max_steps: break
    return trace


def make_problem(op, width, base, rng):
    if op == "mul":
        A = rng.randint(0, base ** width - 1); B = rng.randint(0, base ** width - 1); D = 0
    else:
        A = rng.randint(0, base ** width - 1); B = 0; D = rng.randint(1, base - 1)
    return A, B, D


# ----------------------------------------------------------------------------
# Controller: GRU over the 4-dim observation -> next instruction.
# ----------------------------------------------------------------------------
class Controller(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.gru = nn.GRU(N_OBS, hidden, batch_first=True)
        self.head = nn.Linear(hidden, NI)

    def forward(self, obs, h=None):
        y, h = self.gru(obs, h)
        return self.head(y), h


def make_batch(n, widths, base, seed):
    rng = random.Random(seed)
    seqs = []
    for _ in range(n):
        op = rng.choice(OPS); w = rng.choice(widths)
        A, B, D = make_problem(op, w, base, rng)
        tr = run_reference(VM(op, A, B, D, base))
        obs = torch.tensor([o for o, _ in tr], dtype=torch.float32)
        tgt = torch.tensor([i for _, i in tr], dtype=torch.long)
        seqs.append((obs, tgt))
    L = max(s[0].shape[0] for s in seqs)
    obs_b = torch.zeros(n, L, N_OBS); tgt_b = torch.zeros(n, L, dtype=torch.long)
    mask = torch.zeros(n, L)
    for i, (obs, tgt) in enumerate(seqs):
        k = obs.shape[0]; obs_b[i, :k] = obs; tgt_b[i, :k] = tgt; mask[i, :k] = 1.0
    return obs_b, tgt_b, mask


def train(model, base=10, steps=6000, bs=64, lr=3e-3, widths=(1, 2, 3, 4)):
    model.to(DEVICE); opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss(reduction="none")
    for step in range(1, steps + 1):
        obs, tgt, mask = make_batch(bs, widths, base, seed=step)
        obs, tgt, mask = obs.to(DEVICE), tgt.to(DEVICE), mask.to(DEVICE)
        logits, _ = model(obs)
        loss = (lossf(logits.reshape(-1, NI), tgt.reshape(-1)) * mask.reshape(-1)).sum() / mask.sum()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 1000 == 0 or step == 1:
            print(f"    step {step:5d} loss {loss.item():.5f}")
    return model


# ----------------------------------------------------------------------------
# Inference: controller drives the VM (no Python algorithm — just the learned policy).
# ----------------------------------------------------------------------------
@torch.no_grad()
def controller_run(model, op, A, B, D, base, step_cap=None):
    model.eval()
    vm = VM(op, A, B, D, base)
    if step_cap is None:
        step_cap = 6 * vm.N + vm.N * base + 20
    h = None; steps = 0
    while not vm.halted and steps < step_cap:
        obs = torch.tensor([vm.obs()], dtype=torch.float32, device=DEVICE).unsqueeze(0)  # (1,1,4)
        logits, h = model(obs, h)
        instr = INSTRS[int(logits[0, -1].argmax())]
        vm.execute(instr); steps += 1
    return vm.answer(), steps, vm.halted


def lengen(model, base, widths=(1, 2, 3, 4, 6, 8, 12, 16, 20), n=300, seed=0):
    print("  CONTROLLER-driven exact accuracy (the model emits & runs the program):")
    for op in OPS:
        rep = {}
        for w in widths:
            rng = random.Random(seed + w); ok = 0
            for _ in range(n):
                A, B, D = make_problem(op, w, base, rng)
                got, _, halted = controller_run(model, op, A, B, D, base)
                exp = A * B if op == "mul" else A // D
                ok += (halted and got == exp)
            rep[w] = ok / n
        print(f"    {op}: " + "  ".join(f"w{w}:{rep[w]:.3f}" for w in widths))


def div_by_divisor(model, base, n=200):
    """Division per-divisor (does the controller cross the base-coprime wall internally?)."""
    print("  CONTROLLER division by each divisor (w=12; * = base-coprime 'wall' divisor):")
    for d in range(2, base):
        rng = random.Random(1000 + d); ok = 0
        for _ in range(n):
            A = rng.randint(0, base ** 12 - 1)
            got, _, halted = controller_run(model, "div", A, 0, d, base)
            ok += (halted and got == A // d)
        star = "" if base % d == 0 else " *"
        print(f"    /{d}{star}: {ok/n:.3f}")


def show_program(model, base, op, A, B, D):
    """Print the instruction sequence the controller EMITS (the program it composed)."""
    vm = VM(op, A, B, D, base); h = None; prog = []
    cap = 6 * vm.N + vm.N * base + 20; steps = 0
    with torch.no_grad():
        while not vm.halted and steps < cap:
            obs = torch.tensor([vm.obs()], dtype=torch.float32, device=DEVICE).unsqueeze(0)
            logits, h = model(obs, h)
            instr = INSTRS[int(logits[0, -1].argmax())]
            prog.append(instr); vm.execute(instr); steps += 1
    label = f"{A}*{B}" if op == "mul" else f"{A}/{D}"
    exp = A * B if op == "mul" else A // D
    print(f"  emitted program for {label} (={exp}, got {vm.answer()}): {' '.join(prog)}")


if __name__ == "__main__":
    import os
    os.makedirs("runs", exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    base = 10

    # sanity: the reference policy + VM compute correct answers
    rng = random.Random(0)
    for _ in range(2000):
        op = rng.choice(OPS); w = rng.randint(1, 6); A, B, D = make_problem(op, w, base, rng)
        vm = VM(op, A, B, D, base); run_reference(vm)
        exp = A * B if op == "mul" else A // D
        assert vm.halted and vm.answer() == exp, (op, A, B, D, vm.answer(), exp)
    print("reference VM+policy compute mul and div EXACTLY on 2000 checks.")

    if args.smoke:
        args.steps = 1500
    print(f"device={DEVICE}  training recurrent controller ({args.steps} steps)")
    torch.manual_seed(0)
    model = Controller(hidden=args.hidden)
    print(f"  params: {sum(p.numel() for p in model.parameters())}")
    train(model, base=base, steps=args.steps)

    print("\n----- EVAL: internal composition (trained on widths 1..4) -----")
    lengen(model, base)
    div_by_divisor(model, base)
    print("\n  example emitted programs:")
    show_program(model, base, "mul", 47, 83, 0)
    show_program(model, base, "div", 1234, 0, 7)
    show_program(model, base, "div", 9999, 0, 3)
    if not args.smoke:
        torch.save(model.state_dict(), "runs/expG_controller.pt")
        print("\nsaved runs/expG_controller.pt")
