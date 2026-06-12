"""
expM_tower.py — THE TOWER: multiplication grounded entirely on the digit-successor (+1), with BOTH the
adder AND the multiply-composition DISCOVERED from outcome (no add primitive, no multiply primitive).

A learned LIBRARY, bottom-up:
  rung 2 (expM_add): ADDITION discovered from outcome = carry-add over the digit-wheel successor TICK.
                     Discovered body [LOADA CARRYTICK* TICK* EMIT], exact to w30, 4/4 seeds.
  rung 1 (expM_muldigit): MULTIPLICATION-composition discovered from outcome = per-digit loop with an
                     inner repeated-ADD partial product [GETDIGIT ADD_STEP* SHL ADD_ACC INC_J], exact w30.
Here we GROUND rung 1 on rung 2: the rung-1 MUL controller runs UNCHANGED, but every whole-number add it
relies on (ADD_STEP: VAL+=A ; ADD_ACC: ACC+=VAL) is performed by the DISCOVERED carry-adder, which itself
bottoms out in the digit-wheel successor. SHL is a structural place-value shift (append zeros). So the
ONLY arithmetic operation anywhere in the stack is +1 (TICK); addition and multiplication are both
DISCOVERED programs over it. We verify the grounded tower computes A*B EXACTLY and length-generalizes, and
report the digit-successor (TICK) count -- the whole multiplication reduced to counting.

Run: python expM_tower.py
"""
from __future__ import annotations
import random
import torch

import expM_add as ADDM
import expM_muldigit as MULM

base = 10
DEVICE = MULM.DEVICE

# The DISCOVERED adder (expM_add, robust 4/4 seeds). body token = (instr_id, is_loop, gate_obs_index).
A_I = ADDM.IID
ADD_BODY = (
    (A_I["LOADA"],     False, 0),
    (A_I["CARRYTICK"], True,  1),   # while cin: consume the carry by one wheel tick
    (A_I["TICK"],      True,  0),   # while K<b_i: tick the wheel (add B's digit)
    (A_I["EMIT"],      False, 0),
)

_tick_ids = {A_I["TICK"], A_I["CARRYTICK"]}


def add_fn(x, y, count=None):
    """x+y computed ONLY by the discovered carry-adder over digit-wheel ticks. Optionally tally ticks."""
    _, act, got, halted = ADDM.interpret("add", ADD_BODY, x, y, 0, base, cap=(base + 4) * (ADDM.E._ndigits(max(x, y, 1), base) + 3) + 80)
    assert halted, (x, y)
    if count is not None:
        count[0] += sum(1 for a in act if a in _tick_ids)
    return got


# ----------------------------------------------------------------------------
# Grounded multiplication VM: identical observable state transitions to expM_muldigit.VM, but ADD_STEP and
# ADD_ACC are performed by the DISCOVERED adder (over ticks), not Python '+'. SHL = structural shift.
# ----------------------------------------------------------------------------
class GroundedMulVM(MULM.VM):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.ticks = [0]

    def execute(self, instr):
        b = self.base
        if instr == "ADD_STEP":
            self.VAL = add_fn(self.VAL, self.A, self.ticks); self.K += 1     # repeated add via discovered adder
        elif instr == "ADD_ACC":
            self.ACC = add_fn(self.ACC, self.VAL, self.ticks)                # accumulate via discovered adder
        else:
            super().execute(instr)                                          # GETDIGIT/SHL/INC_J/HALT unchanged


@torch.no_grad()
def grounded_run(model, A, B, step_cap=None):
    model.eval()
    vm = GroundedMulVM("mul", A, B, 0, base)
    if step_cap is None:
        step_cap = (base + 6) * max(1, vm.N) + 20
    h = None; steps = 0
    while not vm.halted and steps < step_cap:
        obs = torch.tensor([vm.obs()], dtype=torch.float32, device=DEVICE).unsqueeze(0)
        logits, h = model(obs, h)
        instr = MULM.INSTRS[int(logits[0, -1].argmax())]
        vm.execute(instr); steps += 1
    return vm.answer(), vm.ticks[0], vm.halted


if __name__ == "__main__":
    # 1) the discovered ADDER is exact (built only from the digit-wheel successor)
    rng = random.Random(0)
    for _ in range(5000):
        x = rng.randint(0, 10 ** rng.randint(1, 8)); y = rng.randint(0, 10 ** rng.randint(1, 8))
        assert add_fn(x, y) == x + y, (x, y)
    print("discovered ADDER (digit-successor only): exact on 5000 checks incl. 8-digit operands.")

    # 2) load the discovered MUL controller (rung 1) and run it on the GROUNDED VM
    model = MULM.Controller(hidden=64).to(DEVICE)
    model.load_state_dict(torch.load("runs/expM_muldigit.pt", map_location=DEVICE))
    print("loaded discovered MUL controller (runs/expM_muldigit.pt).\n")

    print("TOWER length-gen: A*B with the ONLY arithmetic primitive = +1 (TICK); adder AND mul both DISCOVERED.")
    for w in (1, 2, 3, 4, 6, 8, 12, 16, 20):
        rng = random.Random(1234 + w); ok = 0; n = 60
        for _ in range(n):
            A = rng.randint(0, base ** w - 1); B = rng.randint(0, base ** w - 1)
            got, _, halted = grounded_run(model, A, B)
            ok += (halted and got == A * B)
        print(f"    w{w}: {ok/n:.3f}")

    print("\n  examples (digit-successor TICK count = the whole multiplication reduced to counting):")
    for (A, B) in [(47, 83), (9, 9), (123, 456), (99999, 99999)]:
        got, ticks, halted = grounded_run(model, A, B)
        print(f"    {A}*{B} = {got}  (expect {A*B}, {'OK' if got==A*B else 'WRONG'})  via {ticks} digit-successor ticks")
