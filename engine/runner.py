"""foundry runner v0 — the only path anything executes on.

Deterministic: same program + same input = same result, forever. Counts
everything on every run: raw steps, per-family op counts, mold-native tagged
units. Hard step budget — overruns raise, they are never silently truncated.
Candidates see inputs only; there is no way to reach the checker, the shelf,
or other candidates from in here.
"""

from collections import Counter
from dataclasses import dataclass, field

from engine.core_lang import OPS, Program


@dataclass
class CostVector:
    steps: int = 0
    by_family: Counter = field(default_factory=Counter)
    native: Counter = field(default_factory=Counter)

    def as_dict(self):
        return {"steps": self.steps,
                "by_family": dict(self.by_family),
                "native": dict(self.native)}


class BudgetExceeded(Exception):
    pass


def run(program: Program, inputs, step_budget: int = 100_000):
    """Execute; return (outputs, CostVector).

    Memory model is in-place: outputs are slots 0..n_inputs-1 after the run.
    """
    assert len(inputs) == program.n_inputs
    if len(program.instrs) > step_budget:
        raise BudgetExceeded(
            f"{len(program.instrs)} instrs > step budget {step_budget}")
    m = list(inputs) + [0] * (program.n_slots - program.n_inputs)
    cost = CostVector()
    for ins in program.instrs:
        op = ins.op
        if op == "CONST":
            m[ins.dst] = ins.imm
        elif op == "MOV":
            m[ins.dst] = m[ins.a]
        elif op == "MIN":
            m[ins.dst] = m[ins.a] if m[ins.a] <= m[ins.b] else m[ins.b]
        elif op == "MAX":
            m[ins.dst] = m[ins.a] if m[ins.a] >= m[ins.b] else m[ins.b]
        elif op == "LT":
            m[ins.dst] = 1 if m[ins.a] < m[ins.b] else 0
        elif op == "ADD":
            m[ins.dst] = m[ins.a] + m[ins.b]
        elif op == "SUB":
            m[ins.dst] = m[ins.a] - m[ins.b]
        elif op == "MUL":
            m[ins.dst] = m[ins.a] * m[ins.b]
        elif op == "AND":
            m[ins.dst] = m[ins.a] & m[ins.b]
        elif op == "OR":
            m[ins.dst] = m[ins.a] | m[ins.b]
        elif op == "XOR":
            m[ins.dst] = m[ins.a] ^ m[ins.b]
        cost.steps += 1
        cost.by_family[OPS[op][1]] += 1
        for t in ins.tags:
            cost.native[t] += 1
    return m[: program.n_inputs], cost


if __name__ == "__main__":
    from engine.core_lang import Instr
    p = Program(n_inputs=2, n_slots=3, instrs=[
        Instr("MIN", dst=2, a=0, b=1, tags=("comparator",)),
        Instr("MAX", dst=1, a=0, b=1),
        Instr("MOV", dst=0, a=2),
    ]).validate()
    out, cost = run(p, [9, 4])
    assert out == [4, 9], out
    assert cost.steps == 3 and cost.native["comparator"] == 1
    out2, _ = run(p, [-5, -7])
    assert out2 == [-7, -5], out2
    print("runner v0 ok:", out, cost.as_dict())
