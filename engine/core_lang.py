"""foundry core language v0 — straight-line, integer-only, cost-tagged.

Deliberately tiny. No branches, no loops: none of the launch domains
(comparator networks, bit programs, fixed-size decompositions) need them.
Control flow gets added the day a domain demands it — not before.

Every instruction can carry tags. Tags are how molds mark native cost units
(e.g. one "comparator" per compare-exchange) so the runner counts raw core
ops and mold-native units at the same time, on the same execution.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# op name -> (number of source operands, counter family it bumps)
OPS = {
    "CONST": (0, "mov"),   # m[dst] <- imm
    "MOV":   (1, "mov"),   # m[dst] <- m[a]
    "MIN":   (2, "cmp"),   # m[dst] <- min(m[a], m[b])
    "MAX":   (2, "cmp"),
    "LT":    (2, "cmp"),   # m[dst] <- 1 if m[a] < m[b] else 0
    "ADD":   (2, "add"),
    "SUB":   (2, "add"),
    "MUL":   (2, "mul"),
    "AND":   (2, "bit"),
    "OR":    (2, "bit"),
    "XOR":   (2, "bit"),
    # --- 32-bit "bits-typed" subset (added for the approximants domain,
    # the first domain that demanded floats). These ops read the LOW 32
    # BITS of a slot and interpret them per-op: F-ops view bits as IEEE-754
    # float32, compute, round to float32, store the result's bits; *32 ops
    # are uint32 arithmetic. Under bits-typing, int<->float reinterpretation
    # is free — which is exactly how the magic-constant trick family sees
    # the world. FDIV is deliberately absent for now: a float64
    # intermediate double-rounds division (add/sub/mul are exact in f64),
    # which would break bit-exactness against vectorized evaluators.
    "ADD32": (2, "bit"),
    "SUB32": (2, "bit"),
    "SHR32": (2, "bit"),   # shift amount = low 5 bits of m[b]
    "SHL32": (2, "bit"),
    "XOR32": (2, "bit"),
    "AND32": (2, "bit"),
    "OR32":  (2, "bit"),
    "FADD":  (2, "fadd"),
    "FSUB":  (2, "fadd"),
    "FMUL":  (2, "fmul"),
    # FDIV semantics (added for rationals, the saturating-family tool):
    # the float64 quotient rounded once to float32. NOT necessarily the
    # IEEE-correctly-rounded float32 quotient (that would need a non-stdlib
    # divide in the core runner) — but a single, deterministic, well-defined
    # function computed identically in both execution paths (runner: Python
    # f64 divide -> round f32; npfunc: f64 divide -> cast f32). x/0 = +-inf,
    # 0/0 = nan, like IEEE. Opt-in per mold (not in the default op set).
    "FDIV":  (2, "fdiv"),
    # value CONVERSIONS (not reinterpretations) — added for the log2/exp2
    # family (Blinn 1997 integer-aliasing tricks need them). Both are
    # single-rounding: uint32 -> float64 is exact, the float32 round is
    # the only rounding; F2U truncates toward zero with non-finite and
    # out-of-range mapped to 0 (defined semantics, matched in vectorized
    # evaluators).
    "U2F":   (1, "cvt"),
    "F2U":   (1, "cvt"),
}


@dataclass(frozen=True)
class Instr:
    op: str
    dst: int
    a: int = 0
    b: int = 0
    imm: Optional[int] = None
    tags: Tuple[str, ...] = ()

    def __post_init__(self):
        assert self.op in OPS, f"unknown op {self.op}"


@dataclass
class Program:
    n_inputs: int                 # m[0..n_inputs-1] hold the input on entry
    n_slots: int                  # total memory incl. temps
    instrs: List[Instr] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def validate(self):
        assert 0 < self.n_inputs <= self.n_slots
        for ins in self.instrs:
            srcs, _family = OPS[ins.op]
            assert 0 <= ins.dst < self.n_slots
            if srcs >= 1:
                assert 0 <= ins.a < self.n_slots
            if srcs >= 2:
                assert 0 <= ins.b < self.n_slots
            if ins.op == "CONST":
                assert ins.imm is not None
        return self


if __name__ == "__main__":
    p = Program(n_inputs=2, n_slots=3, instrs=[
        Instr("MIN", dst=2, a=0, b=1, tags=("comparator",)),
        Instr("MAX", dst=1, a=0, b=1),
        Instr("MOV", dst=0, a=2),
    ]).validate()
    assert len(p.instrs) == 3
    print("core_lang v0 ok")
