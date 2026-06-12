"""
core_data.py — exact arithmetic data + codecs + eval. Pure Python (no torch/numpy),
so it can be sanity-checked with any python3.

Conventions used across the project:
- Numbers are sequences of digits in a fixed base, **least-significant-digit first**
  (digit[0] is the ones place). This lets a left-to-right scan carry.
- All arithmetic ground truth is exact integer arithmetic. Eval is exact-match on
  the integer result. No partial credit (see .claude/RULES.md).
- The headline metric for the project is LENGTH GENERALIZATION: train on short
  numbers, test on long ones. A lookup table fails this; a real algorithm passes.
"""
from __future__ import annotations
import random
from typing import Callable, List, Sequence, Tuple

# ----------------------------------------------------------------------------
# Digit codecs (LSB-first)
# ----------------------------------------------------------------------------

def to_digits(n: int, width: int, base: int = 10) -> List[int]:
    """Non-negative int -> list of `width` digits, least-significant first.
    Raises if n doesn't fit in `width` digits."""
    assert n >= 0, "to_digits handles non-negative ints only"
    ds = []
    x = n
    for _ in range(width):
        ds.append(x % base)
        x //= base
    assert x == 0, f"{n} does not fit in {width} base-{base} digits"
    return ds


def from_digits(ds: Sequence[int], base: int = 10) -> int:
    """list of digits (LSB-first) -> int."""
    n = 0
    for i, d in enumerate(ds):
        assert 0 <= d < base, f"digit {d} out of range for base {base}"
        n += d * (base ** i)
    return n


# ----------------------------------------------------------------------------
# Exact operation semantics
# ----------------------------------------------------------------------------
OPS = ("add", "sub", "mul", "div")


def exact_result(a: int, b: int, op: str) -> int:
    if op == "add":
        return a + b
    if op == "sub":
        return a - b           # may be negative
    if op == "mul":
        return a * b
    if op == "div":
        assert b != 0, "division by zero"
        return a // b          # floor division; remainder handled separately if needed
    raise ValueError(f"unknown op {op}")


# ----------------------------------------------------------------------------
# Operand sampling with digit-width control
# ----------------------------------------------------------------------------

def max_for_width(width: int, base: int = 10) -> int:
    return base ** width - 1


def sample_operand(width: int, base: int = 10, rng: random.Random | None = None) -> int:
    """Uniform integer in [0, base**width - 1] (i.e. up to `width` digits)."""
    rng = rng or random
    return rng.randint(0, max_for_width(width, base))


def sample_pairs(n: int, width: int, base: int = 10, seed: int | None = None,
                 nonneg_sub: bool = False, op: str = "add") -> List[Tuple[int, int]]:
    """Sample n operand pairs, each operand up to `width` digits.
    If nonneg_sub and op=='sub', enforce a>=b. If op=='div', enforce b!=0."""
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        a = sample_operand(width, base, rng)
        b = sample_operand(width, base, rng)
        if op == "sub" and nonneg_sub and a < b:
            a, b = b, a
        if op == "div" and b == 0:
            continue
        out.append((a, b))
    return out


def all_pairs(width: int, base: int = 10, op: str = "add") -> List[Tuple[int, int]]:
    """Exhaustive list of all operand pairs up to `width` digits.
    Only call for tiny width (e.g. base 10 width 2 -> 10_000 pairs)."""
    hi = max_for_width(width, base)
    out = []
    for a in range(hi + 1):
        for b in range(hi + 1):
            if op == "div" and b == 0:
                continue
            out.append((a, b))
    return out


# ----------------------------------------------------------------------------
# Addition transducer IO  (digit-serial, LSB-first)
# ----------------------------------------------------------------------------

def addition_io(a: int, b: int, width: int, base: int = 10) -> Tuple[List[int], List[int], List[int]]:
    """For the digit-serial addition transducer.
    Returns (a_digits, b_digits, sum_digits) where a_digits/b_digits have length
    `width` and sum_digits has length `width+1` (room for a final carry-out).
    All LSB-first."""
    a_d = to_digits(a, width, base)
    b_d = to_digits(b, width, base)
    s_d = to_digits(a + b, width + 1, base)   # +1 for carry-out
    return a_d, b_d, s_d


# ----------------------------------------------------------------------------
# Exact eval harness
# ----------------------------------------------------------------------------

def exact_accuracy(predict_fn: Callable[[int, int], int],
                   pairs: Sequence[Tuple[int, int]], op: str) -> float:
    """Fraction of pairs where predict_fn(a,b) exactly equals exact_result(a,b,op)."""
    if not pairs:
        return float("nan")
    correct = 0
    for a, b in pairs:
        if predict_fn(a, b) == exact_result(a, b, op):
            correct += 1
    return correct / len(pairs)


def length_gen_report(predict_fn: Callable[[int, int], int], op: str, base: int = 10,
                      widths: Sequence[int] = (1, 2, 3, 4, 6, 8, 10, 12),
                      n_per_width: int = 2000, seed: int = 0) -> dict:
    """Exact accuracy at a range of operand widths. The whole point: accuracy
    should stay ~1.0 across widths if a real algorithm was found, and collapse
    on widths beyond the training range if only a lookup table was learned."""
    report = {}
    for w in widths:
        pairs = sample_pairs(n_per_width, w, base=base, seed=seed + w, op=op,
                             nonneg_sub=(op == "sub"))
        report[w] = exact_accuracy(predict_fn, pairs, op)
    return report


# ----------------------------------------------------------------------------
# Sanity checks
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    # codec round-trips
    for base in (2, 10, 16):
        for _ in range(1000):
            w = random.randint(1, 6)
            n = random.randint(0, base ** w - 1)
            assert from_digits(to_digits(n, w, base), base) == n
    # LSB-first check: 23 in base 10 -> [3, 2]
    assert to_digits(23, 3, 10) == [3, 2, 0], to_digits(23, 3, 10)
    assert from_digits([3, 2, 0], 10) == 23

    # exact ops
    assert exact_result(2, 3, "add") == 5
    assert exact_result(2, 3, "sub") == -1
    assert exact_result(7, 6, "mul") == 42
    assert exact_result(7, 2, "div") == 3

    # addition_io: 999 + 1 -> sum needs width+1 digits
    a_d, b_d, s_d = addition_io(999, 1, 3, 10)
    assert a_d == [9, 9, 9] and b_d == [1, 0, 0]
    assert from_digits(s_d, 10) == 1000 and len(s_d) == 4

    # eval harness: a perfect adder scores 1.0 at every width
    rep = length_gen_report(lambda a, b: a + b, "add",
                            widths=(1, 2, 3, 6, 10), n_per_width=500)
    assert all(abs(v - 1.0) < 1e-9 for v in rep.values()), rep
    # a "lookup table" that only knows pairs < 100 collapses on long numbers
    def lut(a, b):
        return a + b if (a < 100 and b < 100) else 0
    rep2 = length_gen_report(lut, "add", widths=(1, 2, 6), n_per_width=500)
    assert rep2[1] == 1.0 and rep2[2] == 1.0 and rep2[6] == 0.0, rep2

    print("core_data.py sanity checks PASS")
    print("len-gen report for true adder:", rep)
    print("len-gen report for <100 lookup:", rep2)
