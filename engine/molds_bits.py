"""Bit-program mold: candidates are short straight-line programs over
word operations — and they ARE core programs (op, dst, a, b over a small
slot set), so pour() is nearly the identity. Slots 0,1 hold the inputs
x,y; 2,3 are temps; the output is slot 0, masked to w bits by the pack.

Masking note: +, *, &, |, ^ commute with reduction mod 2^w (Python's
bitwise negatives are infinite two's complement, consistent under the
mask), so candidates run on the unmodified core runner and only the final
output is masked. No special bit-width machinery needed.
"""

from engine.core_lang import Instr, Program

OPS = ("XOR", "AND", "OR", "ADD", "MUL")


class BitProgMold:
    name = "bit-program"

    def __init__(self, n_slots=4, max_len=8, ops=OPS):
        self.n_slots = n_slots
        self.max_len = max_len
        self.ops = ops

    # --- shape ----------------------------------------------------------
    def _rand_instr(self, rng):
        return (rng.choice(self.ops), rng.randrange(self.n_slots),
                rng.randrange(self.n_slots), rng.randrange(self.n_slots))

    def random_candidate(self, rng, length=4):
        return tuple(self._rand_instr(rng)
                     for _ in range(min(length, self.max_len)))

    # --- edit moves -------------------------------------------------------
    def mutate(self, cand, rng):
        prog = list(cand)
        r = rng.random()
        if r < 0.60 and prog:                       # tweak one field
            i = rng.randrange(len(prog))
            op, dst, a, b = prog[i]
            field = rng.randrange(4)
            if field == 0:
                op = rng.choice(self.ops)
            elif field == 1:
                dst = rng.randrange(self.n_slots)
            elif field == 2:
                a = rng.randrange(self.n_slots)
            else:
                b = rng.randrange(self.n_slots)
            prog[i] = (op, dst, a, b)
        elif r < 0.80 and len(prog) < self.max_len:  # insert
            prog.insert(rng.randrange(len(prog) + 1), self._rand_instr(rng))
        elif len(prog) > 1:                          # delete
            prog.pop(rng.randrange(len(prog)))
        return tuple(prog)

    # --- tidy-up: drop ops whose result is never used (cheap liveness) ----
    def tidy(self, cand):
        cand = tuple(cand[: self.max_len])
        live = {0}                  # output slot
        keep = [False] * len(cand)
        for i in range(len(cand) - 1, -1, -1):
            op, dst, a, b = cand[i]
            if dst in live:
                keep[i] = True
                live.discard(dst)
                live.update((a, b))
        return tuple(c for c, k in zip(cand, keep) if k)

    # --- native cost ----------------------------------------------------------
    def native_cost(self, cand):
        return {"ops": len(cand), "dl": len(cand)}

    # --- pour: straight to core --------------------------------------------
    def pour(self, cand) -> Program:
        ins = [Instr(op, dst=dst, a=a, b=b, tags=("bitop",))
               for (op, dst, a, b) in cand]
        return Program(n_inputs=2, n_slots=self.n_slots, instrs=ins,
                       meta={"mold": self.name}).validate()

    # --- human view -----------------------------------------------------------
    def pretty(self, cand):
        names = ["x", "y", "t2", "t3"]
        sym = {"XOR": "^", "AND": "&", "OR": "|", "ADD": "+", "MUL": "*"}
        return "; ".join(f"{names[d]}={names[a]}{sym[o]}{names[b]}"
                         for (o, d, a, b) in cand) or "(empty)"


if __name__ == "__main__":
    import random
    from engine import runner as core_runner
    m = BitProgMold()
    cand = (("XOR", 2, 0, 1), ("MUL", 3, 2, 2), ("ADD", 0, 3, 1))
    out, cost = core_runner.run(m.pour(cand), [5, 3])
    t = 5 ^ 3
    assert out[0] == t * t + 3, out
    assert cost.native["bitop"] == 3
    # tidy drops dead code: the t3 write below is never used
    dead = (("XOR", 2, 0, 1), ("MUL", 3, 2, 2), ("ADD", 0, 2, 1))
    assert m.tidy(dead) == (("XOR", 2, 0, 1), ("ADD", 0, 2, 1))
    rng = random.Random(0)
    c = m.random_candidate(rng, 4)
    for _ in range(300):
        c = m.mutate(c, rng)
        assert len(c) <= 8
    print("bit mold ok:", m.pretty(cand))
