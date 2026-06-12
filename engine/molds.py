"""Molds v0. A mold = candidate shape + edit moves + tidy-up + pour + pretty.

The mold owns the representation; proposers stay domain-blind by only ever
asking the mold for random candidates and mutations. v0 ships the
comparator-list mold (sorting networks). The bit-program mold (domain C) and
the decomposition mold (domain B) arrive in later build steps.
"""

import itertools

from engine.core_lang import Instr, Program


class ComparatorMold:
    """Candidates are tuples of (i, j) wire pairs with i < j.

    tidy() canonicalizes by greedy layering: each comparator lands in the
    earliest layer where neither of its wires is busy, layers are sorted
    internally, and immediately-repeated comparators (no-ops) are dropped.
    This is a v0 canonical form — enough to collapse reordered-but-identical
    networks; a stronger form is recognizer work (build step 2).
    """

    name = "comparator-list"

    def __init__(self, n_wires: int):
        assert n_wires >= 2
        self.n = n_wires
        self._pairs = list(itertools.combinations(range(n_wires), 2))

    # --- shape ----------------------------------------------------------
    def random_candidate(self, rng, length):
        return tuple(rng.choice(self._pairs) for _ in range(length))

    # --- edit moves -------------------------------------------------------
    def mutate(self, cand, rng):
        cand = list(cand)
        move = rng.choice(("add", "drop", "replace", "swap"))
        if move == "add" or not cand:
            cand.insert(rng.randrange(len(cand) + 1), rng.choice(self._pairs))
        elif move == "drop" and len(cand) > 1:
            cand.pop(rng.randrange(len(cand)))
        elif move == "replace":
            cand[rng.randrange(len(cand))] = rng.choice(self._pairs)
        elif len(cand) >= 2:  # swap two positions
            i, j = rng.sample(range(len(cand)), 2)
            cand[i], cand[j] = cand[j], cand[i]
        return tuple(cand)

    # --- layering (shared by tidy/depth/pretty) ---------------------------
    def _layers(self, cand):
        depth_of_wire = [0] * self.n
        layers = {}
        for (i, j) in cand:
            layer = max(depth_of_wire[i], depth_of_wire[j])
            layers.setdefault(layer, []).append((i, j))
            depth_of_wire[i] = depth_of_wire[j] = layer + 1
        return [sorted(layers[k]) for k in sorted(layers)]

    # --- tidy-up (canonical form) ------------------------------------------
    def tidy(self, cand):
        flat = []
        for pair in cand:
            if not flat or flat[-1] != pair:
                flat.append(pair)
        canon = []
        for layer in self._layers(flat):
            canon.extend(layer)
        return tuple(canon)

    def depth(self, cand):
        return len(self._layers(cand))

    # --- native cost ---------------------------------------------------------
    def native_cost(self, cand):
        return {"comparators": len(cand), "depth": self.depth(cand)}

    # --- pour: compile to core ------------------------------------------------
    def pour(self, cand) -> Program:
        t_lo, t_hi = self.n, self.n + 1
        instrs = []
        for (i, j) in cand:
            instrs.append(Instr("MIN", dst=t_lo, a=i, b=j, tags=("comparator",)))
            instrs.append(Instr("MAX", dst=t_hi, a=i, b=j))
            instrs.append(Instr("MOV", dst=i, a=t_lo))
            instrs.append(Instr("MOV", dst=j, a=t_hi))
        return Program(n_inputs=self.n, n_slots=self.n + 2, instrs=instrs,
                       meta={"mold": self.name, "n_wires": self.n}).validate()

    # --- human view --------------------------------------------------------------
    def pretty(self, cand):
        if not cand:
            return "(empty)"
        return " | ".join(
            " ".join(f"({i},{j})" for i, j in layer)
            for layer in self._layers(cand)
        )


if __name__ == "__main__":
    import random
    m = ComparatorMold(4)
    # the classic 5-comparator network for n=4 — its correctness is asserted
    # by the domain pack's checker, not trusted from memory
    net = ((0, 1), (2, 3), (0, 2), (1, 3), (1, 2))
    reordered = ((2, 3), (0, 1), (0, 2), (1, 3), (1, 2))
    assert m.tidy(net) == m.tidy(reordered), "tidy must be reorder-invariant"
    assert m.native_cost(net) == {"comparators": 5, "depth": 3}
    assert m.tidy(((0, 1), (0, 1), (2, 3))) == m.tidy(((0, 1), (2, 3))), "dup drop"
    rng = random.Random(0)
    c = m.random_candidate(rng, 6)
    for _ in range(300):
        c = m.mutate(c, rng)
        assert all(0 <= i < j < 4 for i, j in c)
    p = m.pour(net)
    assert len(p.instrs) == 5 * 4
    print("molds v0 ok:", m.pretty(net))
