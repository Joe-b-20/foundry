"""Bilinear-decomposition mold (the "decomposition IR" from the design
discussions). A candidate is a tuple of R products:

    ( (u, v, w), ... )   with u in Z^2, v in Z^2, w in Z^3 (small entries)

meaning: compute p_r = (u.a)(v.b) — ONE multiplication each — and emit
output_k = sum_r w_r[k] * p_r. The induced tensor is
T'[i][j][k] = sum_r u_r[i] v_r[j] w_r[k]; the candidate computes the
target bilinear map exactly iff T' equals the target tensor entry-for-
entry — an integer identity, which by bilinearity proves correctness for
ALL inputs. R is the multiplication count (Karatsuba: R=3 vs naive R=4).

This mold POURS TO CORE (first new mold that does): straight-line integer
program, products tagged "mult" for the counter bank.
"""

from engine.core_lang import Instr, Program


class BilinearMold:
    name = "bilinear-2x2x3"

    def __init__(self, entry_cap=2, max_rank=6):
        self.entry_cap = entry_cap
        self.max_rank = max_rank

    # --- shape ----------------------------------------------------------
    def random_candidate(self, rng, length=4):
        def vec(n):
            return tuple(rng.randint(-1, 1) for _ in range(n))
        return self.tidy(tuple((vec(2), vec(2), vec(3))
                               for _ in range(length)))

    # --- edit moves -------------------------------------------------------
    def mutate(self, cand, rng):
        prods = [list(map(list, p)) for p in cand]
        r = rng.random()
        if r < 0.70 and prods:                      # bump one entry
            p = rng.choice(prods)
            part = rng.choice((0, 1, 2))
            i = rng.randrange(len(p[part]))
            p[part][i] += rng.choice((-1, 1))
            cap = self.entry_cap
            p[part][i] = max(-cap, min(cap, p[part][i]))
        elif r < 0.85 and len(prods) < self.max_rank:   # add a product
            prods.append([[rng.randint(-1, 1) for _ in range(2)],
                          [rng.randint(-1, 1) for _ in range(2)],
                          [rng.randint(-1, 1) for _ in range(3)]])
        elif len(prods) > 1:                        # remove a product
            prods.pop(rng.randrange(len(prods)))
        return self.tidy(tuple(tuple(tuple(x) for x in p) for p in prods))

    # --- tidy-up: drop dead products, canonical signs, sorted order --------
    def tidy(self, cand):
        out = []
        for (u, v, w) in cand[: self.max_rank]:
            if not any(u) or not any(v) or not any(w):
                continue                      # contributes nothing: drop
            # sign canon: first nonzero of u positive (flip u AND v: (-u)(-v)
            # is the same product); then first nonzero of v positive (flip v
            # and absorb the sign into w)
            if next(x for x in u if x) < 0:
                u, v = tuple(-x for x in u), tuple(-x for x in v)
            if next(x for x in v if x) < 0:
                v, w = tuple(-x for x in v), tuple(-x for x in w)
            out.append((tuple(u), tuple(v), tuple(w)))
        return tuple(sorted(out))

    # --- the induced tensor (12 integers, order: i, j, k) -------------------
    @staticmethod
    def tensor(cand):
        t = []
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    t.append(sum(u[i] * v[j] * w[k] for (u, v, w) in cand))
        return tuple(t)

    # --- canonical key incl. the target's own symmetries --------------------
    def canonical_key(self, cand):
        """Min over the polymul2 symmetry group: swap a<->b (u<->v per
        product; the target tensor is i/j-symmetric) and reversal
        (i->1-i, j->1-j, k->2-k). Used for KNOWN-matching."""
        variants = []
        for swap in (False, True):
            for rev in (False, True):
                prods = []
                for (u, v, w) in cand:
                    uu, vv = (v, u) if swap else (u, v)
                    if rev:
                        uu, vv, w2 = uu[::-1], vv[::-1], w[::-1]
                    else:
                        w2 = w
                    prods.append((uu, vv, w2))
                variants.append(self.tidy(tuple(prods)))
        return min(variants)

    # --- native cost ----------------------------------------------------------
    def native_cost(self, cand):
        dl = sum(abs(x) for (u, v, w) in cand for x in (*u, *v, *w))
        return {"mults": len(cand), "dl": dl}

    # --- pour: compile to core ---------------------------------------------------
    def pour(self, cand) -> Program:
        # slots: 0..3 = a0,a1,b0,b1 (inputs); 4..6 = output accumulators;
        # 7 = t_u, 8 = t_v, 9 = scratch
        ACC, TU, TV, SCR = 4, 7, 8, 9
        ins = []

        def emit_combo(dst, base, coeffs):
            ins.append(Instr("CONST", dst=dst, imm=0))
            for off, c in enumerate(coeffs):
                slot = base + off
                for _ in range(abs(c)):
                    ins.append(Instr("ADD" if c > 0 else "SUB",
                                     dst=dst, a=dst, b=slot))

        for k in range(3):
            ins.append(Instr("CONST", dst=ACC + k, imm=0))
        for (u, v, w) in cand:
            emit_combo(TU, 0, u)
            emit_combo(TV, 2, v)
            ins.append(Instr("MUL", dst=SCR, a=TU, b=TV, tags=("mult",)))
            for k, c in enumerate(w):
                for _ in range(abs(c)):
                    ins.append(Instr("ADD" if c > 0 else "SUB",
                                     dst=ACC + k, a=ACC + k, b=SCR))
        for k in range(3):
            ins.append(Instr("MOV", dst=k, a=ACC + k))
        return Program(n_inputs=4, n_slots=10, instrs=ins,
                       meta={"mold": self.name}).validate()

    # --- human view -----------------------------------------------------------------
    @staticmethod
    def _lin(coeffs, names):
        s = ""
        for c, nm in zip(coeffs, names):
            if c == 0:
                continue
            s += ("+" if c > 0 else "-")
            s += (str(abs(c)) if abs(c) != 1 else "") + nm
        return (s[1:] if s.startswith("+") else s) or "0"

    def pretty(self, cand):
        lines = []
        for r, (u, v, w) in enumerate(cand):
            lines.append(f"m{r}=({self._lin(u, ('a0', 'a1'))})"
                         f"({self._lin(v, ('b0', 'b1'))})")
        outs = []
        for k in range(3):
            terms = "".join(("+" if w[k] > 0 else "-")
                            + (str(abs(w[k])) if abs(w[k]) != 1 else "")
                            + f"m{r}"
                            for r, (u, v, w) in enumerate(cand) if w[k])
            outs.append(f"c{k}=" + (terms[1:] if terms.startswith("+")
                                    else terms or "0"))
    # note: pretty is for humans; equality checks use tidy/canonical_key
        return " ; ".join(lines + outs)


if __name__ == "__main__":
    import random
    from engine import runner as core_runner
    m = BilinearMold()
    KARATSUBA = (((1, 0), (1, 0), (1, -1, 0)),
                 ((1, 1), (1, 1), (0, 1, 0)),
                 ((0, 1), (0, 1), (0, -1, 1)))
    TARGET = (1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1)
    assert m.tensor(m.tidy(KARATSUBA)) == TARGET
    # pour and execute: must equal direct polynomial multiplication
    prog = m.pour(m.tidy(KARATSUBA))
    rng = random.Random(0)
    for _ in range(30):
        a0, a1, b0, b1 = (rng.randint(-50, 50) for _ in range(4))
        out, cost = core_runner.run(prog, [a0, a1, b0, b1])
        assert out[:3] == [a0 * b0, a0 * b1 + a1 * b0, a1 * b1], out
    assert cost.native["mult"] == 3
    # canonical key absorbs the a<->b and reversal symmetries
    swapped = tuple((v, u, w) for (u, v, w) in KARATSUBA)
    assert m.canonical_key(m.tidy(swapped)) == m.canonical_key(m.tidy(KARATSUBA))
    c = m.random_candidate(rng, 4)
    for _ in range(300):
        c = m.mutate(c, rng)
        assert len(c) <= 6
    print("bilinear mold ok:", m.pretty(m.tidy(KARATSUBA)))
