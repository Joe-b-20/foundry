"""PCF mold v2: a(n) dense, b(n) FACTORED — candidates are
    (a_coeffs, (coef, power, factor_constants))
representing a(n) = sum a_k n^k and b(n) = coef * n^power * prod (n + c_i).

v1 mutated raw dense coefficients and the replication walk went nowhere:
109 mutants, zero family members reached (runs/pcf-replication-s0-*).
The published kappa/Table-7 families are parameterized by FACTOR shifts —
(n+mu) -> (n+mu+1) is one move in factored space but a far, junk-filled
jump in coefficient space. The parent prototype knew this: its mutation
grammar was B-shift / B-split / power moves on factored forms
(reference/mathlab/src/foundry.py:262-294). Moves define the search
geometry; this mold makes published-family adjacency one move long.

Still no pour to core (needs loops + bignum); the metered path is
engine/numeric.py. Dense expansion for evaluation is mold.dense().
"""


class PCFMold:
    name = "pcf-factored"
    pours_to_core = False

    def __init__(self, max_deg_a=3, max_factors=4, max_power=6,
                 coeff_cap=16, const_cap=6):
        self.max_deg_a = max_deg_a
        self.max_factors = max_factors
        self.max_power = max_power
        self.coeff_cap = coeff_cap
        self.const_cap = const_cap

    # --- shape -----------------------------------------------------------
    def random_candidate(self, rng, _length=None):
        a = tuple(rng.randint(-4, 4) for _ in range(rng.randint(2, self.max_deg_a + 1)))
        b = (rng.choice((-3, -2, -1, 1, 2)), rng.randint(0, 3),
             tuple(sorted(rng.randint(-2, 3)
                          for _ in range(rng.randint(0, 2)))))
        return self.tidy((a, b))

    # --- edit moves ---------------------------------------------------------
    def mutate(self, cand, rng):
        a = list(cand[0])
        coef, power, factors = cand[1]
        factors = list(factors)
        r = rng.random()
        if r < 0.30 and factors:                       # B-shift: factor const +-1
            i = rng.randrange(len(factors))
            factors[i] += rng.choice((-1, 1))
        elif r < 0.45 and len(factors) < self.max_factors:   # B-split/add factor
            factors.append(rng.choice((-2, -1, 0, 1, 2)))
        elif r < 0.55 and factors:                     # remove factor
            factors.pop(rng.randrange(len(factors)))
        elif r < 0.70:                                 # monomial power +-1
            power = max(0, min(self.max_power, power + rng.choice((-1, 1))))
        elif r < 0.80:                                 # leading coef bump
            coef += rng.choice((-2, -1, 1, 2))
        else:                                          # a(n) coefficient bump
            if rng.random() < 0.15 and len(a) <= self.max_deg_a:
                a.append(rng.choice((-2, -1, 1, 2)))
            else:
                i = rng.randrange(len(a))
                a[i] += rng.choice((-2, -1, 1, 2))
        return self.tidy((tuple(a), (coef, power, tuple(factors))))

    # --- tidy-up --------------------------------------------------------------
    def tidy(self, cand):
        a = list(cand[0])
        while len(a) > 1 and a[-1] == 0:
            a.pop()
        if not any(a):
            a = [1]
        coef, power, factors = cand[1]
        coef = max(-self.coeff_cap, min(self.coeff_cap, coef))
        if coef == 0:
            coef = -1
        power = max(0, min(self.max_power, power))
        factors = tuple(sorted(max(-self.const_cap, min(self.const_cap, c))
                               for c in factors))[: self.max_factors]
        return (tuple(a), (coef, power, factors))

    # --- dense expansion (what the numeric engine evaluates) ----------------
    @staticmethod
    def dense(cand):
        a, (coef, power, factors) = cand
        b = [0] * power + [coef]
        for c in factors:
            out = [0] * (len(b) + 1)
            for i, x in enumerate(b):
                out[i] += x * c
                out[i + 1] += x
            b = out
        return (tuple(a), tuple(b))

    # --- native cost ------------------------------------------------------------
    def native_cost(self, cand):
        a, b = self.dense(cand)
        dl = sum(len(str(abs(c))) + (1 if c < 0 else 0)
                 for c in (*a, *b) if c != 0)
        return {"dl": dl, "deg_a": len(a) - 1, "deg_b": len(b) - 1}

    # --- human view ----------------------------------------------------------------
    @staticmethod
    def _poly_str(cs, var="n"):
        parts = []
        for k in range(len(cs) - 1, -1, -1):
            c = cs[k]
            if c == 0:
                continue
            term = (f"{abs(c)}" if k == 0 else
                    f"{'' if abs(c) == 1 else abs(c)}{var}"
                    + (f"^{k}" if k > 1 else ""))
            parts.append(("-" if c < 0 else "+") + term)
        s = "".join(parts) or "0"
        return s[1:] if s.startswith("+") else s

    def pretty(self, cand):
        a, (coef, power, factors) = cand
        bstr = f"{coef}" + (f"n^{power}" if power > 1 else "n" if power else "")
        for c in factors:
            bstr += f"(n{'+' if c >= 0 else '-'}{abs(c)})" if c else "(n)"
        return f"a(n)={self._poly_str(a)} ; b(n)={bstr}"


if __name__ == "__main__":
    import random
    m = PCFMold()
    # rm-8-7zeta3: a=(2n+1)(3n^2+3n+1) dense, b=-n^6 factored
    rm = ((1, 5, 9, 6), (-1, 6, ()))
    assert m.dense(rm) == ((1, 5, 9, 6), (0, 0, 0, 0, 0, 0, -1))
    # kappa(k,c): b = -2 n^2 (n+2k)(n+c)
    kap10 = ((3, 7, 3), (-2, 2, (0, 2)))
    assert m.dense(kap10)[1] == (0, 0, 0, -4, -2), m.dense(kap10)[1]
    # one B-shift away: kappa(1,0) -> kappa(1,1) is factors (0,2)->(1,2)
    shifted = m.tidy(((3, 7, 3), (-2, 2, (1, 2))))
    assert shifted[1][2] == (1, 2)
    rng = random.Random(0)
    c = rm
    for _ in range(400):
        c = m.mutate(c, rng)
        a, (coef, power, factors) = c
        assert len(a) <= 4 and len(factors) <= 4 and 0 <= power <= 6
    print("pcf mold v2 ok:", m.pretty(rm), "|", m.pretty(kap10))
