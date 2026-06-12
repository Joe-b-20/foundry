"""Domain pack: hidden bit-mixers — the C1/C2 calibration pair (the roof).

Two targets behind ONE interface (the search only ever sees input/output
behavior on a corpus):

- planted (C1): a short program sampled from the SAME op set as the mold,
  filtered to be non-trivial. Reachable by construction; its length is the
  difficulty dial. PASS = the engine finds it (and the doctor stays quiet).
- keyed (C2): a 6-round ARX-style mixer with rotations and a 64-bit key —
  rotation is NOT a mold primitive and the key injects entropy every
  round. The parent's expFF measured this class un-learnable from outcome.
  PASS = the wall doctor recommends abandoning within budget.

Verification for C1 finds is exhaustive: all 2^16 input pairs at w=8
through the core runner (L1-exhaustive-in-bounds). Search-time scoring is
bit-agreement on the corpus (graded signal), with one-op baselines
defining "chance-plus" for the doctor.
"""

import random


def _rotl(v, r, w):
    return ((v << r) | (v >> (w - r))) & ((1 << w) - 1)


class BitMixerPack:
    name = "bitmixer"

    def __init__(self, target="planted", seed=0, w=8, planted_len=4,
                 corpus_size=128, heldout_size=256, plant=None):
        self.target_name = target
        self.w = w
        self.mask = (1 << w) - 1
        self.seed = seed
        rng = random.Random(seed * 7919 + 13)
        self.cost_rules = "primary: exact match; secondary: op count"
        self._planted = None
        if target == "planted":
            from engine.molds_bits import BitProgMold
            mold = BitProgMold()
            if plant is not None:                 # explicit plant (probes)
                cand = mold.tidy(tuple(plant))
                self._planted = cand
                self._f = lambda x, y, c=cand: self._interp(c, x, y)
            else:
                while True:
                    cand = mold.tidy(mold.random_candidate(rng, planted_len))
                    if len(cand) < max(2, planted_len - 1):
                        continue
                    f = lambda x, y, c=cand: self._interp(c, x, y)
                    if self._nontrivial(f, rng):
                        self._planted = cand
                        self._f = f
                        break
        elif target == "keyed":
            key = rng.getrandbits(64)
            def f(x, y, key=key, w=w):
                # 8 rounds, rotations and shifts (NOT mold primitives),
                # key entropy folded in every round
                M = (1 << w) - 1
                a, b = x & M, y & M
                for r in range(8):
                    k = (key >> (8 * (r % 8))) & 0xFF
                    a = (a + ((b * 0xA7) & M) + k) & M
                    a = _rotl(a, 1 + (r * 3) % (w - 1), w)
                    b = (b ^ ((a * 0x3B) & M) ^ _rotl(k & M, r % w, w)) & M
                    b = _rotl(b, 3, w)
                return (a ^ b) & M
            self._f = f
        else:
            raise KeyError(target)
        self.corpus = [(rng.getrandbits(w), rng.getrandbits(w))
                       for _ in range(corpus_size - 3)]
        self.corpus += [(0, 0), (self.mask, self.mask), (0, self.mask)]
        self.expected = [self._f(x, y) for x, y in self.corpus]
        # held-out set from a DISTINCT stream: the overfit detector.
        # Search never sees it; the doctor judges generalization on it.
        rng2 = random.Random(seed * 104729 + 57)
        self.heldout = [(rng2.getrandbits(w), rng2.getrandbits(w))
                        for _ in range(heldout_size)]
        self.heldout_expected = [self._f(x, y) for x, y in self.heldout]

    # --- the hidden function & helpers ------------------------------------
    def _interp(self, cand, x, y):
        m = [x, y, 0, 0]
        for (op, dst, a, b) in cand:
            if op == "XOR":
                m[dst] = m[a] ^ m[b]
            elif op == "AND":
                m[dst] = m[a] & m[b]
            elif op == "OR":
                m[dst] = m[a] | m[b]
            elif op == "ADD":
                m[dst] = m[a] + m[b]
            elif op == "MUL":
                m[dst] = m[a] * m[b]
        return m[0] & self.mask

    def _nontrivial(self, f, rng):
        pairs = [(rng.getrandbits(self.w), rng.getrandbits(self.w))
                 for _ in range(64)]
        outs = [f(x, y) for x, y in pairs]
        if len(set(outs)) < 4:
            return False                       # (nearly) constant
        for base in self._baseline_funcs().values():
            agree = sum(1 for (x, y), o in zip(pairs, outs)
                        if base(x, y) == o)
            if agree > 0.9 * len(pairs):
                return False                   # too close to a one-op fn
        dep_x = any(f(x ^ 1, y) != o for (x, y), o in zip(pairs, outs))
        dep_y = any(f(x, y ^ 1) != o for (x, y), o in zip(pairs, outs))
        return dep_x and dep_y

    def _baseline_funcs(self):
        M = self.mask
        return {"zero": lambda x, y: 0, "ones": lambda x, y: M,
                "x": lambda x, y: x, "y": lambda x, y: y,
                "x^y": lambda x, y: x ^ y, "x&y": lambda x, y: x & y,
                "x|y": lambda x, y: x | y,
                "x+y": lambda x, y: (x + y) & M}

    def baselines(self):
        """Bit-agreement of one-op functions on the HELD-OUT set: this is
        the 'chance-plus' level for generalization, which is what the
        doctor compares against (corpus scores can be inflated by
        overfitting)."""
        total = self.w * len(self.heldout)
        out = {}
        for name, fn in self._baseline_funcs().items():
            bits = sum(self.w - bin((fn(x, y) ^ e) & self.mask).count("1")
                       for (x, y), e in zip(self.heldout,
                                            self.heldout_expected))
            out[name] = bits / total
        return out

    def heldout_frac(self, cand):
        """Bit-agreement of a candidate on the held-out set (search never
        optimizes this; the doctor reads it)."""
        total = self.w * len(self.heldout)
        bits = sum(self.w - bin((self._interp(cand, x, y) ^ e)
                                & self.mask).count("1")
                   for (x, y), e in zip(self.heldout, self.heldout_expected))
        return bits / total

    # --- judge contract -----------------------------------------------------
    def gate1(self, mold, tidy):
        total = self.w * len(self.corpus)
        bits = 0
        for (x, y), e in zip(self.corpus, self.expected):
            bits += self.w - bin((self._interp(tidy, x, y) ^ e)
                                 & self.mask).count("1")
        cost = mold.native_cost(tidy)
        exact = int(bits == total and len(tidy) > 0)
        return ((exact, bits, -cost["ops"], -cost["dl"]), cost)

    def verify_trusted(self, mold, cand):
        """Exhaustive: all 2^(2w) input pairs through the CORE RUNNER."""
        from engine import runner as core_runner
        tidy = mold.tidy(cand)
        prog = mold.pour(tidy)
        n = 0
        for x in range(1 << self.w):
            for y in range(1 << self.w):
                out, _ = core_runner.run(prog, [x, y])
                if (out[0] & self.mask) != self._f(x, y):
                    return False, {"checked": n, "failed_on": [x, y]}
                n += 1
        return True, {"certificate": {
            "level": "L1-exhaustive-in-bounds",
            "claim": f"equals the hidden target on all {n} input pairs "
                     f"(w={self.w})",
            "evidence": "exhaustive core-runner re-execution"}}

    def reveal(self):
        """Post-exam only: what was planted (None for keyed)."""
        return self._planted


if __name__ == "__main__":
    from engine.molds_bits import BitProgMold
    mold = BitProgMold()
    p1 = BitMixerPack("planted", seed=0)
    sc, cost = p1.gate1(mold, p1._planted)
    assert sc[0] == 1, sc                      # planted scores exact on corpus
    ok, det = p1.verify_trusted(mold, p1._planted)
    assert ok and det["certificate"]["level"].startswith("L1"), det
    bl = p1.baselines()
    assert all(v < 1.0 for v in bl.values())
    p2 = BitMixerPack("keyed", seed=0)
    sc2, _ = p2.gate1(mold, p1._planted)       # planted vs keyed: not exact
    assert sc2[0] == 0
    assert p2.reveal() is None
    print(f"bitmixer pack ok: planted len={len(p1._planted)} verified "
          f"exhaustively; baselines max={max(bl.values()):.3f}")
