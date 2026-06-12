"""Domain pack: fast inverse square root, float32 (portfolio domain #1,
the bit-trick capital of numerical approximants).

Problem: a straight-line program (<= 9 ops over the 32-bit core subset,
3 float ops max cost-class fmul/fadd, integer bit ops, 4 constant genes)
approximating 1/sqrt(x) on the DECLARED SCOPE: every positive normal
float32 with exponent in [lo_exp, hi_exp) — a finite set, so verification
is EXHAUSTIVE (this domain's 0/1 principle).

Metric (predeclared): maximum relative error vs the float64-computed
reference over the scope. Reference noise (~1e-16) is negligible at the
1e-3..1e-4 claim level; ULP-exact correct-rounding hunts are out of scope
for this pack version and would need a correctly-rounded oracle.

gate1 = max rel error on a fixed stratified sample (search-time).
verify_trusted = exhaustive sweep over the whole scope via the vectorized
evaluator, PLUS a bit-exactness cross-check of 4096 random inputs against
the core runner (the two paths must agree to the bit; add/sub/mul are
exact in float64 before the final float32 rounding, so they do).
"""

import random

import numpy as np


class RsqrtPack:
    name = "rsqrt"

    def __init__(self, lo_exp=-8, hi_exp=8, sample_per_octave=256,
                 sample_random=1024, seed=12345):
        assert -126 <= lo_exp < hi_exp <= 127
        self.lo_exp, self.hi_exp = lo_exp, hi_exp
        self.octaves = list(range(lo_exp, hi_exp))
        self.scope_size = (1 << 23) * len(self.octaves)
        self.cost_rules = ("primary: max relative error over the declared "
                           "scope (exhaustive); secondary: op count, dl; "
                           "op budget <= 9; no FDIV")
        rng = random.Random(seed)
        bits = []
        for e in self.octaves:
            base = (e + 127) << 23
            mant = np.linspace(0, (1 << 23) - 1, sample_per_octave,
                               dtype=np.int64)
            bits.append(base + mant)
        extra = [((rng.randrange(lo_exp, hi_exp) + 127) << 23)
                 + rng.getrandbits(23) for _ in range(sample_random)]
        self.sample_bits = np.unique(
            np.concatenate([np.concatenate(bits),
                            np.array(extra, dtype=np.int64)])
        ).astype(np.uint32)
        self.sample_truth = self._truth(self.sample_bits)

    @staticmethod
    def _truth(bits_u32):
        x = bits_u32.view(np.float32).astype(np.float64)
        return 1.0 / np.sqrt(x)

    def _max_rel(self, mold, cand, bits_u32, truth):
        with np.errstate(all="ignore"):
            y = mold.npfunc(cand)(bits_u32).astype(np.float64)
            if not np.all(np.isfinite(y)):
                return float("inf"), None
            rel = np.abs(y - truth) / truth
            i = int(np.argmax(rel))
            return float(rel[i]), int(bits_u32[i])

    # --- judge contract ---------------------------------------------------
    def gate1(self, mold, tidy):
        """SEARCH fitness, not the claim metric: clipped mean-log relative
        error. Raw max-rel-error has a huge 'output a tiny constant'
        attractor (error saturates near 1.0) while any exploration of
        bit-trick territory scores astronomically worse — the first hunt
        collapsed into that trap. The claim metric stays exhaustive
        max-rel-error (verify_trusted); shaping is search-internal."""
        cost = mold.native_cost(tidy)
        with np.errstate(all="ignore"):
            y = mold.npfunc(tidy)(self.sample_bits).astype(np.float64)
            rel = np.abs(y - self.sample_truth) / self.sample_truth
            rel = np.where(np.isfinite(rel), rel, 1e6)
            rel = np.minimum(rel, 1e6)
            shaped = -float(np.mean(np.log1p(rel)))
        return ((shaped, -cost["ops"], -cost["dl"], 0.0), cost)

    def sample_max_rel(self, mold, tidy):
        """The true (claim) metric on the search sample — used by the
        constants-only polish phase and progress reporting."""
        err, _ = self._max_rel(mold, tidy, self.sample_bits,
                               self.sample_truth)
        return err

    def dense_max_rel(self, mold, tidy, per_octave=4096):
        """Denser (but still sampled) max-rel for final tie-breaking
        between near-optimal constants before the exhaustive verify."""
        chunks = []
        for e in self.octaves:
            base = (e + 127) << 23
            mant = np.linspace(0, (1 << 23) - 1, per_octave, dtype=np.int64)
            chunks.append(base + mant)
        bits = np.concatenate(chunks).astype(np.uint32)
        err, _ = self._max_rel(mold, tidy, bits, self._truth(bits))
        return err

    def verify_trusted(self, mold, cand):
        tidy = mold.tidy(cand)
        worst, worst_at = 0.0, None
        chunk = 1 << 21
        for e in self.octaves:
            base = (e + 127) << 23
            for off in range(0, 1 << 23, chunk):
                bits = (np.arange(off, off + chunk, dtype=np.int64)
                        + base).astype(np.uint32)
                err, at = self._max_rel(mold, tidy, bits, self._truth(bits))
                if err == float("inf"):
                    return False, {"reason": "non-finite output in scope",
                                   "octave": e}
                if err > worst:
                    worst, worst_at = err, at
        # cross-check: vectorized path == core runner, bit for bit
        from engine import runner as core_runner
        rng = random.Random(999)
        xs = np.array([((rng.randrange(self.lo_exp, self.hi_exp) + 127) << 23)
                       + rng.getrandbits(23) for _ in range(4096)],
                      dtype=np.uint32)
        vec = mold.npfunc(tidy)(xs).view(np.uint32)
        prog = mold.pour(tidy)
        for xb, vb in zip(xs[:: 16], vec[:: 16]):   # 256 through the runner
            out, _ = core_runner.run(prog, [int(xb)])
            if (out[0] & 0xFFFFFFFF) != int(vb):
                return False, {"reason": "core-runner cross-check FAILED",
                               "input": hex(int(xb))}
        return True, {"certificate": {
            "level": "L1-exhaustive-in-bounds",
            "claim": f"max relative error {worst:.6e} vs float64 reference "
                     f"over ALL {self.scope_size:,} float32 in "
                     f"[2^{self.lo_exp}, 2^{self.hi_exp})",
            "evidence": "exhaustive vectorized sweep; 256 random inputs "
                        "bit-exact against the core runner"},
            "max_rel_err": worst, "worst_at_bits": hex(worst_at)}


if __name__ == "__main__":
    from engine.molds_float import FloatProgMold
    import struct as _s

    def fbits(v):
        return _s.unpack("<I", _s.pack("<f", v))[0]
    mold = FloatProgMold()
    pack = RsqrtPack(lo_exp=-2, hi_exp=2)        # small scope for sanity
    QUAKE = ((("SHR32", 5, 0, 2), ("SUB32", 6, 1, 5), ("FMUL", 7, 0, 6),
              ("FMUL", 7, 7, 6), ("FMUL", 7, 7, 3), ("FSUB", 7, 4, 7),
              ("FMUL", 0, 6, 7)),
             (0x5F3759DF, 1, fbits(0.5), fbits(1.5)))
    sc, cost = pack.gate1(mold, mold.tidy(QUAKE))
    assert -sc[0] < 2e-3, sc
    ok, det = pack.verify_trusted(mold, QUAKE)
    assert ok and det["max_rel_err"] < 2e-3, det
    print(f"rsqrt pack ok: quake exhaustive max rel err "
          f"{det['max_rel_err']:.6e} on the sanity scope")
