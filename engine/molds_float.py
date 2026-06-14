"""Float-program mold: straight-line programs over the 32-bit bits-typed
core subset, with a SEARCHABLE CONSTANT POOL — constants are genes (the
Quake magic number is a learned 32-bit integer; Newton coefficients are
learned float32 bit patterns). Mutations include bit flips on constants.

Candidate = (instrs, consts):
    instrs: tuple of (op, dst, a, b) over slots
    consts: tuple of 4 uint32 bit patterns, preloaded into slots 1..4
Slot map: 0 = input x (bits, also the output slot), 1..4 = constants,
5..8 = temps. Everything is 32 bits; F-ops view bits as float32.

Two execution paths, bit-exact by construction (add/sub/mul of float32
values are exact in float64 before the final float32 rounding, and the
runner rounds exactly like numpy):
    pour()   -> core Program for the trust path
    npfunc() -> vectorized numpy callable for search + exhaustive sweeps
"""

import numpy as np

from engine.core_lang import Instr, Program

OPS_F = ("FADD", "FSUB", "FMUL")
OPS_I = ("ADD32", "SUB32", "SHR32", "SHL32", "XOR32", "AND32", "OR32")
OPS_CVT = ("U2F", "F2U")
OPS_DIV = ("FDIV",)        # opt-in: rationals (saturating family)


class FloatProgMold:
    name = "float-program"

    def __init__(self, max_len=9, ops=OPS_F + OPS_I, n_const=4):
        self.max_len = max_len
        self.ops = ops
        self.N_CONST = n_const
        self.N_SLOTS = 1 + n_const + 4   # 0 x | consts | 4 temps
        self._writable = (0,) + tuple(range(1 + n_const, self.N_SLOTS))

    # --- shape ----------------------------------------------------------
    def _rand_instr(self, rng):
        return (rng.choice(self.ops), rng.choice(self._writable),
                rng.randrange(self.N_SLOTS), rng.randrange(self.N_SLOTS))

    def random_candidate(self, rng, length=6):
        consts = tuple(rng.getrandbits(32) for _ in range(self.N_CONST))
        instrs = tuple(self._rand_instr(rng)
                       for _ in range(min(length, self.max_len)))
        return self.tidy((instrs, consts))

    # --- edit moves -------------------------------------------------------
    def mutate(self, cand, rng):
        instrs, consts = list(cand[0]), list(cand[1])
        r = rng.random()
        if r < 0.45 and instrs:                     # tweak an instruction
            i = rng.randrange(len(instrs))
            op, dst, a, b = instrs[i]
            f = rng.randrange(4)
            if f == 0:
                op = rng.choice(self.ops)
            elif f == 1:
                dst = rng.choice(self._writable)    # consts are read-only
            elif f == 2:
                a = rng.randrange(self.N_SLOTS)
            else:
                b = rng.randrange(self.N_SLOTS)
            instrs[i] = (op, dst, a, b)
        elif r < 0.85:                              # mutate a constant gene
            i = rng.randrange(self.N_CONST)
            c = consts[i]
            m = rng.random()
            if m < 0.5:                             # flip one bit
                c ^= 1 << rng.randrange(32)
            elif m < 0.8:                           # small integer nudge
                c = (c + rng.choice((-4, -2, -1, 1, 2, 4))) & 0xFFFFFFFF
            else:                                   # nudge a byte-scale step
                c = (c + rng.choice((-1, 1)) * (1 << rng.randrange(8, 24))) \
                    & 0xFFFFFFFF
            consts[i] = c
        elif r < 0.93 and len(instrs) < self.max_len:
            instrs.insert(rng.randrange(len(instrs) + 1),
                          self._rand_instr(rng))
        elif len(instrs) > 1:
            instrs.pop(rng.randrange(len(instrs)))
        return self.tidy((tuple(instrs), tuple(consts)))

    def mutate_consts(self, cand, rng):
        """Constants-only move (polish phase): one gene, one nudge."""
        instrs, consts = cand[0], list(cand[1])
        i = rng.randrange(self.N_CONST)
        c = consts[i]
        m = rng.random()
        if m < 0.5:
            c ^= 1 << rng.randrange(32)
        elif m < 0.8:
            c = (c + rng.choice((-4, -2, -1, 1, 2, 4))) & 0xFFFFFFFF
        else:
            c = (c + rng.choice((-1, 1)) * (1 << rng.randrange(8, 24))) \
                & 0xFFFFFFFF
        consts[i] = c
        return (instrs, tuple(consts))

    # --- crossover: candidates are (instrs, consts) PAIRS, so the generic
    # sequence splice cannot apply; splice instruction lists and inherit
    # each constant gene from either parent ---------------------------------
    def crossover(self, a, b, rng):
        ia, ib = a[0], b[0]
        if ia and ib:
            instrs = (ia[: rng.randrange(1, len(ia) + 1)]
                      + ib[rng.randrange(len(ib)):])
        else:
            instrs = ia or ib
        consts = tuple(rng.choice((ca, cb))
                       for ca, cb in zip(a[1], b[1]))
        return self.tidy((instrs, consts))

    # --- tidy: drop instructions whose result is never used ----------------
    def tidy(self, cand):
        instrs, consts = cand[0][: self.max_len], cand[1]
        live = {0}
        keep = [False] * len(instrs)
        for i in range(len(instrs) - 1, -1, -1):
            op, dst, a, b = instrs[i]
            if dst in live:
                keep[i] = True
                live.discard(dst)
                live.update((a, b))
        return (tuple(t for t, k in zip(instrs, keep) if k), tuple(consts))

    # --- native cost ----------------------------------------------------------
    def native_cost(self, cand):
        instrs, _ = cand
        fam = {"f": 0, "i": 0}
        for (op, *_rest) in instrs:
            fam["f" if (op in OPS_F or op == "FDIV") else "i"] += 1
        return {"ops": len(instrs), "fops": fam["f"], "iops": fam["i"],
                "dl": len(instrs) + self.N_CONST}

    # --- pour: core Program (trust path) ------------------------------------
    def pour(self, cand) -> Program:
        instrs, consts = cand
        ins = [Instr("CONST", dst=1 + i, imm=int(c))
               for i, c in enumerate(consts)]
        ins += [Instr(op, dst=dst, a=a, b=b, tags=("approxop",))
                for (op, dst, a, b) in instrs]
        return Program(n_inputs=1, n_slots=self.N_SLOTS, instrs=ins,
                       meta={"mold": self.name}).validate()

    # --- vectorized path (search + exhaustive sweeps) ------------------------
    def npfunc(self, cand):
        instrs, consts = cand

        def run(x_bits):
            # temps zero-initialized, exactly like the core runner;
            # uint32 wraparound and float32 inf/nan are DEFINED semantics
            # here, so numpy's warnings are silenced for the whole body
            _es = np.errstate(all="ignore")
            _es.__enter__()
            m = [np.uint32(0)] * self.N_SLOTS
            m[0] = x_bits.astype(np.uint32)
            for i, c in enumerate(consts):
                m[1 + i] = np.uint32(c)
            for (op, dst, a, b) in instrs:
                if op in OPS_F or op == "FDIV":
                    fa = (np.asarray(m[a]).view(np.float32)
                          if hasattr(m[a], "view") else
                          np.uint32(m[a]).view(np.float32))
                    fb = (np.asarray(m[b]).view(np.float32)
                          if hasattr(m[b], "view") else
                          np.uint32(m[b]).view(np.float32))
                    with np.errstate(all="ignore"):
                        if op == "FADD":
                            r = fa + fb
                        elif op == "FSUB":
                            r = fa - fb
                        elif op == "FMUL":
                            r = fa * fb
                        else:   # FDIV: f64 quotient -> f32, matches runner
                            r = (fa.astype(np.float64)
                                 / fb.astype(np.float64)).astype(np.float32)
                    m[dst] = r.view(np.uint32) if hasattr(r, "view") \
                        else np.float32(r).view(np.uint32)
                elif op in ("U2F", "F2U"):
                    ua = np.asarray(m[a], dtype=np.uint32)
                    if op == "U2F":
                        r = ua.astype(np.float32).view(np.uint32)
                    else:
                        f = ua.view(np.float32).astype(np.float64)
                        okm = np.isfinite(f) & (np.abs(f) < 4294967296.0)
                        t = np.trunc(np.where(okm, f, 0.0))
                        r = ((t.astype(np.int64) & 0xFFFFFFFF)
                             .astype(np.uint32))
                    m[dst] = r
                else:
                    ua, ub = m[a], m[b]
                    if op == "ADD32":
                        r = np.uint32(ua) + np.uint32(ub) \
                            if not hasattr(ua, "shape") else ua + ub
                    elif op == "SUB32":
                        r = ua - ub
                    elif op == "SHR32":
                        r = ua >> (ub & np.uint32(31))
                    elif op == "SHL32":
                        r = ua << (ub & np.uint32(31))
                    elif op == "AND32":
                        r = ua & ub
                    elif op == "OR32":
                        r = ua | ub
                    else:
                        r = ua ^ ub
                    m[dst] = np.uint32(r) if not hasattr(r, "shape") else r
            out = m[0]
            if not hasattr(out, "view"):
                out = np.uint32(out)
            res = np.asarray(out).view(np.float32)
            if res.ndim == 0:      # x-independent program: broadcast anyway
                res = np.full(x_bits.shape, res, dtype=np.float32)
            _es.__exit__(None, None, None)
            return res
        return run

    # --- human view ------------------------------------------------------------
    def pretty(self, cand):
        instrs, consts = cand
        names = (["x"] + [f"c{i}" for i in range(self.N_CONST)]
                 + [f"t{i}" for i in range(4)])
        sym = {"FADD": "+f", "FSUB": "-f", "FMUL": "*f", "FDIV": "/f",
               "ADD32": "+i", "SUB32": "-i", "SHR32": ">>", "SHL32": "<<",
               "XOR32": "^", "AND32": "&", "OR32": "|"}
        body = "; ".join(
            f"{names[d]}={o.lower()}({names[a]})" if o in OPS_CVT
            else f"{names[d]}={names[a]}{sym[o]}{names[b]}"
            for (o, d, a, b) in instrs)
        cstr = " ".join(f"c{i}=0x{c:08X}" for i, c in enumerate(consts))
        return f"[{cstr}] {body}"


if __name__ == "__main__":
    import random
    import struct as _s
    from engine import runner as core_runner
    m = FloatProgMold()

    def fbits(v):
        return _s.unpack("<I", _s.pack("<f", v))[0]

    # the canonical Quake program: c0=magic, c1=1, c2=0.5 bits, c3=1.5 bits
    QUAKE = ((("SHR32", 5, 0, 2),       # t0 = x >> 1
              ("SUB32", 6, 1, 5),       # t1 = magic - t0   (the seed y)
              ("FMUL", 7, 0, 6),        # t2 = x * y
              ("FMUL", 7, 7, 6),        # t2 = x * y * y
              ("FMUL", 7, 7, 3),        # t2 = 0.5 * x * y * y
              ("FSUB", 7, 4, 7),        # t2 = 1.5 - t2
              ("FMUL", 0, 6, 7)),       # out = y * t2
             (0x5F3759DF, 1, fbits(0.5), fbits(1.5)))
    rng = random.Random(0)
    xs = np.array([fbits(v) for v in (0.0156, 0.7, 1.0, 2.5, 37.5, 144.0)],
                  dtype=np.uint32)
    vec = m.npfunc(m.tidy(QUAKE))(xs)
    prog = m.pour(m.tidy(QUAKE))
    for xb, vy in zip(xs, vec):
        out, cost = core_runner.run(prog, [int(xb)])
        scalar = np.uint32(out[0] & 0xFFFFFFFF).view(np.float32)
        assert scalar == vy, (hex(xb), scalar, vy)   # bit-exact paths
        true = 1.0 / np.sqrt(np.uint32(xb).view(np.float32).astype(np.float64))
        assert abs(scalar - true) / true < 2e-3      # the trick really works
    # mutation storm keeps shape legal
    c = m.random_candidate(rng, 6)
    for _ in range(400):
        c = m.mutate(c, rng)
        assert len(c[0]) <= 9 and len(c[1]) == 4
    print("float mold ok: quake program bit-exact across both paths; "
          f"cost={m.native_cost(m.tidy(QUAKE))}")
