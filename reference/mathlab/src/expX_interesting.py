"""
expX_interesting.py — THE BRIDGE between the two walls (the genuine moonshot engine).

Wall 1 (rediscovery): target + efficiency budget -> CONVERGES to the known.
Wall 2 (pure open-endedness, expW): no target/objective, select on distinctness -> DIVERGES to a noise zoo.
The human-unknown lives in the band between: an INTRINSIC signal that yields MEANINGFUL structure with NO target.

This engine implements that signal as MDL/SOPHISTICATION surprise: a sequence is INTERESTING if WEAK models fail to
predict it (it's not trivially periodic / not a short linear recurrence / not a simple LFSR) BUT a STRONG general
compressor (zlib) compresses it well (it has rich self-similar structure -- it is NOT pseudo-random noise). That product
-- "weak models fail AND strong model succeeds" -- is exactly the band: it excludes the trivial (Wall-1-ish: simple
rules win) AND the noise (Wall-2-ish: nothing wins). It is target-free: nothing is told WHICH sequence to make; we
generate sequences from tiny rules and let the sophistication signal rank them.

VALIDATION QUESTION: does it surface RECOGNIZABLE meaningful sequences (Thue-Morse, Rudin-Shapiro, paperfolding,
period-doubling) UN-TARGETED, far above noise? If yes, the bridge-engine concept WORKS (intrinsic signal -> meaning).
Honest scope: binary sequences from small loop-free rules over the 2-kernel (a(n)=R(a(n>>1), bits of n)) and short
recurrences -- a rich but limited slice. Run: python expX_interesting.py
"""
from __future__ import annotations
import argparse, heapq, random, time, zlib


def lc_gf2(s):
    """Berlekamp-Massey linear complexity over GF(2): the order of the shortest LFSR generating s. Small => simple."""
    n = len(s); b = [1] + [0] * (n - 1); c = [1] + [0] * (n - 1); L = 0; m = -1
    for i in range(n):
        d = s[i]
        for j in range(1, L + 1):
            d ^= c[j] & s[i - j]
        if d:
            t = c[:]; sh = i - m
            for j in range(n - sh):
                c[j + sh] ^= b[j]
            if 2 * L <= i:
                L = i + 1 - L; m = i; b = t
    return L


def zratio(seq):
    # pack bits 8-per-byte so a RANDOM bit sequence is genuinely INCOMPRESSIBLE (one-byte-per-bit would fake-compress
    # every sequence to ~0.24 because 7/8 bits are always zero -- the artifact that made noise score 0.76).
    by = bytearray()
    for i in range(0, len(seq) - 7, 8):
        v = 0
        for k in range(8):
            v = (v << 1) | seq[i + k]
        by.append(v)
    if not by:
        return 1.0
    return len(zlib.compress(bytes(by), 9)) / len(by)


def small_period(seq, maxp=64):
    n = len(seq)
    for p in range(1, maxp + 1):
        if all(seq[i] == seq[i - p] for i in range(p, min(n, 4 * p + 40))) and all(seq[i] == seq[i - p] for i in range(p, n)):
            return p
    return None


# ---- tiny boolean rules (the "simple rule") ----
BOPS = ["XOR", "AND", "OR"]


def gen_ast(vars, rng, d):
    if d == 0 or rng.random() < 0.32:
        return ("var", rng.choice(vars)) if rng.random() < 0.85 else ("const", rng.randint(0, 1))
    if rng.random() < 0.18:
        return ("NOT", gen_ast(vars, rng, d - 1))
    return (rng.choice(BOPS), gen_ast(vars, rng, d - 1), gen_ast(vars, rng, d - 1))


def ev(ast, env):
    t = ast[0]
    if t == "var": return env[ast[1]]
    if t == "const": return ast[1]
    if t == "NOT": return 1 - ev(ast[1], env)
    a = ev(ast[1], env); b = ev(ast[2], env)
    return a ^ b if t == "XOR" else (a & b if t == "AND" else a | b)


def astr(ast):
    t = ast[0]
    if t == "var": return ast[1]
    if t == "const": return str(ast[1])
    if t == "NOT": return f"~{astr(ast[1])}"
    return f"({astr(ast[1])}{'^' if t=='XOR' else ('&' if t=='AND' else '|')}{astr(ast[2])})"


SCHEMES = {
    "auto": ["h", "b0", "b1", "b2"],          # a(n) = R(a(n>>1), low bits of n)   -- 2-automatic sequences
    "rec":  ["p1", "p2", "b0"],               # a(n) = R(a(n-1), a(n-2), n&1)       -- recurrences
    "nfun": ["b0", "b1", "b2", "b3"],         # a(n) = R(low bits of n)             -- digit functions
}


def make_seq(scheme, ast, L, init):
    a = [0] * L
    if scheme == "auto":
        a[0] = init & 1
        for n in range(1, L):
            a[n] = ev(ast, {"h": a[n >> 1], "b0": n & 1, "b1": (n >> 1) & 1, "b2": (n >> 2) & 1})
    elif scheme == "rec":
        a[0] = init & 1; a[1] = (init >> 1) & 1
        for n in range(2, L):
            a[n] = ev(ast, {"p1": a[n - 1], "p2": a[n - 2], "b0": n & 1})
    else:
        for n in range(L):
            a[n] = ev(ast, {"b0": n & 1, "b1": (n >> 1) & 1, "b2": (n >> 2) & 1, "b3": (n >> 3) & 1})
    return a


def known_sequences(L):
    tm = [bin(n).count("1") & 1 for n in range(L)]
    rs = [(bin(n & (n >> 1)).count("1")) & 1 for n in range(L)]              # Rudin-Shapiro
    pf = [((n >> (((n & -n).bit_length() - 1) + 1)) & 1) ^ 1 for n in range(1, L + 1)]  # regular paperfolding
    pd = [1 if (lambda k: (k.bit_length() - 1) % 2 == 0 if k else False)(n & -n) else 0 for n in range(1, L + 1)]  # period-doubling-ish
    return {"Thue-Morse": tm, "~Thue-Morse": [1 - x for x in tm], "Rudin-Shapiro": rs,
            "paperfolding": pf, "period-doubling": pd}


def label(seq, known):
    for name, k in known.items():
        if seq == k[:len(seq)]:
            return name
    return None


def score(seq, L, lc_prefix=256):
    if len(set(seq)) < 2:
        return None
    ones = sum(seq) / len(seq)
    if ones < 0.15 or ones > 0.85:                       # too sparse/dense -> near-trivial
        return None
    if small_period(seq, 64) is not None:                # trivially periodic
        return None
    LC = lc_gf2(seq[:lc_prefix])
    if LC < 20:                                          # short LFSR / simple linear recurrence -> simple (Wall-1-ish)
        return None
    zr = zratio(seq)
    # interesting = weak models FAIL (high LC, aperiodic) AND strong model SUCCEEDS (compressible structure)
    return (min(LC, lc_prefix//2)/(lc_prefix//2)) * (1 - zr), LC, zr


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=4000, help="random rules per scheme")
    ap.add_argument("--L", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    L = args.L; rng = random.Random(args.seed)
    known = known_sequences(L)
    print("BRIDGE ENGINE — target-free search ranked by SOPHISTICATION (weak models fail x strong compressor succeeds).")
    print(f"generating tiny binary rules; L={L}; ranking by intrinsic 'interestingness', NO target.\n")

    # NOISE BASELINE: a pseudo-random binary sequence -- the signal MUST reject this (that's "escaping the noise wall")
    noise = [rng.randint(0, 1) for _ in range(L)]
    ns = score(noise, L)
    print(f"  noise baseline (random bits): score = {('REJECTED' if ns is None else round(ns[0],3))}  "
          f"(LC {lc_gf2(noise[:256])}, zr {zratio(noise):.3f}) -- the engine must NOT rank noise highly.\n")

    seen = set(); top = []; cnt = 0; nkept = 0; t0 = time.time()    # memory-safe: hash-dedup + top-K heap
    for scheme, vrs in SCHEMES.items():
        for _ in range(args.per):
            ast = gen_ast(vrs, rng, rng.randint(2, 5))
            for init in (1, 2, 3):
                try:
                    seq = make_seq(scheme, ast, L, init)
                except Exception:
                    continue
                h = hash(bytes(seq))
                if h in seen:
                    continue
                seen.add(h)
                sc = score(seq, L)
                if sc is None:
                    continue
                nkept += 1; cnt += 1
                item = (sc[0], cnt, scheme, ast, init, sc[1], sc[2], bytes(seq))
                if len(top) < 400:
                    heapq.heappush(top, item)
                elif sc[0] > top[0][0]:
                    heapq.heapreplace(top, item)
    ranked = [(x[0], x[2], x[3], x[4], x[5], x[6], list(x[7])) for x in sorted(top, key=lambda x: -x[0])]
    print(f"  kept {nkept} distinct NON-trivial NON-simple sequences (of {len(seen)} examined)  [{time.time()-t0:.0f}s]\n")

    print("  === TOP 25 by intrinsic interestingness (un-targeted) ===")
    print("  score = 1 - zlib_ratio  (higher = more structured-yet-not-simply-predictable)\n")
    hits_known = 0
    for sc, scheme, ast, init, LC, zr, seq in ranked[:25]:
        lab = label(seq, known)
        if lab:
            hits_known += 1
        tag = f"  <<< {lab} (recognized!)" if lab else ""
        bits = "".join(map(str, seq[:64]))
        print(f"   score {sc:.3f} LC {LC:3d} zr {zr:.3f}  [{scheme}] a(n)={astr(ast)} init{init}{tag}")
        print(f"       {bits}...")
    print(f"\n  recognized meaningful sequences in the top 25: {hits_known} "
          f"(Thue-Morse / Rudin-Shapiro / paperfolding / period-doubling)")
    print("  => if the top is dominated by recognizable structured sequences, the SOPHISTICATION signal surfaced MEANING")
    print("     un-targeted (escaped the noise wall). Unfamiliar high-score lines are candidates to inspect.")
