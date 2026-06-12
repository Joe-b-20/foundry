"""
expU_superopt.py — SUPEROPTIMIZATION: discover branchless straight-line bit-trick programs from OUTCOME ALONE.

A totally different modality for the project (no neural net, no loops): exhaustive bottom-up SEARCH over a low-level
word-op set for the SHORTEST straight-line program computing a target function of two words a,b -- found purely from
its input/output behaviour (the "outcome" = the exact truth table), no algorithm given.

Two project-signature twists:
  * EXACT verification is EXHAUSTIVE -> a PROOF: on w-bit words we check ALL 2^(2w) input pairs.
  * WIDTH-GENERALIZATION is the "real algorithm vs lookup" test: a program discovered on small words (w=3) counts
    only if it stays EXACTLY correct at w=4,5,6,8 (exhaustive) and w=12,16,32 (sampled). A bit-trick that holds
    across word size is a real structural procedure, not a w=3 coincidence (the test has teeth -- it REJECTS the
    overflow-prone "simple" min/max sign trick, which works at w=3 but fails wider).

Targets = famous branchless "Hacker's Delight" gems -- procedures humans rarely write by hand (the carry-save flavor:
how hardware/experts do it). Does outcome-only search REDISCOVER them, and is anything surprising?
Run: python expU_superopt.py
"""
from __future__ import annotations
import numpy as np
import time

UNARY = ["NOT", "NEG", "SHL", "SHR", "ASR", "SMEAR"]      # SMEAR = broadcast sign bit (arith shift by w-1)
BINARY = ["AND", "OR", "XOR", "ANDN", "ADD", "SUB"]


def un(name, x, mask, w):
    if name == "NOT": return (~x) & mask
    if name == "NEG": return (-x) & mask
    if name == "SHL": return (x << 1) & mask
    if name == "SHR": return x >> 1
    if name == "ASR":
        s = (x >> (w - 1)) & 1
        return ((x >> 1) | (s << (w - 1))) & mask
    if name == "SMEAR":
        return (((x >> (w - 1)) & 1) * mask)               # all-ones if sign set, else 0 (width-general)
    raise ValueError(name)


def bi(name, x, y, mask, w):
    if name == "AND": return x & y
    if name == "OR": return x | y
    if name == "XOR": return x ^ y
    if name == "ANDN": return x & ((~y) & mask)
    if name == "ADD": return (x + y) & mask
    if name == "SUB": return (x - y) & mask
    raise ValueError(name)


def ev(ast, a, b, mask, w):
    t = ast[0]
    if t == "base":
        nm = ast[1]
        if nm == "a": return a
        if nm == "b": return b
        z = np.zeros_like(a)
        return {"0": z, "1": z + 1, "M1": z + mask, "S": z + (1 << (w - 1))}[nm]
    if len(ast) == 2:
        return un(t, ev(ast[1], a, b, mask, w), mask, w)
    return bi(t, ev(ast[1], a, b, mask, w), ev(ast[2], a, b, mask, w), mask, w)


def ast_str(ast):
    t = ast[0]
    if t == "base": return ast[1]
    if len(ast) == 2: return f"{t}({ast_str(ast[1])})"
    return f"{t}({ast_str(ast[1])},{ast_str(ast[2])})"


def ast_size(ast):
    return 0 if ast[0] == "base" else 1 + sum(ast_size(c) for c in ast[1:])


def targets_for(a, b, mask, w):
    A = a.astype(np.int64); B = b.astype(np.int64)                                 # signed math in wide ints,
    sa = A - (mask + 1) * (A >> (w - 1)); sb = B - (mask + 1) * (B >> (w - 1))      # then cast back to a.dtype
    out = {
        "avg_floor  (a+b)>>1 no-overflow": (A + B) >> 1,
        "avg_ceil   (a+b+1)>>1":           (A + B + 1) >> 1,
        "abs(a) signed":                   np.abs(sa),
        "lowbit a&(-a)":                   A & ((-A) & mask),
        "smin(a,b) signed":                np.minimum(sa, sb),
        "smax(a,b) signed":                np.maximum(sa, sb),
    }
    return {k: (v & mask).astype(a.dtype) for k, v in out.items()}


def search(w=3, maxsize=5, hard_cap=4000, cur_cap=1_200_000, verbose=True):
    mask = (1 << w) - 1
    dt = np.uint8 if w <= 8 else np.int64                      # compact functions -> 8x less memory/faster hashing
    vals = np.arange(1 << w, dtype=dt)
    a = np.repeat(vals, 1 << w); b = np.tile(vals, 1 << w)
    base = [(ev(("base", nm), a, b, mask, w), ("base", nm)) for nm in ("a", "b", "0", "1", "M1", "S")]
    tgt_keys = {name: t.tobytes() for name, t in targets_for(a, b, mask, w).items()}

    seen = {}; buckets = [[]]
    for arr, ast in base:
        k = arr.tobytes()
        if k not in seen:
            seen[k] = ast; buckets[0].append((arr, ast))
    found = {}

    def check():
        for name, key in tgt_keys.items():
            if name not in found and key in seen:
                found[name] = seen[key]
    check()
    t0 = time.time()
    for s in range(1, maxsize + 1):
        cur = {}

        def add(arr, ast):
            k = arr.tobytes()
            if k not in seen and k not in cur:
                cur[k] = (arr, ast)
        for arr, ast in buckets[s - 1]:
            for op in UNARY:
                add(un(op, arr, mask, w), (op, ast))
        for i in range(0, s):
            if len(cur) >= cur_cap:
                break
            j = s - 1 - i
            Bi, Bj = buckets[i], buckets[j]
            if not Bi or not Bj:
                continue
            if len(Bi) <= len(Bj):                                       # vectorize over the larger bucket
                M = np.stack([x for x, _ in Bj]); asts = [a2 for _, a2 in Bj]
                for arr1, ast1 in Bi:
                    if len(cur) >= cur_cap:
                        break
                    for op in BINARY:
                        R = bi(op, arr1[None, :], M, mask, w)
                        uq, idx = np.unique(R, axis=0, return_index=True)
                        for row, ii in zip(uq, idx):
                            add(row, (op, ast1, asts[ii]))
            else:
                M = np.stack([x for x, _ in Bi]); asts = [a1 for _, a1 in Bi]
                for arr2, ast2 in Bj:
                    if len(cur) >= cur_cap:
                        break
                    for op in BINARY:
                        R = bi(op, M, arr2[None, :], mask, w)
                        uq, idx = np.unique(R, axis=0, return_index=True)
                        for row, ii in zip(uq, idx):
                            add(row, (op, asts[ii], ast2))
        for k, (arr, ast) in cur.items():
            seen[k] = ast
        buckets.append(list(cur.values())[:hard_cap])
        check()
        if verbose:
            print(f"  size {s}: {len(cur):7d} new functions  (total {len(seen)})  "
                  f"found {len(found)}/{len(tgt_keys)}  [{time.time()-t0:.0f}s]")
        if len(found) == len(tgt_keys):
            break
    return found, tgt_keys


def width_general(ast, name, widths=(4, 5, 6, 8, 12, 16, 32)):
    for w in widths:
        mask = (1 << w) - 1
        if w <= 8:
            vals = np.arange(1 << w, dtype=np.int64)
            a = np.repeat(vals, 1 << w); b = np.tile(vals, 1 << w)
        else:
            rng = np.random.default_rng(w)
            a = rng.integers(0, 1 << w, 40000, dtype=np.int64); b = rng.integers(0, 1 << w, 40000, dtype=np.int64)
        true = targets_for(a, b, mask, w)[name]
        if not np.array_equal(ev(ast, a, b, mask, w) & mask, true & mask):
            return False, w
    return True, None


if __name__ == "__main__":
    print("SUPEROPTIMIZATION — discover branchless bit-trick programs from OUTCOME ALONE (exhaustive = proof).")
    print("search word width w=3 (all 64 input pairs); accept only WIDTH-GENERAL programs (proven w=4..8, sampled to 32).\n")
    found, tgt_keys = search(w=3, maxsize=5, hard_cap=200000, cur_cap=2_500_000)

    print("\n  === DISCOVERED PROGRAMS (shortest found from outcome) ===")
    for name in tgt_keys:
        if name not in found:
            print(f"  {name:33s}: NOT FOUND within search budget"); continue
        ast = found[name]; ok, badw = width_general(ast, name)
        tag = "WIDTH-GENERAL real bit-trick (proven w<=8, sampled to 32)" if ok else f"w=3 COINCIDENCE -> fails at w={badw} (correctly rejected)"
        print(f"  {name:33s}: [{ast_size(ast)} ops] {ast_str(ast)}")
        print(f"  {'':33s}   -> {tag}")
