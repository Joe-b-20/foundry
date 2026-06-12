"""
expW_openended.py — CHANGE THE SKELETON. Not "search for the minimal/correct program under a budget" (a rediscovery
engine: target + cost + exact-match => converges to the known optimum). Instead: OPEN-ENDED EXPLORATION with NO target,
NO cost function, NO correctness criterion. Candidates are judged ONLY by being BEHAVIORALLY DIFFERENT from everything
already found (the canonical anti-objective: novelty, not optimization). Nothing is told what to compute.

The system grows an ARCHIVE of distinct integer functions f(a,b) by randomly composing operations on functions already
in the archive (open-ended recombination). We then INSPECT what fell out un-targeted: which named operations did it
REINVENT without being asked, and what unfamiliar-but-structured functions appear in the tail. This is the structural
OPPOSITE of every prior experiment here (all of which fixed a target and optimized toward it).

Honest scope: this explores STRAIGHT-LINE (loop-free) integer/bit functions, so iterative procedures (gcd, sorting) are
OUTSIDE the space by construction -- a real limit, flagged. The point is the skeleton change (discovery without a target),
not a guaranteed moonshot. Run: python expW_openended.py
"""
from __future__ import annotations
import argparse, math, random, time

# probe inputs: small grid (behavior fingerprint) + a few LARGER pairs (so "scale-consistent" structure is visible)
PROBES = [(a, b) for a in range(8) for b in range(8)] + [(13, 8), (20, 7), (31, 12), (50, 17), (100, 30), (7, 41)]
NP = len(PROBES)
CAP = 10 ** 15                                  # behavior invalid if any |output| exceeds this (avoid runaway blowup)


def gd(x, y):                                   # guarded floor-div / mod (by 0 -> invalid)
    return None if y == 0 else x // y


def gm(x, y):
    return None if y == 0 else x % y


BIN = {
    "ADD": lambda x, y: x + y, "SUB": lambda x, y: x - y, "MUL": lambda x, y: x * y,
    "MIN": min, "MAX": max, "ADIF": lambda x, y: abs(x - y),
    "XOR": lambda x, y: x ^ y, "AND": lambda x, y: x & y, "OR": lambda x, y: x | y,
    "FDIV": gd, "MOD": gm,
}
UN = {
    "NEG": lambda x: -x, "ABS": abs, "INC": lambda x: x + 1, "DEC": lambda x: x - 1,
    "DBL": lambda x: 2 * x, "HALF": lambda x: x >> 1 if x >= 0 else -((-x) >> 1),
    "SQ": lambda x: x * x, "NOT": lambda x: ~x,
}


def combine_bin(op, b1, b2):
    f = BIN[op]; out = []
    for x, y in zip(b1, b2):
        try:
            v = f(x, y)
        except Exception:
            return None
        if v is None or abs(v) > CAP:
            return None
        out.append(v)
    return tuple(out)


def combine_un(op, b1):
    f = UN[op]; out = []
    for x in b1:
        try:
            v = f(x)
        except Exception:
            return None
        if v is None or abs(v) > CAP:
            return None
        out.append(v)
    return tuple(out)


def beh_of(ast):
    """Evaluate an AST on the probes (for display / re-derivation)."""
    t = ast[0]
    if t == "a": return tuple(a for a, b in PROBES)
    if t == "b": return tuple(b for a, b in PROBES)
    if t == "const": return tuple(ast[1] for _ in PROBES)
    if t in UN: return combine_un(t, beh_of(ast[1]))
    return combine_bin(t, beh_of(ast[1]), beh_of(ast[2]))


def astr(ast):
    t = ast[0]
    if t == "a": return "a"
    if t == "b": return "b"
    if t == "const": return str(ast[1])
    if t in UN: return f"{t}({astr(ast[1])})"
    return f"{t}({astr(ast[1])},{astr(ast[2])})"


def size(ast):
    if ast[0] in ("a", "b", "const"): return 0
    return 1 + sum(size(c) for c in ast[1:])


def explore(attempts=300000, cap_archive=20000, seed=0, verbose=True):
    rng = random.Random(seed)
    archive = {}                                # behavior_tuple -> ast  (DISTINCT behaviors; no target, no cost)
    szof = {}
    for ast in [("a",), ("b",), ("const", 0), ("const", 1), ("const", 2)]:
        bb = beh_of(ast); archive[bb] = ast; szof[bb] = 0
    keys = list(archive.keys())
    frontier = list(keys)                       # SMALL-size behaviors: keep building from simple parts (still target-free;
    t0 = time.time()                            # this is exploration COVERAGE, not an objective) so simple structure is reached
    binops, unops = list(BIN), list(UN)

    def add(nb, ast):
        if nb is not None and nb not in archive:
            archive[nb] = ast; keys.append(nb)
            s = size(ast); szof[nb] = s
            if s <= 4:
                frontier.append(nb)
    for it in range(attempts):
        pool = frontier if (frontier and rng.random() < 0.7) else keys
        if rng.random() < 0.35:                 # unary mutation
            b1 = rng.choice(pool); op = rng.choice(unops)
            add(combine_un(op, b1), (op, archive[b1]))
        else:                                   # binary recombination
            b1 = rng.choice(pool); b2 = rng.choice(pool); op = rng.choice(binops)
            add(combine_bin(op, b1, b2), (op, archive[b1], archive[b2]))
        if len(archive) >= cap_archive:
            break
        if verbose and it % 100000 == 0 and it:
            print(f"    it {it}: archive {len(archive)} (frontier {len(frontier)})  [{time.time()-t0:.0f}s]")
    return archive


# ---- named operations, to see what the un-targeted search REINVENTED on its own ----
def named_ops():
    import math as _m
    return {
        "a+b": lambda a, b: a + b, "a-b": lambda a, b: a - b, "b-a": lambda a, b: b - a,
        "|a-b|": lambda a, b: abs(a - b), "a*b": lambda a, b: a * b, "max(a,b)": max, "min(a,b)": min,
        "a XOR b": lambda a, b: a ^ b, "a AND b": lambda a, b: a & b, "a OR b": lambda a, b: a | b,
        "a^2": lambda a, b: a * a, "a^2+b^2": lambda a, b: a * a + b * b, "a^2-b^2": lambda a, b: a * a - b * b,
        "(a+b)^2": lambda a, b: (a + b) ** 2, "a*b+a+b": lambda a, b: a * b + a + b, "2a+b": lambda a, b: 2 * a + b,
        "a*(a+b)": lambda a, b: a * (a + b), "(a-b)^2": lambda a, b: (a - b) ** 2,
        "avg_floor (a+b)//2": lambda a, b: (a + b) // 2, "a&~b": lambda a, b: a & ~b,
        "max-min": lambda a, b: max(a, b) - min(a, b), "a+2b": lambda a, b: a + 2 * b,
        "gcd(a,b) [needs a loop!]": lambda a, b: _m.gcd(a, b), "a mod b": lambda a, b: (a % b) if b else a,
        "a//b": lambda a, b: (a // b) if b else a, "a*b-a-b": lambda a, b: a * b - a - b,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempts", type=int, default=400000)
    ap.add_argument("--cap", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    print("OPEN-ENDED EXPLORATION — no target, no cost, no correctness test. Selection = behavioral DISTINCTNESS only.")
    print(f"probes: {NP} input pairs (incl. large ones for scale-structure).\n")

    arch = explore(attempts=args.attempts, cap_archive=args.cap, seed=args.seed)
    print(f"\n  archive: {len(arch)} DISTINCT integer functions f(a,b), grown un-targeted.\n")

    nm = named_ops()
    found, missing = [], []
    for name, f in nm.items():
        try:
            beh = tuple(f(a, b) for a, b in PROBES)
        except Exception:
            beh = None
        if beh is not None and beh in arch:
            found.append((name, size(arch[beh]), astr(arch[beh])))
        else:
            missing.append(name)
    print(f"  === named operations the search REINVENTED un-targeted ({len(found)}/{len(nm)}) ===")
    for name, sz, prog in sorted(found, key=lambda x: x[1]):
        print(f"    {name:24s}  <- discovered as [{sz} ops] {prog}")
    print(f"  not found (outside this straight-line space or just not hit): {', '.join(missing)}")

    # a glimpse of the unnamed tail: structured functions (smallest programs not matching a named op)
    namedbehs = set()
    for name, f in nm.items():
        try:
            namedbehs.add(tuple(f(a, b) for a, b in PROBES))
        except Exception:
            pass
    tail = [(beh, ast) for beh, ast in arch.items() if beh not in namedbehs and size(ast) >= 2]
    tail.sort(key=lambda x: size(x[1]))
    print(f"\n  === a glimpse of the UNNAMED tail ({len(tail)} functions); smallest few, with values on (2,3),(5,2),(7,4): ===")
    for beh, ast in tail[:14]:
        i23 = PROBES.index((2, 3)); i52 = PROBES.index((5, 2)); i74 = PROBES.index((7, 4))
        print(f"    [{size(ast)} ops] {astr(ast):34s} -> f(2,3)={beh[i23]}, f(5,2)={beh[i52]}, f(7,4)={beh[i74]}")
    print("\n  NOTE: every function above emerged with NO target and NO objective -- pure open-ended recombination judged")
    print("  only by behavioral distinctness. The named ops it 'reinvented' were never asked for. Honest limit: this is a")
    print("  STRAIGHT-LINE space, so looped procedures (gcd, sort, Euclid) are unreachable here by construction.")
