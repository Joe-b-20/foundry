"""
expY_ca.py — push the SOPHISTICATION BRIDGE into a richer, CANONICAL object space: cellular automata.

expX validated the bridge (intrinsic sophistication signal escapes both walls) on binary sequences, but my auto-labeler
was too weak to confirm it found the "known interesting" objects. Cellular automata fix that: "simple rule -> rich
behavior" is their defining drama, and Wolfram's classification is GROUND TRUTH (class 4 = complex/interesting: Rule 110
is Turing-complete; class 3 = chaotic; class 1/2 = trivial). The 256 elementary CAs are FULLY ENUMERABLE, so I can test
RIGOROUSLY whether the bridge signal -- with NO target -- ranks the class-4 rules at the top.

Signal (same as expX): SOPHISTICATION = temporal COMPLEXITY (the center column is NOT simple/periodic: high Berlekamp-
Massey linear complexity) x STRUCTURE (the space-time pattern is COMPRESSIBLE by zlib, i.e. not chaotic noise). Class 4
has BOTH (complex AND structured); class 3 is complex but INCOMPRESSIBLE (noise); class 1/2 are simple. So the product
should isolate class 4 -- un-targeted. Part 2 then probes a LARGER, un-catalogued rule space (radius-2, 2^32 rules) as a
genuine (if unprovable) novelty hunt. Run: python expY_ca.py
"""
from __future__ import annotations
import argparse, random, time, zlib
import numpy as np


def lc_gf2(s):
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


def evolve(lut, radius, W, T, init):
    """Evolve a 1D CA (periodic boundary). lut: np.uint8 array of length 2^(2*radius+1). Returns T x W uint8."""
    rows = np.empty((T, W), dtype=np.uint8)
    cur = init.copy()
    k = 2 * radius + 1
    for t in range(T):
        rows[t] = cur
        idx = np.zeros(W, dtype=np.int64)
        for j, off in enumerate(range(radius, -radius - 1, -1)):       # MSB = leftmost neighbor
            idx |= np.roll(cur, off).astype(np.int64) << (k - 1 - j)
        cur = lut[idx]
    return rows


def comp_ratio(pattern):
    packed = np.packbits(pattern.reshape(-1))                          # proper 8-bits/byte packing (noise -> incompressible)
    return len(zlib.compress(packed.tobytes(), 9)) / max(1, len(packed))


def analyze(lut, radius, W, T, seeds):
    cs, lcs, dens = [], [], []
    for sd in seeds:
        rng = np.random.default_rng(sd)
        init = rng.integers(0, 2, W, dtype=np.uint8)
        pat = evolve(lut, radius, W, T, init)
        cs.append(comp_ratio(pat))
        lcs.append(lc_gf2(pat[:, W // 2].tolist()))                    # temporal complexity of the center column
        dens.append(float(pat.mean()))
    c = float(np.mean(cs)); lc = float(np.mean(lcs)); dn = float(np.mean(dens))
    # gate: not uniform/near-dead
    if dn < 0.03 or dn > 0.97:
        return None
    # EDGE-OF-CHAOS: complexity peaks at INTERMEDIATE compressibility. class1/2 too compressible (c->0),
    # class3 incompressible noise (c->1); class4 in between. score = 4 c (1-c) peaks at c=0.5. (LC term dropped:
    # the center-column LC is contaminated by the random IC -- a pure shift copies the random IC down the column.)
    return 4 * c * (1 - c), c, lc, dn


# Wolfram classes for elementary rules (widely-cited; class 4 = complex/interesting).
W_CLASS4 = {54, 106, 110, 124, 137, 147, 193}
W_CLASS3 = {18, 22, 30, 45, 60, 75, 86, 89, 90, 101, 102, 105, 122, 126, 129, 135, 146, 149, 150, 165, 169, 182, 195}
W_CLASS1 = {0, 8, 32, 40, 128, 136, 160, 168, 224, 234, 235, 238, 250, 251, 254, 255}


def wclass(r):
    if r in W_CLASS4: return "4*"          # complex (interesting)
    if r in W_CLASS3: return "3 "          # chaotic
    if r in W_CLASS1: return "1 "          # uniform
    return "2 "                            # periodic (default)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--W", type=int, default=256)
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--sample2", type=int, default=20000, help="radius-2 rules to sample in Part 2")
    args = ap.parse_args()
    W, T = args.W, args.T
    seeds = [1, 2, 3]
    print("SOPHISTICATION BRIDGE on CELLULAR AUTOMATA — target-free; does it find the class-4 (complex) rules?\n")

    # noise baseline: a pattern of random bits must be REJECTED (incompressible -> low score)
    rng = np.random.default_rng(0)
    noise = rng.integers(0, 2, (T, W), dtype=np.uint8)
    print(f"  noise baseline: comp_ratio={comp_ratio(noise):.3f} (incompressible) -> structure factor (1-c)~0 -> rejected.\n")

    t0 = time.time()
    res = []
    for r in range(256):
        a = analyze(np.array([(r >> i) & 1 for i in range(8)], dtype=np.uint8), 1, W, T, seeds)
        if a is not None:
            res.append((a[0], r, a[1], a[2], a[3]))
    res.sort(key=lambda x: -x[0])
    print(f"  === Part 1: all 256 elementary CAs ranked by sophistication (no target)  [{time.time()-t0:.0f}s] ===")
    print(f"  {len(res)} rules passed the non-trivial gate. TOP 20 (score, rule, class, comp, LC, density):")
    top_classes = []
    for sc, r, c, lc, dn in res[:20]:
        top_classes.append(wclass(r).strip())
        star = "  <<< CLASS 4 (complex!)" if r in W_CLASS4 else ""
        print(f"     score {sc:.3f}  rule {r:3d}  class {wclass(r)}  comp {c:.3f}  LC {int(lc):3d}  dens {dn:.2f}{star}")
    print(f"\n  where do the famous CLASS-4 rules rank? (out of {len(res)} gated):")
    rank = {r: i for i, (_, r, *_) in enumerate(res)}
    for r in sorted(W_CLASS4):
        print(f"     rule {r:3d} (class 4): rank {rank.get(r, 'GATED-OUT'):>3}" + (f" / {len(res)}" if r in rank else ""))
    print("  for contrast, famous CLASS-3 chaotic rules (should rank LOWER -- complex but incompressible noise):")
    for r in (30, 90, 18, 45, 105, 150):
        print(f"     rule {r:3d} (class 3): rank {rank.get(r, 'GATED-OUT'):>3}" + (f" / {len(res)}" if r in rank else ""))
    c4_in_top20 = sum(1 for c in top_classes if c == "4*")
    print(f"\n  => class-4 rules in the TOP 20: {c4_in_top20} of {len(W_CLASS4)} known class-4 rules.")

    # Part 2: probe a LARGER, un-catalogued space (radius-2, 2^32 rules) -- genuine novelty hunt
    print(f"\n  === Part 2: radius-2 CAs (2^32 rules, un-catalogued) -- sampling {args.sample2} at random, no target ===")
    rng2 = random.Random(7); res2 = []; t1 = time.time()
    for _ in range(args.sample2):
        rule = rng2.getrandbits(32)
        lut = np.array([(rule >> i) & 1 for i in range(32)], dtype=np.uint8)
        a = analyze(lut, 2, W, T, [1])                                  # 1 seed for speed in the scan
        if a is not None:
            res2.append((a[0], rule, a[1], a[2], a[3]))
    res2.sort(key=lambda x: -x[0])
    print(f"  sampled {args.sample2}, {len(res2)} passed the gate  [{time.time()-t1:.0f}s]. TOP 12 sophistication candidates:")
    for sc, rule, c, lc, dn in res2[:12]:
        print(f"     score {sc:.3f}  rule 0x{rule:08X}  comp {c:.3f}  LC {int(lc):3d}  dens {dn:.2f}")
    print("\n  NOTE: Part 1 is the rigorous test (known ground truth). Part 2's top rules are 'interesting'-by-the-signal")
    print("  in an under-catalogued space -- candidates, not claimed novel (interestingness != provably-unknown).")
