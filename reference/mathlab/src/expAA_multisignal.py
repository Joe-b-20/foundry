"""
expAA_multisignal.py — MULTI-SIGNAL INTERSECTION (the bridge, made multi-dimensional) on the 256 elementary CAs.

WHY: expY validated ONE bridge signal (edge-of-chaos compression) on CA ground truth, but documented an HONEST
LIMITATION: the single signal CONFLATES class-4 (complex, e.g. 110) with the ADDITIVE/Sierpinski-fractal rules
(90,105,150) — both have intermediate compressibility, so 4c(1-c) ranks them together. expZ then showed a SECOND,
independent bridge family exists (algebraic self-consistency). This experiment asks the question the user's example #1
poses directly: if interestingness is MULTI-DIMENSIONAL, does INTERSECTING independent signals beat any single one —
specifically, does it RESOLVE expY's class-4-vs-additive conflation?

THREE INDEPENDENT signals per rule (all exact / cheap), each from a different layer:
  S1 STATISTICAL  — edge-of-chaos compression 4c(1-c) of the space-time pattern              [expY, reused verbatim]
  S2 ALGEBRAIC    — NONLINEARITY of the local Boolean rule (min Hamming dist to an affine fn). additive rules = 0. [NEW]
  S3 DYNAMICAL    — DAMAGE SPREADING: how a 1-cell perturbation grows (Lyapunov-like). chaos->saturates(~.5);
                    class4->bounded/glider; class1/2->dies.                                                       [NEW]
These are independent by construction: S2 is a property of the 8-bit RULE TABLE (not the orbit); S1/S3 are properties
of the ORBIT; S1 is statistical (compressibility), S3 is dynamical (sensitivity).

THESIS TO TEST (flagged uncertain up front, per RULES): each signal alone surfaces known structure imperfectly;
the INTERSECTION is sharper — additive rules fail S2 (they're linear), chaotic rules fail S3 (damage saturates),
and class-4 should survive ALL three. If the intersection is empty or still admits additive/chaos, that's an honest
negative about whether the signals are genuinely complementary. Ground truth = Wolfram classes (same as expY).

Run: python expAA_multisignal.py
"""
from __future__ import annotations
import argparse, time, zlib
import numpy as np

# ------- reused verbatim from expY (the validated S1 machinery) -------
def evolve(lut, radius, W, T, init):
    rows = np.empty((T, W), dtype=np.uint8)
    cur = init.copy(); k = 2 * radius + 1
    for t in range(T):
        rows[t] = cur
        idx = np.zeros(W, dtype=np.int64)
        for j, off in enumerate(range(radius, -radius - 1, -1)):
            idx |= np.roll(cur, off).astype(np.int64) << (k - 1 - j)
        cur = lut[idx]
    return rows

def comp_ratio(pattern):
    packed = np.packbits(pattern.reshape(-1))
    return len(zlib.compress(packed.tobytes(), 9)) / max(1, len(packed))

# ------- S2: NONLINEARITY of the elementary rule (exact; a property of the rule TABLE) -------
def nonlinearity(rule):
    """Min Hamming distance from the 3-input Boolean rule to ANY affine function c0 ^ c1 x0 ^ c2 x1 ^ c3 x2.
    NL=0  <=> the rule IS affine (the additive/linear ECAs: 90,150,60,105,102,...). NL>=1 <=> genuinely nonlinear.
    For 3 inputs NL in {0,1,2}. This is the classic cryptographic nonlinearity (Walsh) computed by brute force."""
    t = np.array([(rule >> x) & 1 for x in range(8)], dtype=np.int8)   # x = (n0<<2)|(n1<<1)|n2 (ordering irrelevant to NL)
    x0 = np.array([(x >> 2) & 1 for x in range(8)]); x1 = np.array([(x >> 1) & 1 for x in range(8)]); x2 = np.array([x & 1 for x in range(8)])
    best = 8
    for c0 in (0, 1):
        for c1 in (0, 1):
            for c2 in (0, 1):
                for c3 in (0, 1):
                    aff = (c0 ^ (c1 & x0) ^ (c2 & x1) ^ (c3 & x2)) & 1
                    best = min(best, int(np.sum(aff != t)))
    return best

# ------- S3: DAMAGE SPREADING (exact given seeds; a property of the orbit's SENSITIVITY) -------
def damage(lut, radius, W, T, seeds):
    """Evolve init and a copy with ONE cell flipped; return mean final fraction of cells that differ.
    chaos -> ~0.5 (decorrelates); class4 -> small/bounded (localized, gliders); class1/2 -> ~0 (heals)."""
    ds = []
    for sd in seeds:
        rng = np.random.default_rng(1000 + sd)
        a = rng.integers(0, 2, W, dtype=np.uint8); b = a.copy(); b[W // 2] ^= 1
        for _ in range(T):
            ka = np.zeros(W, np.int64); kb = np.zeros(W, np.int64); k = 2 * radius + 1
            for j, off in enumerate(range(radius, -radius - 1, -1)):
                ka |= np.roll(a, off).astype(np.int64) << (k - 1 - j)
                kb |= np.roll(b, off).astype(np.int64) << (k - 1 - j)
            a = lut[ka]; b = lut[kb]
        ds.append(float(np.mean(a != b)))
    return float(np.mean(ds))

def s1_edge(lut, radius, W, T, seeds):
    cs, dens = [], []
    for sd in seeds:
        rng = np.random.default_rng(sd)
        init = rng.integers(0, 2, W, dtype=np.uint8)
        pat = evolve(lut, radius, W, T, init)
        cs.append(comp_ratio(pat)); dens.append(float(pat.mean()))
    c = float(np.mean(cs)); dn = float(np.mean(dens))
    if dn < 0.03 or dn > 0.97:
        return None
    return 4 * c * (1 - c), c, dn

# Wolfram classes (same source as expY)
W_CLASS4 = {54, 106, 110, 124, 137, 147, 193}
W_CLASS3 = {18, 22, 30, 45, 60, 75, 86, 89, 90, 101, 102, 105, 122, 126, 129, 135, 146, 149, 150, 165, 169, 182, 195}
W_CLASS1 = {0, 8, 32, 40, 128, 136, 160, 168, 224, 234, 235, 238, 250, 251, 254, 255}
def wclass(r):
    return "4*" if r in W_CLASS4 else "3 " if r in W_CLASS3 else "1 " if r in W_CLASS1 else "2 "

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--W", type=int, default=256); ap.add_argument("--T", type=int, default=256)
    args = ap.parse_args(); W, T = args.W, args.T
    seeds = [1, 2, 3]
    print("MULTI-SIGNAL INTERSECTION on the 256 elementary CAs — does combining INDEPENDENT signals beat expY's single one?\n")

    # First: prove S2 is what I claim — print the affine (NL=0) rules and confirm class-4 are all nonlinear.
    affine = [r for r in range(256) if nonlinearity(r) == 0]
    print(f"  S2 sanity: {len(affine)} of 256 rules are AFFINE (NL=0) — the linear/additive ECAs.")
    print(f"    additive landmarks 90,150,60,105,102 affine? -> {[r in affine for r in (90,150,60,105,102)]}")
    print(f"    class-4 rules nonlinear (NL>=1)? -> {[(r, nonlinearity(r)) for r in sorted(W_CLASS4)]}\n")

    t0 = time.time(); rows = []
    for r in range(256):
        lut = np.array([(r >> i) & 1 for i in range(8)], dtype=np.uint8)
        a = s1_edge(lut, 1, W, T, seeds)
        if a is None:
            continue
        s1, c, dn = a
        rows.append({"r": r, "s1": s1, "c": c, "dn": dn, "nl": nonlinearity(r), "dmg": damage(lut, 1, W, T, seeds)})
    print(f"  computed all 3 signals for {len(rows)} gated rules  [{time.time()-t0:.0f}s]\n")

    by_s1 = sorted(rows, key=lambda d: -d["s1"])
    def precision_at(lst, k):
        top = lst[:k]; return sum(1 for d in top if d["r"] in W_CLASS4), k

    # (A) reproduce expY: rank by S1 alone, show additive contamination
    print("  === (A) S1 ALONE (reproduce expY): top 12 by edge-of-chaos ===")
    print("      rank  rule  class  S1     comp   NL  damage")
    for i, d in enumerate(by_s1[:12]):
        flag = "  <-- ADDITIVE (NL=0) contaminant" if d["nl"] == 0 and d["dn"] >= 0.03 else (" <-- class4" if d["r"] in W_CLASS4 else "")
        print(f"      {i:>3}   {d['r']:3d}   {wclass(d['r'])}   {d['s1']:.3f}  {d['c']:.3f}  {d['nl']}   {d['dmg']:.3f}{flag}")
    hit, k = precision_at(by_s1, 12); print(f"    class-4 precision@12 (S1 alone): {hit}/{len(W_CLASS4)} known class-4 in top-12\n")

    # (B) intersect S1 with S2 (nonlinear): drop the additive rules, re-rank
    nonlin = [d for d in by_s1 if d["nl"] >= 1]
    print("  === (B) S1 ∩ S2  (edge-of-chaos AND nonlinear NL>=1): top 12 ===")
    print("      rank  rule  class  S1     comp   NL  damage")
    for i, d in enumerate(nonlin[:12]):
        flag = "  <-- class4" if d["r"] in W_CLASS4 else ""
        print(f"      {i:>3}   {d['r']:3d}   {wclass(d['r'])}   {d['s1']:.3f}  {d['c']:.3f}  {d['nl']}   {d['dmg']:.3f}{flag}")
    hit, k = precision_at(nonlin, 12); print(f"    class-4 precision@12 (S1∩S2): {hit}/{len(W_CLASS4)} — additive contaminants removed?\n")

    # (C) S3 damage separates chaos: show class-3 chaotic vs class-4 damage fractions
    print("  === (C) S3 DAMAGE SPREADING: chaos saturates (~0.5), class-4 stays bounded ===")
    for label, group in (("class-4", sorted(W_CLASS4)), ("class-3 chaotic (30,45,90,150,18,22,126)", [30,45,90,150,18,22,126])):
        vals = [(r, next((d["dmg"] for d in rows if d["r"] == r), None)) for r in group]
        vals = [(r, v) for r, v in vals if v is not None]
        mean = np.mean([v for _, v in vals]) if vals else float("nan")
        print(f"    {label:42s} mean damage {mean:.3f}   per-rule {[(r, round(v,2)) for r,v in vals]}")
    print()

    # (D) FULL 3-signal intersection. KEY: every axis is an EDGE signal — interesting = INTERMEDIATE on all three.
    #     S1 edge-of-chaos (already a band via 4c(1-c)); S2 nonlinear (NL>=1); S3 damage in a BAND (not too high=chaos,
    #     not too low=trivial-heals). The one-sided damage<0.30 admitted both class-4 AND trivial healers; a band fixes it.
    s1_thresh = np.percentile([d["s1"] for d in rows], 75)   # top quartile of edge-of-chaos
    DLO, DHI = 0.10, 0.32                                     # damage band: exclude trivial-heal (<0.10) AND chaos (>0.32)
    survivors = [d for d in rows if d["nl"] >= 1 and d["s1"] >= s1_thresh and DLO <= d["dmg"] <= DHI]
    survivors.sort(key=lambda d: -d["s1"])
    print(f"  === (D) FULL INTERSECTION  S1(top quartile, >= {s1_thresh:.3f})  ∩  S2(nonlinear)  ∩  S3(damage in [{DLO},{DHI}]) ===")
    print(f"      {len(survivors)} rules survive all three. (How many are class-4 of {len(W_CLASS4)} known?)")
    print("      rule  class  S1     NL  damage")
    n4 = 0
    for d in survivors:
        if d["r"] in W_CLASS4: n4 += 1
        print(f"      {d['r']:3d}   {wclass(d['r'])}   {d['s1']:.3f}  {d['nl']}   {d['dmg']:.3f}" + ("  <-- class4" if d["r"] in W_CLASS4 else ""))
    print(f"\n    class-4 captured by full intersection: {n4}/{len(W_CLASS4)};  purity: {n4}/{len(survivors)} survivors are class-4.")
    print("    HONESTY on purity: my Wolfram label set is incomplete (7 class-4, 23 class-3, 16 class-1 explicit; the")
    print("    other 210 DEFAULT to '2', which is wrong for many). So non-class-4 survivors may include genuinely")
    print("    interesting UNLABELED rules — purity here is a LOWER bound, not a verdict that the extras are trivial.")
    print("\n  HONEST READ: ground truth (Wolfram class) is KNOWN; this tests whether MULTI-signal intersection is")
    print("  more discriminating than expY's single signal — it cannot find a human-UNKNOWN rule (all 256 are catalogued).")
