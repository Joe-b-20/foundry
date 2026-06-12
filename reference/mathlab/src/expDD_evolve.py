"""
expDD_evolve.py — Is the BRIDGE signal OPTIMIZABLE (generative), not just SELECTIVE? And inspect the UNCATALOGUED space.

Every bridge experiment (expX/Y/Z, expAA) used the intrinsic signal to SELECT from a fixed sample/enumeration. expW
showed that DISTINCTNESS-driven open-ended search (no objective) DIVERGES to a noise zoo. The untested question that sits
exactly between them: can an INTRINSIC MULTI-SIGNAL score DRIVE a search — as a fitness function — to actively HUNT a huge
UNCATALOGUED space (radius-2 CAs, 2^32 rules) and PRODUCE genuinely complex objects? If a signal-driven search CONVERGES
to structure (where expW's objective-free one diverged to noise), the bridge is GENERATIVE, not merely discriminative — a
stronger property — and it's the honest way to probe the radius-2 space expY sampled but explicitly never INSPECTED.

Multi-signal fitness (the expAA axes, made smooth, target-free):
  F = S1_edge * NL_factor * dmg_edge   where
    S1_edge  = 4 c (1-c)                       (compression edge-of-chaos; c = zlib ratio of the space-time pattern)
    NL_factor= min(1, NL/4)                    (algebraic nonlinearity of the 32-bit rule; affine rules -> ~0)
    dmg_edge = 4 d (1-d)                        (damage spreading in a band; trivial d->0 and chaos d->1 both punished)
Three DRIVERS compared (same evolutionary loop, only the fitness differs): MULTI (F above), S1-ONLY (compression edge
alone, the expY signal), RANDOM (fitness = fixed random hash of the rule — a control for "evolution itself"). Then INSPECT
the best MULTI rule's dynamics concretely (compression, damage, transient/period, localized structure) — described, NOT
claimed novel (interestingness != provably-unknown; radius-2 rules are uncatalogued so I can only characterize them).
Run: python expDD_evolve.py
"""
from __future__ import annotations
import argparse, zlib
import numpy as np


def evolve(lut, radius, W, T, init):
    rows = np.empty((T, W), dtype=np.uint8); cur = init.copy(); k = 2 * radius + 1
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

def lut_of(rule, nbits):
    return np.array([(rule >> i) & 1 for i in range(1 << nbits)], dtype=np.uint8)

def nonlinearity32(rule):
    """NL of a radius-2 (5-input) Boolean rule: min Hamming distance of its 32-entry truth table to any of the 64 affine
    functions over 5 vars. Affine (additive/linear) rules -> 0."""
    n = 32
    t = np.array([(rule >> x) & 1 for x in range(n)], dtype=np.int8)
    xs = [np.array([(x >> b) & 1 for x in range(n)]) for b in range(5)]
    best = n
    for c0 in (0, 1):
        for mask in range(32):                       # 5 linear coeffs
            aff = np.full(n, c0)
            for b in range(5):
                if (mask >> b) & 1:
                    aff = aff ^ xs[b]
            best = min(best, int(np.sum(aff != t)))
            if best == 0:
                return 0
    return best

def damage(lut, radius, W, T, seed):
    rng = np.random.default_rng(1000 + seed)
    a = rng.integers(0, 2, W, dtype=np.uint8); b = a.copy(); b[W // 2] ^= 1
    k = 2 * radius + 1
    for _ in range(T):
        ka = np.zeros(W, np.int64); kb = np.zeros(W, np.int64)
        for j, off in enumerate(range(radius, -radius - 1, -1)):
            ka |= np.roll(a, off).astype(np.int64) << (k - 1 - j)
            kb |= np.roll(b, off).astype(np.int64) << (k - 1 - j)
        a = lut[ka]; b = lut[kb]
    return float(np.mean(a != b))

def signals(rule, W, T, seeds=(1, 2)):
    lut = lut_of(rule, 5)
    cs, dn = [], []
    for sd in seeds:
        rng = np.random.default_rng(sd)
        pat = evolve(lut, 2, W, T, rng.integers(0, 2, W, dtype=np.uint8))
        cs.append(comp_ratio(pat)); dn.append(float(pat.mean()))
    c = float(np.mean(cs)); d = damage(lut, 2, W, T, 7)
    dens = float(np.mean(dn))
    return c, d, dens

def fitness(rule, W, T, mode, rngseed_cache={}):
    c, d, dens = signals(rule, W, T)
    if dens < 0.02 or dens > 0.98:                  # dead/saturated -> reject
        return -1.0, (c, d, dens, 0)
    s1 = 4 * c * (1 - c)
    if mode == "s1":
        return s1, (c, d, dens, None)
    if mode == "random":
        # deterministic pseudo-random fitness from the rule bits (control: "does evolution itself find structure?")
        h = (rule * 2654435761) & 0xFFFFFFFF
        return (h / 0xFFFFFFFF), (c, d, dens, None)
    nl = nonlinearity32(rule)
    nlf = min(1.0, nl / 4.0)
    # damage BUMP centered on the class-4-like regime (~0.22 in expAA), punishing BOTH heal (d->0) and chaos (d->0.5).
    # (NOT 4d(1-d), which peaks at 0.5 = chaos — the bug the smoke test exposed.)
    dmg_fac = float(np.exp(-((d - 0.22) / 0.12) ** 2))
    return s1 * nlf * dmg_fac, (c, d, dens, nl)


def evolve_search(mode, W, T, pop=40, gens=25, seed=0):
    rng = np.random.RandomState(seed)
    population = [int(rng.randint(0, 1 << 30)) | (rng.randint(0, 4) << 30) for _ in range(pop)]
    scored = [(fitness(r, W, T, mode)[0], r) for r in population]
    for g in range(gens):
        scored.sort(key=lambda x: -x[0])
        elite = [r for _, r in scored[:max(2, pop // 5)]]
        children = list(elite)
        while len(children) < pop:
            parent = elite[rng.randint(0, len(elite))]
            nflip = rng.randint(1, 4)
            child = parent
            for _ in range(nflip):
                child ^= (1 << rng.randint(0, 32))
            children.append(int(child))
        scored = [(fitness(r, W, T, mode)[0], r) for r in children]
    scored.sort(key=lambda x: -x[0])
    return scored[0][1], scored[0][0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--W", type=int, default=128); ap.add_argument("--T", type=int, default=128)
    ap.add_argument("--pop", type=int, default=40); ap.add_argument("--gens", type=int, default=25)
    args = ap.parse_args()
    W, T = args.W, args.T
    print("IS THE BRIDGE SIGNAL OPTIMIZABLE? Intrinsic-multi-signal-DRIVEN evolution over UNCATALOGUED radius-2 CAs.\n")

    # noise/affine references so we can judge "did it find structure"
    print("  references (what scores look like):")
    for label, rule in [("all-0 (dead)", 0), ("affine XOR-ish 0x33333333", 0x33333333), ("random 0x9E3779B9", 0x9E3779B9)]:
        c, d, dens = signals(rule, W, T)
        print(f"    {label:28s} comp={c:.3f} damage={d:.3f} dens={dens:.2f} NL={nonlinearity32(rule)}")
    print()

    best = {}
    print("  DRIVER comparison (3 seeds each; report mean structural quality of the evolved winners):")
    for mode in ("multi", "s1", "random"):
        runs = []
        for sd in (1, 2, 3):
            r, f = evolve_search(mode, W, T, pop=args.pop, gens=args.gens, seed=sd)
            c, d, dens = signals(r, W, T); nl = nonlinearity32(r)
            runs.append((r, f, c, d, dens, nl))
        mc = np.mean([x[2] for x in runs]); md = np.mean([x[3] for x in runs]); mnl = np.mean([x[5] for x in runs])
        best[mode] = sorted(runs, key=lambda x: -x[1])[0]              # keep the top run for inspection
        rr = best[mode]
        print(f"    DRIVER={mode:7s} mean(comp={mc:.3f} damage={md:.3f} NL={mnl:.1f})  | top rule 0x{rr[0]:08X} comp={rr[2]:.3f} damage={rr[3]:.3f} NL={rr[5]}")

    # INSPECT the multi-driven winner's dynamics (the honest part: characterize, do not claim novel)
    print("\n  === INSPECT the MULTI-signal winner (characterize dynamics; NOT a novelty claim) ===")
    r = best["multi"][0]; lut = lut_of(r, 5)
    rng = np.random.default_rng(3)
    pat = evolve(lut, 2, W, 200, rng.integers(0, 2, W, dtype=np.uint8))
    # transient/period: hash each row, find first repeat
    seen = {}; period = None; trans = None
    for t, row in enumerate(pat):
        h = row.tobytes()
        if h in seen:
            period = t - seen[h]; trans = seen[h]; break
        seen[h] = t
    # localized-structure probe: does a single perturbation stay bounded (glider-like) or fill the cone (chaos)?
    dmg_final = damage(lut, 2, W, 200, 11)
    print(f"    rule 0x{r:08X}: comp={best['multi'][2]:.3f} damage={best['multi'][3]:.3f} NL={best['multi'][5]}")
    print(f"    cycle: " + (f"transient {trans}, period {period} (within 200 steps, W={W})" if period else "no exact cycle within 200 steps (long/aperiodic transient)"))
    print(f"    perturbation spread after 200 steps: {dmg_final:.3f}  (->0 heals, ->0.5 chaos, intermediate = localized/glider-ish)")
    print("\n  READ: compare DRIVER rows. If MULTI converges to high-NL intermediate-comp intermediate-damage rules while")
    print("  RANDOM does not, the bridge signal is OPTIMIZABLE (drives a search to structure), not merely selective —")
    print("  the generative counterpart to expW's distinctness-driven DIVERGENCE to noise. Ceiling unchanged: uncatalogued")
    print("  != provably-unknown; the winner is characterized, not claimed novel.")
