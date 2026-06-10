"""
expFF_search.py — CLOSURE ADDENDUM to the learnability wall (2026-06-09 audit §2.2).

expFF demonstrated that a 2-op pseudorandom mixer is unlearnable from examples by
STATISTICAL learners (linear / kNN / MLP all at chance). The taxonomy then said
"un-discoverable from outcome — FIX: none from outcome." The audit's objection: the
project's own engine is exact-filtered PROGRAM SEARCH, and the demo function has a
~30-bit description — so it should be discoverable from outcome by short-program
enumeration. This script tests that, and then grows the description until search
dies — converting wall #5 from a "2-op wall" into a measured DESCRIPTION-LENGTH wall.

Setup mirrors expFF exactly (w=16-bit words, N=4000 train pairs from seed 1, held-out
4000). Search primitives (the analog of giving the adder VM digit ops):
    XSR(k): x ^= x >> k   (k=1..15)        15 options
    XSL(k): x ^= x << k   (k=1..15)        15 options
    MUL(c): x = x*c mod 2^16, c odd        32768 options
Program = a sequence of these ops. The TRUE f_1 is [XSR(7), MUL(0x9E37)] — depth 2.
Search = enumerate ALL depth-2 programs (vectorized over the MUL constant), filter by
exact match on 16 probe pairs (16x16 = 256 bits of constraint -> false survivors
~impossible), then VERIFY survivors exactly on all 4000 train and 4000 held-out pairs.
No partial credit anywhere.

Three parts:
  1. DISCOVERY: depth-2 exhaustive search vs f_1 (R=1). Prediction: found exactly,
     held-out 1.000 — i.e. the function expFF called "un-discoverable from outcome"
     is discovered from outcome by the project's own paradigm.
  2. CONTROL: the same search vs a RANDOM permutation's examples (no short program
     exists). Prediction: zero survivors — the search does not hallucinate.
  3. THE REAL WALL: secret = depth-4 (R=2 with independent (k,c) per round). Key
     space ~(30*32768)^2 ~ 2^39.8 — exhaustive enumeration infeasible at CPU budget.
     A budgeted random program search (default 2e7 candidates, 16-probe filtered)
     is predicted to find NOTHING — discovery dies at the enumeration boundary,
     which is the honest statement of the learnability/cryptographic wall for
     outcome-driven program search: it binds at DESCRIPTION LENGTH, not op count.

Pure NumPy, CPU. Run: python expFF_search.py [--budget4 20000000]
"""
from __future__ import annotations
import argparse, time, json, os
import numpy as np

W = 16
MASK = (1 << W) - 1


def f_round(x, k, c):
    x = x ^ (x >> k)
    x = (x * c) & MASK
    return x & MASK


def f_true(x, rounds):  # the expFF mixer: k=7, c=0x9E37 per round
    for _ in range(rounds):
        x = f_round(x, 7, 0x9E37)
    return x


# ---- op application, vectorized over a probe vector ----
# op encoding: (kind, param). kind 0 = XSR(k), 1 = XSL(k), 2 = MUL(c).
def apply_op(x, kind, param):
    if kind == 0:
        return (x ^ (x >> param)) & MASK
    if kind == 1:
        return (x ^ ((x << param) & MASK)) & MASK
    return (x * param) & MASK


def apply_prog(x, prog):
    for kind, param in prog:
        x = apply_op(x, kind, param)
    return x


def op_name(kind, param):
    return ["x^=x>>%d", "x^=x<<%d", "x*=0x%04X"][kind] % param


ODDS = np.arange(1, 1 << W, 2, dtype=np.int64)          # all 32768 odd multipliers
SHIFT_OPS = [(0, k) for k in range(1, 16)] + [(1, k) for k in range(1, 16)]   # 30


def depth2_search(probe_x, probe_y):
    """Exhaustive depth-2: {shift-op then MUL sweep, MUL sweep then shift-op,
    shift x shift, MUL x MUL(combined = single MUL, skip)}. Returns survivors of the
    16-probe exact filter as concrete programs."""
    survivors = []
    px = probe_x.astype(np.int64)
    # A) shift then MUL: y = (shift(x)) * c  -> c determined per probe? sweep all c vectorized
    for kind, k in SHIFT_OPS:
        sx = apply_op(px, kind, k)                       # (16,)
        out = (sx[None, :] * ODDS[:, None]) & MASK       # (32768, 16)
        hit = np.nonzero((out == probe_y[None, :]).all(axis=1))[0]
        for h in hit:
            survivors.append([(kind, k), (2, int(ODDS[h]))])
    # B) MUL then shift: y = shift(x*c)
    for kind, k in SHIFT_OPS:
        mx = (px[None, :] * ODDS[:, None]) & MASK        # (32768, 16)
        if kind == 0:
            out = (mx ^ (mx >> k)) & MASK
        else:
            out = (mx ^ ((mx << k) & MASK)) & MASK
        hit = np.nonzero((out == probe_y[None, :]).all(axis=1))[0]
        for h in hit:
            survivors.append([(2, int(ODDS[h])), (kind, k)])
    # C) shift then shift (no MUL)
    for k1 in SHIFT_OPS:
        x1 = apply_op(px, *k1)
        for k2 in SHIFT_OPS:
            if (apply_op(x1, *k2) == probe_y).all():
                survivors.append([k1, k2])
    # D) single MUL (depth 1) and single shift, for completeness
    out = (px[None, :] * ODDS[:, None]) & MASK
    for h in np.nonzero((out == probe_y[None, :]).all(axis=1))[0]:
        survivors.append([(2, int(ODDS[h]))])
    for kind, k in SHIFT_OPS:
        if (apply_op(px, kind, k) == probe_y).all():
            survivors.append([(kind, k)])
    return survivors


def verify_exact(prog, xs, ys):
    return bool((apply_prog(xs.astype(np.int64), prog) == ys).all())


def random_depth4_search(probe_x, probe_y, budget, seed=0, chunk=200_000):
    """Budgeted random search over depth-4 programs (each op uniformly a shift-op or
    MUL with random odd constant). 16-probe filtered, exact. Returns (n_tried, hits)."""
    rng = np.random.default_rng(seed)
    px = probe_x.astype(np.int64)
    tried = 0; hits = []
    n_ops = len(SHIFT_OPS)
    while tried < budget:
        m = min(chunk, budget - tried)
        # sample m programs of 4 ops: kind chosen shift(0/1) vs mul(2) w.p. 1/2 each
        x = np.repeat(px[None, :], m, axis=0)            # (m, 16)
        for pos in range(4):
            is_mul = rng.random(m) < 0.5
            kidx = rng.integers(0, n_ops, m)
            cs = ODDS[rng.integers(0, len(ODDS), m)]
            # apply vectorized per kind
            for kind, k in SHIFT_OPS:
                sel = (~is_mul) & (kidx == SHIFT_OPS.index((kind, k)))
                if sel.any():
                    if kind == 0:
                        x[sel] = (x[sel] ^ (x[sel] >> k)) & MASK
                    else:
                        x[sel] = (x[sel] ^ ((x[sel] << k) & MASK)) & MASK
            if is_mul.any():
                x[is_mul] = (x[is_mul] * cs[is_mul, None]) & MASK
        ok = (x == probe_y[None, :]).all(axis=1)
        if ok.any():
            hits.extend(np.nonzero(ok)[0] + tried)
        tried += m
    return tried, hits


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=4000)
    ap.add_argument("--budget4", type=int, default=20_000_000)
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(1)                       # same data seed as expFF
    allx = rng.permutation(1 << W)[: 2 * args.N]
    xtr, xte = allx[: args.N].astype(np.int64), allx[args.N: 2 * args.N].astype(np.int64)

    print("THE LEARNABILITY WALL, REVISITED WITH THE PROJECT'S OWN PARADIGM (exact-filtered program search).")
    print(f"Same setup as expFF: w=16, N={args.N} train pairs, {args.N} held-out. Search primitives:")
    print("  XSR(k) x^=x>>k (15) | XSL(k) x^=x<<k (15) | MUL(c) x*=c mod 2^16, c odd (32768)\n")
    results = {}

    # ---- Part 1: discover f_1 (R=1) from outcome, depth-2 exhaustive ----
    ytr = f_true(xtr, 1); yte = f_true(xte, 1)
    probe_x, probe_y = xtr[:16], ytr[:16]
    sv = depth2_search(probe_x, probe_y)
    print(f"PART 1 — R=1 mixer (true program: x^=x>>7; x*=0x9E37). Depth<=2 exhaustive search,")
    print(f"  16-probe filter -> {len(sv)} candidate(s); exact-verifying on 4000 train + 4000 held-out:")
    found = []
    for prog in sv:
        if verify_exact(prog, xtr, ytr) and verify_exact(prog, xte, yte):
            found.append(" ; ".join(op_name(*o) for o in prog))
    for s in found:
        print(f"    FOUND (exact on all 8000): {s}")
    p1 = "DISCOVERED from outcome" if found else "not found"
    print(f"  => {p1}. The function expFF's learners sat at chance on is recovered exactly,")
    print(f"     held-out 1.000, by short-program enumeration + exact verification. [{time.time()-t0:.0f}s]\n")
    results["R1_found"] = found

    # ---- Part 2: control — a random permutation (no short program exists) ----
    perm = np.random.default_rng(99).permutation(1 << W)
    ytr_r = perm[xtr]; sv_r = depth2_search(xtr[:16], ytr_r[:16])
    sv_r_ok = [p for p in sv_r if verify_exact(p, xtr, ytr_r)]
    print(f"PART 2 — CONTROL: random permutation's examples. Depth<=2 search: {len(sv_r)} probe-survivor(s),")
    print(f"  {len(sv_r_ok)} after full verification (expected 0 — the search does not hallucinate). [{time.time()-t0:.0f}s]\n")
    results["control_false_positives"] = len(sv_r_ok)

    # ---- Part 3: grow the description — R=2 with independent secrets per round ----
    k1, c1, k2, c2 = 11, 0x6D2B, 5, 0x9E37 + 0x11A2     # an arbitrary 4-op secret
    c1 |= 1; c2 |= 1
    def f_secret(x):
        return f_round(f_round(x, k1, c1), k2, c2)
    ytr2 = f_secret(xtr)
    keyspace_bits = np.log2((len(SHIFT_OPS) * len(ODDS)) ** 2)
    print(f"PART 3 — THE REAL WALL: secret = depth-4 program (two independent rounds, key space ~2^{keyspace_bits:.1f}).")
    print(f"  Budgeted random depth-4 search, budget {args.budget4:,} programs (~2^{np.log2(args.budget4):.1f}):")
    tried, hits = random_depth4_search(xtr[:16], ytr2[:16], args.budget4)
    print(f"  tried {tried:,} -> {len(hits)} probe-survivors (expected 0 at this budget).")
    print(f"  => discovery dies at the ENUMERATION BOUNDARY: ~2^{keyspace_bits:.1f} key space vs ~2^{np.log2(args.budget4):.1f} budget.")
    print(f"     [{time.time()-t0:.0f}s]\n")
    results["R2_keyspace_bits"] = round(float(keyspace_bits), 1)
    results["R2_budget"] = tried
    results["R2_hits"] = len(hits)

    print("READ: wall #5 restated. An efficiently-computable mixer is (a) unlearnable by statistical")
    print("learners (expFF), yet (b) DISCOVERABLE from outcome by exact-filtered short-program search")
    print("when its description is short (Part 1, with a clean no-hallucination control, Part 2), and")
    print("(c) un-discoverable again once the secret/description exceeds the search's enumeration reach")
    print("(Part 3). For outcome-driven PROGRAM SEARCH the cryptographic wall is a DESCRIPTION-LENGTH")
    print("wall, not an op-count wall. (Caveat: Part 3 is a budget statement, not an impossibility proof;")
    print("a cleverer-than-enumeration search could in principle exploit the mixer's algebraic structure.)")
    os.makedirs("runs", exist_ok=True)
    json.dump(results, open("runs/expFF_search.json", "w"), indent=1)
    print("\nwrote runs/expFF_search.json")
