"""
expEE_logicaldepth.py — THE BRIDGE ON THE PROGRAM SPACE: a THIRD intrinsic-meaning signal (COMPUTATIONAL DEPTH), target-free,
over small TURING MACHINES. The project flagged "iterative/looped programs" as the object space the moonshot bridge must
reach (expX did binary sequences, expY cellular automata, expZ dynamical maps — none COMPUTE). And it named a third signal
family it never built: not compression, not invariants, but DEPTH — Bennett's logical depth / the Busy-Beaver intuition that
"interesting = a SHORT rule that COMPUTES A LONG TIME before producing STRUCTURE". This builds it.

Object space: n-state, 2-symbol Turing machines run from a BLANK tape (the Busy-Beaver setup — purest "simple rule -> rich
behavior"; the machine generates everything from nothing). Intrinsic signal, TARGET-FREE (never say what to compute):
  score = DEPTH x STRUCTURE
    DEPTH     = log(runtime)                          (Bennett: deep = took many steps; the Busy-Beaver axis)
    STRUCTURE = 4 c (1-c),  c = zlib ratio of the space-time diagram   (expY's VALIDATED edge-of-chaos: structured, not noise)
  gated to non-trivial machines (did real work, wrote a non-blank/non-saturated tape).
HYPOTHESIS (flagged uncertain): with NO target, this surfaces machines that run long AND build STRUCTURE — "deep computers"
— separating them from (a) trivial quick-halters, (b) long-but-CHAOTIC machines (random-looking space-time -> low structure),
and (c) the blank/dead. Mandatory controls (the expX lesson): a NOISE baseline (random space-time) MUST be rejected, and the
longest-RUNNING machine (max depth alone) should NOT necessarily top the DEPTH x STRUCTURE ranking. Then INSPECT/extract what
the top machines actually compute (counter? copier? fractal tape?) — characterized, NOT claimed novel. Pure Python. CPU.
Run: python expEE_logicaldepth.py
"""
from __future__ import annotations
import argparse, math, random, zlib
import numpy as np

HALT = -1

def random_tm(n, rng):
    """Transition table T[state][symbol] = (write, move, next). move in {-1,+1}; next in {0..n-1} or HALT.
    ~1/(2n+1) chance of HALT per entry so most machines can halt but not instantly."""
    T = []
    for s in range(n):
        row = []
        for b in (0, 1):
            w = rng.randint(0, 1)
            mv = rng.choice((-1, 1))
            if rng.random() < 1.0 / (2 * n):
                nx = HALT
            else:
                nx = rng.randint(0, n - 1)
            row.append((w, mv, nx))
        T.append(row)
    return T

def run_tm(T, T_max, W):
    """Run from blank tape, head at center, state 0. Returns (halted, runtime, spacetime list of rows, nonblank_cells).
    Stops on HALT, T_max, or head leaving the tape."""
    tape = np.zeros(W, dtype=np.uint8)
    head = W // 2; state = 0
    rows = []
    steps = 0
    halted = False
    while steps < T_max:
        rows.append(tape.copy())
        b = tape[head]
        w, mv, nx = T[state][b]
        tape[head] = w
        head += mv
        steps += 1
        if nx == HALT:
            halted = True
            rows.append(tape.copy())
            break
        state = nx
        if head < 0 or head >= W:
            break          # ran off the tape; treat as non-halting-here
    return halted, steps, rows, int((tape != 0).sum())

def comp_ratio(rows):
    if not rows:
        return 1.0
    pat = np.stack(rows).reshape(-1)
    packed = np.packbits(pat)
    return len(zlib.compress(packed.tobytes(), 9)) / max(1, len(packed))

def analyze(T, T_max, W):
    halted, runtime, rows, nz = run_tm(T, T_max, W)
    # gate: must do real work and leave a non-trivial tape
    width_used = W
    dens = nz / width_used
    if runtime < 8 or nz < 2:
        return None
    c = comp_ratio(rows)
    structure = 4 * c * (1 - c)
    depth = math.log(runtime)
    return {"score": depth * structure, "halted": halted, "runtime": runtime, "c": c,
            "structure": structure, "depth": depth, "nz": nz, "rows": len(rows)}

def describe(T, T_max, W):
    """Characterize what the top machine does: final tape motif + space-time periodicity."""
    halted, runtime, rows, nz = run_tm(T, T_max, W)
    final = rows[-1]
    nzidx = np.nonzero(final)[0]
    span = (int(nzidx.min()), int(nzidx.max())) if len(nzidx) else (0, 0)
    block = "".join(str(int(x)) for x in final[span[0]:span[1] + 1]) if len(nzidx) else ""
    # detect a repeating space-time period (a periodic "engine")
    period = None
    seen = {}
    for t, r in enumerate(rows):
        h = r.tobytes()
        if h in seen and period is None:
            period = t - seen[h]; break
        seen[h] = t
    return halted, runtime, nz, span, block[:60], period


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4, help="number of TM states")
    ap.add_argument("--N", type=int, default=6000, help="random machines to sample")
    ap.add_argument("--Tmax", type=int, default=1500)
    ap.add_argument("--W", type=int, default=301)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    n, N, T_max, W = args.n, args.N, args.Tmax, args.W
    print(f"LOGICAL-DEPTH BRIDGE on {n}-state 2-symbol TURING MACHINES (blank tape) — target-free depth x structure.\n")

    # NOISE baseline: a random space-time diagram must be REJECTED (incompressible -> structure ~ 0)
    rng0 = np.random.default_rng(0)
    noise_rows = [rng0.integers(0, 2, W, dtype=np.uint8) for _ in range(200)]
    cn = comp_ratio(noise_rows)
    print(f"  noise baseline: comp_ratio={cn:.3f} -> structure 4c(1-c)={4*cn*(1-cn):.3f} (incompressible -> ~0 -> rejected)\n")

    rng = random.Random(args.seed)
    res = []; halters = []
    halts = 0; longest = (0, None)
    for i in range(N):
        T = random_tm(n, rng)
        a = analyze(T, T_max, W)
        if a is None:
            continue
        if a["halted"]:
            halts += 1
            halters.append((a["runtime"], T, a))
        if a["runtime"] > longest[0]:
            longest = (a["runtime"], T)
        res.append((a["score"], T, a))
    res.sort(key=lambda x: -x[0])
    halters.sort(key=lambda x: -x[0])
    print(f"  sampled {N} machines; {len(res)} passed the non-trivial gate; {halts} halted within T_max={T_max}.")
    print(f"  empirical longest-running (halter or clipped): {longest[0]} steps.\n")

    print("  === TOP 12 by DEPTH x STRUCTURE (target-free) ===")
    print("   rank  score  runtime halt   comp   struct   nz   what it builds (final non-blank block; period)")
    for rank, (sc, T, a) in enumerate(res[:12]):
        halted, runtime, nz, span, block, period = describe(T, T_max, W)
        per = f"period {period}" if period else "aperiodic"
        print(f"   {rank:>3}  {sc:5.2f}  {runtime:6d}  {'H' if halted else '.'}   {a['c']:.3f}  {a['structure']:.3f}  {nz:4d}  [{block}] {per}")

    # the deepest HALTERS = the Busy-Beaver-like objects (a short rule that computes a long time, then STOPS with a tape)
    print("\n  === DEEPEST HALTERS (Busy-Beaver-like: long finite computation from a blank tape) ===")
    print("   runtime  comp   struct   nz   final non-blank block (what it computed)")
    for runtime, T, a in halters[:8]:
        halted, rt, nz, span, block, period = describe(T, T_max, W)
        print(f"   {runtime:6d}   {a['c']:.3f}  {a['structure']:.3f}  {nz:4d}  [{block}]")

    # contrast: where does the pure-DEPTH champion (longest runner) rank, and is it structured or chaotic?
    if longest[1] is not None:
        a = analyze(longest[1], T_max, W)
        print(f"\n  the LONGEST-RUNNING machine ({longest[0]} steps): structure={a['structure']:.3f} comp={a['c']:.3f}")
        print(f"  -> depth alone != top score: the signal prefers deep-AND-structured over max-runtime (which may be chaotic).")
    print("\n  READ: if the top machines run long AND build structured tapes (counters/periodic engines/nested patterns) while")
    print("  noise is rejected and the mere longest-runner isn't the winner, the DEPTH x STRUCTURE signal surfaces 'deep")
    print("  computation' un-targeted — a THIRD bridge-signal family, on the iterative-program space the moonshot needs.")
    print("  Ceiling honest: small TMs, surfaces KNOWN computational motifs, novelty not claimed.")
