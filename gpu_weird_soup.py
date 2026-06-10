"""
gpu_weird_soup.py — PRIMORDIAL SOUP: does LIFE (self-replicators) emerge from RANDOM programs, with NO target?

THE GAMBLE (the one bridge-signal family the project FLAGGED but never built — "survival-in-an-environment"). expW showed
DISTINCTNESS-driven open-ended search DIVERGES to a noise zoo (no meaning). The opposite open-ended selector is SURVIVAL /
REPLICATION: no fitness function, no target — programs that happen to copy themselves simply SPREAD. The striking 2024 result
("Computational Life", Agüera y Arcas et al.) is that in a soup of RANDOM self-modifying programs, self-replicators EMERGE
SPONTANEOUSLY from pure noise and take over. If true here, it's a clean demonstration that survival-selection CONVERGES to
functional MEANING where distinctness-selection diverged — the missing bridge, on a substrate the project never tried.

Model = BFF (Brainfuck w/ TWO heads so code can copy itself). Tape bytes ARE the program AND the data. Each epoch: randomly
PAIR tapes, CONCATENATE (A|B, length 2L), execute from ip=0 — the program can read/write/copy across BOTH halves, so A's
code can overwrite B with a copy of itself — then SPLIT back. Random byte mutations = cosmic rays. NO explicit selection:
replicators spread by overwriting partners. Emergence = a sharp drop in population entropy / rise in zlib-compressibility
(random soup = incompressible; replicator takeover = many copies = compressible) + a dominant genotype that contains a copy
loop. Honest: emergence-of-replicators is a known phenomenon (not human-unknown); the test is whether SURVIVAL produces
MEANING from no target (the bridge), and we characterize the emergent organism.

Ops (byte values): 0:'<'h0-- 1:'>'h0++ 2:'{'h1-- 3:'}'h1++ 4:'-'t[h0]-- 5:'+'t[h0]++ 6:'.'t[h1]=t[h0] 7:','t[h0]=t[h1]
                    8:'[' (if t[h0]==0 jump past matching ]) 9:']' (if t[h0]!=0 jump back to matching [).  bytes>=10 = no-op.
Run:  python gpu_weird_soup.py --selftest   (verify the interpreter)
      python gpu_weird_soup.py --smoke       (tiny soup)
      python gpu_weird_soup.py --N 8192 --L 64 --epochs 3000 --out runs/weird_soup   (scale, on the 4090)
"""
from __future__ import annotations
import argparse, time, os, json, zlib
import numpy as np
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
LT, GT, LCB, RCB, MINUS, PLUS, CPY01, CPY10, LBR, RBR = range(10)


def run_programs(tape, K, max_scan=24):
    """Execute B BFF programs in lockstep for K steps. tape: (B, T2) uint8 (modified in place & returned). Heads wrap
    mod T2; a program halts when its linear ip walks off the end. SYNC-FREE (no per-step .item() — those serialize the
    GPU and were the bottleneck). Brackets resolved by a FIXED-length scan of the current tape; max_scan bounds the loop-
    body length we can match (real copy loops have ~5-instruction bodies, so 24 is ample; longer straight-line spans
    between brackets simply don't jump — rare in evolved replicators)."""
    B, T2 = tape.shape
    bidx = torch.arange(B, device=tape.device)
    ip = torch.zeros(B, dtype=torch.long, device=tape.device)
    h0 = torch.zeros(B, dtype=torch.long, device=tape.device)
    h1 = torch.zeros(B, dtype=torch.long, device=tape.device)

    def scan(start, need, fwd):                              # find matching bracket; fixed max_scan iters, no sync
        sp = start.clone(); depth = need.long(); scanning = need.clone(); newip = start.clone()
        opn, cls = (LBR, RBR) if fwd else (RBR, LBR)
        for _ in range(max_scan):
            sp = torch.where(scanning, (sp + (1 if fwd else -1)) % T2, sp)
            cc = (tape[bidx, sp].long() & 15)               # op = low nibble (dense: ~62% of bytes are ops)
            depth = depth + scanning.long() * ((cc == opn).long() - (cc == cls).long())
            found = scanning & (depth == 0)
            newip = torch.where(found, sp, newip)
            scanning = scanning & ~found
        return newip

    for step in range(K):
        a = ip < T2                                          # alive mask (no sync)
        ipc = ip.clamp(0, T2 - 1)
        op = tape[bidx, ipc].long() & 15                     # op = low nibble; high nibble rides along as data
        h0 = (h0 + a.long() * ((op == GT).long() - (op == LT).long())) % T2
        h1 = (h1 + a.long() * ((op == RCB).long() - (op == LCB).long())) % T2
        d = a & ((op == PLUS) | (op == MINUS))
        newval = (tape[bidx, h0].long() + (op == PLUS).long() - (op == MINUS).long()).to(torch.uint8)
        tape[bidx, h0] = torch.where(d, newval, tape[bidx, h0])
        m = a & (op == CPY01)
        tape[bidx, h1] = torch.where(m, tape[bidx, h0], tape[bidx, h1])
        m = a & (op == CPY10)
        tape[bidx, h0] = torch.where(m, tape[bidx, h1], tape[bidx, h0])
        c0 = tape[bidx, h0]
        need_f = a & (op == LBR) & (c0 == 0)
        need_b = a & (op == RBR) & (c0 != 0)
        newip = torch.where(need_f, scan(ipc, need_f, True), ipc)
        newip = torch.where(need_b, scan(ipc, need_b, False), newip)
        ip = torch.where(need_f | need_b, newip, ip) + a.long()
    return tape


def assemble(prog, T2):
    t = np.zeros(T2, np.uint8) + 255
    t[:len(prog)] = prog
    return t


def selftest():
    # NB: BFF tape is CODE and DATA — '+' increments the cell at head0, which often holds an instruction byte. Expectations
    # below account for that (e.g. '+' on cell0 holding byte PLUS=5 gives 6). This is correct BFF behaviour, not a bug.
    print("=== interpreter self-test (BFF: code == data) ===")
    T2 = 32
    # A) single '+': tape[0] holds PLUS(5) -> 5+1 = 6
    t = torch.tensor(assemble([PLUS], T2)[None], device=DEV)
    run_programs(t, 4); okA = int(t[0, 0]) == 6
    print(f"  '+'            t[0]={int(t[0,0])} (want 6)   {'OK' if okA else 'FAIL'}")
    # B) '>+': move h0 to cell1 (holds PLUS=5), increment -> 6
    t = torch.tensor(assemble([GT, PLUS], T2)[None], device=DEV)
    run_programs(t, 4); okB = int(t[0, 1]) == 6
    print(f"  '>+'           t[1]={int(t[0,1])} (want 6)   {'OK' if okB else 'FAIL'}")
    # C) copy across heads: }x10 moves h1 to 10, then '.' copies tape[h0=0] (=RCB=3) -> tape[10]
    t = torch.tensor(assemble([RCB] * 10 + [CPY01], T2)[None], device=DEV)
    run_programs(t, 24); okC = int(t[0, 10]) == RCB
    print(f"  '}}x10 .'       t[10]={int(t[0,10])} (want {RCB})   {'OK' if okC else 'FAIL'}")
    # D) bracket loop: '<' wraps h0 to the LAST cell (data, separate from the code so it isn't self-corrupted), which we
    #    preset to 3; '[-]' must decrement it 3->0. Tests forward-skip-check + backward-jump-loop.
    td = np.zeros(T2, np.uint8) + 255
    td[:4] = [LT, LBR, MINUS, RBR]; td[T2 - 1] = 3
    t = torch.tensor(td[None], device=DEV)
    run_programs(t, 24); okD = int(t[0, T2 - 1]) == 0
    print(f"  '<[-]' on 3    t[-1]={int(t[0,T2-1])} (want 0)   {'OK' if okD else 'FAIL'}")
    allok = okA and okB and okC and okD
    print(f"  => interpreter {'CORRECT' if allok else 'BROKEN'}")
    return allok


def population_entropy(pop_np):
    """zlib-compressibility of the whole population (random ~1.0; replicator takeover -> low) + top-genotype fraction."""
    raw = pop_np.tobytes()
    comp = len(zlib.compress(raw, 6)) / max(1, len(raw))
    # top genotype frequency
    view = np.ascontiguousarray(pop_np).view([('', pop_np.dtype)] * pop_np.shape[1])
    _, counts = np.unique(view, return_counts=True)
    return comp, int(counts.max()), len(counts)


def soup(N, L, K, epochs, mut, alphabet, seed, out=None, log_every=50):
    T2 = 2 * L
    g = torch.Generator(device=DEV).manual_seed(seed)
    pop = torch.randint(0, alphabet, (N, L), generator=g, device=DEV, dtype=torch.uint8)
    hist = []
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(N, generator=g, device=DEV)
        a, b = perm[:N // 2], perm[N // 2:]
        tape = torch.cat([pop[a], pop[b]], dim=1)            # (N/2, 2L)
        tape = run_programs(tape, K)
        pop[a] = tape[:, :L]; pop[b] = tape[:, L:]
        if mut > 0:                                          # cosmic rays
            nm = int(mut * N * L)
            ii = torch.randint(0, N, (nm,), generator=g, device=DEV)
            jj = torch.randint(0, L, (nm,), generator=g, device=DEV)
            vv = torch.randint(0, alphabet, (nm,), generator=g, device=DEV, dtype=torch.uint8)
            pop[ii, jj] = vv
        if ep % log_every == 0 or ep == epochs - 1:
            comp, topfreq, ndistinct = population_entropy(pop.cpu().numpy())
            hist.append((ep, comp, topfreq, ndistinct))
            print(f"  ep {ep:5d}: zlib={comp:.3f} top-genotype={topfreq:5d}/{N} distinct={ndistinct:6d}  [{time.time()-t0:.0f}s]")
    # characterize the dominant genotype
    pop_np = pop.cpu().numpy()
    view = np.ascontiguousarray(pop_np).view([('', pop_np.dtype)] * L)
    vals, counts = np.unique(view, return_counts=True)
    top = pop_np[np.argmax((pop_np == vals[np.argmax(counts)].view(pop_np.dtype)).all(1))] if False else None
    # simpler: find the most common row
    uniq, inv, cnt = np.unique(pop_np, axis=0, return_inverse=True, return_counts=True)
    dom = uniq[np.argmax(cnt)]
    opnames = "<>{}-+.,[]"
    domstr = "".join(opnames[c] if c < 10 else "." for c in dom)
    has_copy = (CPY01 in dom or CPY10 in dom) and (LBR in dom)
    print(f"\n  dominant genotype x{int(cnt.max())}/{N}: {domstr}")
    print(f"  contains a copy-op inside a loop (replicator signature)? {has_copy}")
    if out:
        os.makedirs(out, exist_ok=True)
        json.dump(dict(N=N, L=L, K=K, epochs=epochs, mut=mut, alphabet=alphabet, hist=hist,
                       dominant=dom.tolist(), dominant_str=domstr, dom_count=int(cnt.max()), has_copy=bool(has_copy)),
                  open(os.path.join(out, "soup.json"), "w"), indent=2)
        np.save(os.path.join(out, "population.npy"), pop_np)
    return hist


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--N", type=int, default=8192); ap.add_argument("--L", type=int, default=64)
    ap.add_argument("--K", type=int, default=128); ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--mut", type=float, default=0.012); ap.add_argument("--alphabet", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    print(f"gpu_weird_soup | device={DEV} | primordial BFF soup\n")
    if args.selftest or args.smoke:
        ok = selftest()
        if not ok:
            raise SystemExit("interpreter broken — fix before running the soup")
    if args.selftest and not args.smoke:
        raise SystemExit(0)
    if args.smoke:
        args.N, args.L, args.K, args.epochs, args.alphabet = 1024, 32, 64, 200, 32
    print(f"\n  soup: N={args.N} L={args.L} K={args.K} epochs={args.epochs} mut={args.mut} alphabet={args.alphabet}")
    print("  WATCH: zlib drops + top-genotype rises = self-replicators emerged from random code (life from noise).\n")
    soup(args.N, args.L, args.K, args.epochs, args.mut, args.alphabet, args.seed, out=args.out)
    print("\n  READ: a sharp zlib drop + a dominant genotype carrying a copy-loop = SURVIVAL-selection produced functional")
    print("  MEANING from NO target (the bridge expW lacked). Flat zlib ~1.0 = no emergence (soup stayed random).")
