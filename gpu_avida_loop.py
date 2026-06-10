"""
gpu_avida_loop.py — DOES REACHABLE DEPTH CHANGE THE NAMING-DENSITY? (post-audit phase 2b)

THE RESTATED WALL #6 (09_fable5_audit.md §2.3): the avida_oe ceiling was NOT "the op
set's closure is the named functions" (false — closure of a universal set is
everything). It is: target-free search reaches only SHORT/SHALLOW compositions, and
the shallow region of standard vocabularies is DENSELY NAMED. The sharp test of that
claim: GROW the reachable composition depth and watch the nearest-named similarity of
the discovered functions. Two outcomes, both informative:
  (A) deeper-reachable discoveries DROP in nearest-named similarity -> depth was the
      gate; the named region is shallow; Frontier-1-via-depth is real.
  (B) similarity stays high however deep we reach -> naming-density extends deeper
      than assumed; the ceiling is recognition, not depth (Frontier 2 dominates).

MECHANISM: take the validated cross-bit stack VM (gpu_avida_oe) and add BOUNDED LOOPS,
so a length-P genome can express FAR deeper computation than P ops. New ops:
  REP2 / REP4 / REP8 : the NEXT op is executed 2/4/8 times (cheap fixed-count repeat).
  LOOP <body...> END : execute the body up to MAXIT times while top-of-stack != 0
                       (a real data-dependent loop; the first op after LOOP that is
                       not consumed... bounded by MAXIT for safety/determinism).
Implementation keeps run_vm STRAIGHT-LINE per emitted op but EXPANDS the genome to an
execution trace first (loops unrolled against a per-organism iteration budget), so the
batched VM is unchanged — only the program each organism RUNS gets deeper. Depth is
controlled by --maxit (effective reachable depth ~ P * maxit).

CONDITIONS (matched wall-clock budget, target-free edge-of-chaos merit, identical to
avida_oe otherwise):
  noloop   maxit=1   (reproduces avida_oe: reachable depth = P)
  loop     maxit=K   (reachable depth up to P*K)
For each: the ~80-entry named suite (gpu_avida_oe._named_suite) exact-matched on 64
probes + the nearest-named bit-similarity for non-matches, logged per snapshot. The
HEADLINE METRIC is the nearest-named similarity of the top organisms as a function of
reachable depth.

Run:  python gpu_avida_loop.py --selftest
      python gpu_avida_loop.py --smoke
      python gpu_avida_loop.py --maxit 8 --N 12288 --gens 1500 --seed 1 --out runs/loop_s1
"""
from __future__ import annotations
import argparse, time, os, json, zlib
import numpy as np
import torch

import gpu_avida_oe as oe          # reuse run_vm, MASK, merit_oe, _named_suite, describe, _probes
DEV = oe.DEV
MASK = oe.MASK

# extended op table: 0..15 = the avida_oe ops; 16..18 = REP2/4/8; 19 = LOOP; 20 = END
NOP = 0
REP2, REP4, REP8, LOOP, END = 16, 17, 18, 19, 20
NUM_OPS_L = 21
OPN_L = oe.OPN + ["rep2", "rep4", "rep8", "loop", "end"]


def decode_l(row):
    return " ".join(OPN_L[int(o)] for o in row)


def expand_program(prog_np, maxit):
    """Expand a genome with REP*/LOOP/END control ops into a STRAIGHT-LINE op sequence
    (data-INDEPENDENT unrolling: LOOP bodies are unrolled `maxit` times; REPk repeats
    the next primitive k times). Returns a padded (N, Pexp) long tensor of base ops
    (0..15) only. Deterministic, no data dependence -> stays fully batched.

    This realizes DEEPER REACHABLE COMPUTATION (the experiment's variable) while
    keeping the VM straight-line. Data-dependent looping is a richer variant (future);
    fixed unrolling already grows reachable depth P -> up to P*maxit, which is the knob
    the naming-density question needs."""
    N, P = prog_np.shape
    base_ops = set(range(16))
    out_rows = []
    maxlen = 0
    for i in range(N):
        seq = []
        j = 0
        row = prog_np[i]
        guard = 0
        while j < P and guard < P * (maxit + 2):
            guard += 1
            op = int(row[j])
            if op < 16:
                seq.append(op); j += 1
            elif op in (REP2, REP4, REP8):
                k = {REP2: 2, REP4: 4, REP8: 8}[op]
                # repeat the NEXT base op k times (skip nested control after a rep)
                if j + 1 < P and int(row[j + 1]) < 16:
                    seq.extend([int(row[j + 1])] * k); j += 2
                else:
                    j += 1
            elif op == LOOP:
                # find matching END (first END after j); unroll body maxit times
                depth = 1; e = j + 1
                while e < P and depth > 0:
                    o2 = int(row[e])
                    if o2 == LOOP: depth += 1
                    elif o2 == END: depth -= 1
                    if depth == 0: break
                    e += 1
                body = [int(o) for o in row[j + 1:e] if int(o) < 16]
                if body:
                    seq.extend(body * maxit)
                j = e + 1
            else:  # stray END or nop-control
                j += 1
        if not seq:
            seq = [NOP]
        out_rows.append(seq)
        maxlen = max(maxlen, len(seq))
    # cap execution length to bound VM cost: grows with maxit (deeper reachable
    # computation is the point) but saturates at 128 ops so the 16x-depth arm stays
    # GPU-feasible. reachable depth still >> the noloop P.
    cap = max(8, min(maxlen, 16 + 12 * max(1, maxit), 128))
    arr = np.zeros((N, cap), dtype=np.int64)
    for i, seq in enumerate(out_rows):
        s = seq[:cap]
        arr[i, :len(s)] = s
    return torch.tensor(arr, device=DEV), cap


def merit_loop(prog, S, maxit, pop_sigs=None):
    """Same target-free edge-of-chaos merit as avida_oe, but on the EXPANDED program."""
    exp, _ = expand_program(prog.cpu().numpy(), maxit)
    return oe.merit_oe(exp, S, pop_sigs)


def describe_loop(prog_row, S, maxit):
    exp, _ = expand_program(prog_row[None], maxit)
    return oe.describe(exp[0].cpu().numpy(), S)


def selftest():
    print("=== loop-VM self-test (expansion correctness) ===")
    # REP4 of push? use a clear case: [PA, REP2, SHL]  -> a, then shl twice  = a<<2
    P = np.array([[oe.PA, REP2, oe.SHL, NOP, NOP, NOP]])
    exp, cap = expand_program(P, 1)
    seq = [int(x) for x in exp[0].cpu().numpy() if True][:5]
    okrep = seq[:3] == [oe.PA, oe.SHL, oe.SHL]
    # LOOP body unrolled maxit times: [PA, LOOP, SHL, END] maxit=3 -> a, shl,shl,shl
    P2 = np.array([[oe.PA, LOOP, oe.SHL, END, NOP, NOP]])
    exp2, _ = expand_program(P2, 3)
    seq2 = [int(x) for x in exp2[0].cpu().numpy()][:5]
    okloop = seq2[:4] == [oe.PA, oe.SHL, oe.SHL, oe.SHL]
    # run the expanded a<<2 program and check it computes a<<2 (value appears on the
    # stack — avida_oe reads the LAST cell as output; selftest searches like oe.selftest)
    a = torch.tensor([0x0101], device=DEV); b = torch.tensor([0], device=DEV)
    full = oe.run_vm(exp, a, b, 8)
    expected = (0x0101 << 2) & MASK
    okval = bool((full[0] == expected).any())
    for nm, ok in [("REP2 expand", okrep), ("LOOP unroll x3", okloop), ("a<<2 value", okval)]:
        print(f"  {nm:16s} {'OK' if ok else 'FAIL'}")
    allok = okrep and okloop and okval and oe.selftest()
    print(f"  => loop-VM {'CORRECT' if allok else 'BROKEN'}")
    return allok


def evolve(N, P, S, gens, seed, maxit, out_dir, snap=10, topk=5):
    os.makedirs(out_dir, exist_ok=True)
    g = torch.Generator(device=DEV).manual_seed(seed)
    prog = torch.randint(0, NUM_OPS_L, (N, P), generator=g, device=DEV)
    log = []; first_match = {}; t0 = time.time(); sigs = None
    for gen in range(gens + 1):
        merit, edge, nontriv, sig = merit_loop(prog, S, maxit, sigs)
        sigs = sig[torch.randperm(N, device=DEV)[:256]]
        if gen % snap == 0 or gen == gens:
            top = torch.argsort(merit, descending=True)[:topk].cpu().numpy()
            tops = []
            for r, bi in enumerate(top):
                match, pairs, near, simv = describe_loop(prog[bi].cpu().numpy(), S, maxit)
                for k in match:
                    first_match.setdefault(k, gen)
                tops.append(dict(rank=int(r), merit=round(float(merit[bi]), 3), edge=round(float(edge[bi]), 3),
                                 match=match, nearest=near, near_sim=simv, pairs=pairs,
                                 prog=decode_l(prog[bi].cpu().numpy())))
            # headline metric: nearest-named similarity of the TOP organism (1.0 if it IS named)
            top_named = 1.0 if tops[0]["match"] else tops[0]["near_sim"]
            m = dict(gen=int(gen), best=round(float(merit.max()), 3), mean=round(float(merit.mean()), 3),
                     best_edge=tops[0]["edge"], match=tops[0]["match"], top_named_sim=round(top_named, 3),
                     n_named_in_top=sum(1 for t in tops if t["match"]), top=tops, first_match=dict(first_match))
            log.append(m)
            json.dump(log, open(os.path.join(out_dir, "loop_log.json"), "w"), indent=1)
            np.save(os.path.join(out_dir, "prog.npy"), prog.cpu().numpy())
            nm = tops[0]["match"] if tops[0]["match"] else f"~{tops[0]['nearest']}@{tops[0]['near_sim']}"
            print(f"  gen {gen:5d}: best={m['best']:.2f} edge={m['best_edge']:.2f} top_named_sim={top_named:.3f} "
                  f"named_in_top5={m['n_named_in_top']} | {nm} | fm={list(first_match)[:6]} [{time.time()-t0:.0f}s]")
        if gen == gens:
            break
        w = (0.02 + merit).clamp(min=1e-3)
        parents = torch.multinomial(w, N, replacement=True, generator=g)
        prog = prog[parents].clone()
        nm = max(1, int(0.5 * N)); idx = torch.randint(0, N, (nm,), generator=g, device=DEV)
        pos = torch.randint(0, P, (nm,), generator=g, device=DEV)
        prog[idx, pos] = torch.randint(0, NUM_OPS_L, (nm,), generator=g, device=DEV)
    return log


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true"); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--N", type=int, default=12288); ap.add_argument("--P", type=int, default=24)
    ap.add_argument("--S", type=int, default=12); ap.add_argument("--gens", type=int, default=1500)
    ap.add_argument("--maxit", type=int, default=8); ap.add_argument("--snap", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--out", type=str, default="runs/avida_loop")
    args = ap.parse_args()
    if args.selftest or args.smoke:
        if not selftest() and args.smoke:
            raise SystemExit("loop-VM broken")
    if args.selftest and not args.smoke:
        raise SystemExit(0)
    if args.smoke:
        args.N, args.P, args.gens, args.snap = 2048, 20, 120, 20
    print(f"\ngpu_avida_loop | device={DEV} | maxit={args.maxit} (reachable depth ~P*maxit={args.P*args.maxit}) "
          f"N={args.N} gens={args.gens}\n")
    print("  HEADLINE: top_named_sim as a function of reachable depth. DROP with depth -> named region is")
    print("  shallow (depth was the gate). FLAT-HIGH -> naming-density extends deep (recognition is the ceiling).\n")
    evolve(args.N, args.P, args.S, args.gens, args.seed, args.maxit, args.out, args.snap, args.topk)
