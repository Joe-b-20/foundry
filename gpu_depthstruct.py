"""
gpu_depthstruct.py — DEPTH-WHILE-STRUCTURED on the program space (post-audit phase 2a).

THE QUESTION (from the restated ceiling, 09_fable5_audit.md §2.3 + wall #4): the one
regime where un-named-but-meaningful objects provably live is DEEP, STRUCTURED
computation — exactly where the generative bridge stalled (expEE) and where pure
depth-evolution at scale climbs (exp2: 6238 @ Tmax 8000, near the detection cap).
But pure depth selects for ANY long-runner — typically boring counters. expEE tried
to add a structure term and hit the documented signal-craft wall: edge-of-chaos on
the TM SPACE-TIME diagram is contaminated by sparsity (head touches 1 cell/step ->
everything compresses to c~0.09). That gap was flagged in sessions 9/10 and never
closed. This experiment closes it and asks the real question:

  Does depth-driven evolution WITH a TM-appropriate structure term find machines
  that are deep AND structured — or does the structure term just tax depth?

TM-APPROPRIATE STRUCTURE TERMS (the fix for the expEE contamination):
  - track_struct: 4c(1-c) on the machine's MOVE-BIT sequence (last <=4096 steps,
    bit-packed 8/byte per the expX lesson). A metronome sweep -> highly compressible
    (c->0, score->0); a random walker -> incompressible (c->1, score->0); structured
    deep computation (Collatz-like, nested phases) -> intermediate.
  - tape_struct: 4c(1-c) on the FINAL WRITTEN BLOCK (bit-packed), gated to span>=32
    (tiny blocks are zlib-overhead noise). Measured on everything; only meaningful
    for machines that write real output.

CONDITIONS (matched budget, same evolution loop, only fitness differs):
  depth        fitness = runtime if halted else 0            (the exp2 baseline)
  depthXtrack  fitness = runtime * (0.1 + track_struct)      (depth, structured)
  trackonly    fitness = track_struct if halted else 0       (structure, no depth)
Both structure scores are MEASURED on every condition's winners regardless of the
fitness used, so the comparison is symmetric. Top-K winners are re-run individually
with full head-position tracking and saved for RENDERING (render-before-verdict).

Run:  python gpu_depthstruct.py --smoke
      python gpu_depthstruct.py --n 5 --Tmax 100000 --batch 4096 --gens 150 --seed 1 --out runs/dstruct_s1
"""
from __future__ import annotations
import argparse, time, os, json, zlib
import numpy as np
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def dev_info():
    if DEV == "cuda":
        p = torch.cuda.get_device_properties(0)
        return f"{p.name} {p.total_memory/1e9:.0f}GB"
    return "cpu"


def random_tms(B, n, gen):
    write = (torch.rand((B, n, 2), generator=gen, device=DEV) < 0.5).to(torch.uint8)
    move = torch.where(torch.rand((B, n, 2), generator=gen, device=DEV) < 0.5,
                       torch.tensor(1, device=DEV), torch.tensor(-1, device=DEV)).to(torch.int64)
    nxt = torch.randint(0, n, (B, n, 2), generator=gen, device=DEV)
    halt = torch.rand((B, n, 2), generator=gen, device=DEV) < (1.0 / (2 * n))
    nxt = torch.where(halt, torch.full_like(nxt, n), nxt)
    return write, move, nxt


def mutate_tms(tabs, gen, n, kflip=2):
    write, move, nxt = (t.clone() for t in tabs)
    B = write.shape[0]; bidx = torch.arange(B, device=DEV)
    for _ in range(kflip):
        s = torch.randint(0, n, (B,), generator=gen, device=DEV)
        b = torch.randint(0, 2, (B,), generator=gen, device=DEV)
        write[bidx, s, b] = (torch.rand((B,), generator=gen, device=DEV) < 0.5).to(torch.uint8)
        move[bidx, s, b] = torch.where(torch.rand((B,), generator=gen, device=DEV) < 0.5,
                                       torch.tensor(1, device=DEV), torch.tensor(-1, device=DEV))
        newn = torch.randint(0, n, (B,), generator=gen, device=DEV)
        halt = torch.rand((B,), generator=gen, device=DEV) < (1.0 / (2 * n))
        nxt[bidx, s, b] = torch.where(halt, torch.full_like(newn, n), newn)
    return write, move, nxt


def flatten(tabs, n):
    write, move, nxt = tabs
    mv01 = (move > 0).to(torch.long)
    return torch.cat([write.reshape(write.shape[0], -1).long(),
                      mv01.reshape(mv01.shape[0], -1),
                      nxt.reshape(nxt.shape[0], -1)], dim=1)


def unflatten(flat, n):
    B = flat.shape[0]; sz = n * 2
    write = flat[:, 0:sz].reshape(B, n, 2).to(torch.uint8)
    move = torch.where(flat[:, sz:2 * sz].reshape(B, n, 2) > 0,
                       torch.tensor(1, device=DEV), torch.tensor(-1, device=DEV))
    nxt = flat[:, 2 * sz:3 * sz].reshape(B, n, 2)
    return write, move, nxt


def run_tms_traced(tabs, n, L, Tmax, trace_len=4096, check_every=256):
    """Batched TM run from blank tape. Returns outcomes + a ring buffer of the last
    `trace_len` MOVE BITS per machine (1 = right) — the head-track structure source."""
    write, move, nxt = tabs
    B = write.shape[0]; bidx = torch.arange(B, device=DEV)
    tape = torch.zeros((B, L), dtype=torch.uint8, device=DEV)
    head = torch.full((B,), L // 2, dtype=torch.long, device=DEV)
    state = torch.zeros((B,), dtype=torch.long, device=DEV)
    halted = torch.zeros((B,), dtype=torch.bool, device=DEV)
    escaped = torch.zeros((B,), dtype=torch.bool, device=DEV)
    runtime = torch.zeros((B,), dtype=torch.long, device=DEV)
    moves = torch.zeros((B, trace_len), dtype=torch.uint8, device=DEV)
    lo = head.clone(); hi = head.clone()
    for t in range(Tmax):
        active = ~halted
        sym = tape[bidx, head].long()
        w = write[bidx, state, sym]
        d = move[bidx, state, sym]
        nx = nxt[bidx, state, sym]
        tape[bidx, head] = torch.where(active, w, tape[bidx, head])
        moves[:, t % trace_len] = torch.where(active, (d > 0).to(torch.uint8), moves[:, t % trace_len])
        newhead = head + torch.where(active, d, torch.zeros_like(d))
        off = (newhead < 0) | (newhead >= L)
        halt_now = active & ((nx == n) | off)
        state = torch.where(active & ~halt_now, nx, state)
        head = newhead.clamp(0, L - 1)
        lo = torch.minimum(lo, head); hi = torch.maximum(hi, head)
        runtime = runtime + active.long()
        escaped = escaped | (active & off & (nx != n))
        halted = halted | halt_now
        if (t & (check_every - 1)) == 0 and bool(halted.all()):
            break
    true_halt = halted & ~escaped
    ones = tape.long().sum(dim=1)
    span = (hi - lo + 1)
    return dict(runtime=runtime, halted=true_halt, escaped=escaped, ones=ones, span=span,
                tape=tape, lo=lo, hi=hi, moves=moves)


def edge_score(bits_u8):
    """4c(1-c) on a 0/1 uint8 numpy array, bit-packed 8/byte (the expX lesson).
    Returns 0.0 for arrays too short to score meaningfully (<256 bits)."""
    if bits_u8.size < 256:
        return 0.0
    packed = np.packbits(bits_u8)
    c = len(zlib.compress(packed.tobytes(), 6)) / len(packed)
    c = min(c, 1.0)
    return 4.0 * c * (1.0 - c)


def struct_scores(out):
    """(track_struct, tape_struct) per machine, CPU zlib. track on the last
    min(runtime, trace_len) move bits; tape on the written block, gated span>=32."""
    B = out["runtime"].shape[0]
    moves = out["moves"].cpu().numpy()
    tape = out["tape"].cpu().numpy()
    lo = out["lo"].cpu().numpy(); hi = out["hi"].cpu().numpy()
    rt = out["runtime"].cpu().numpy()
    tlen = moves.shape[1]
    track = np.zeros(B, dtype=np.float64)
    tstr = np.zeros(B, dtype=np.float64)
    for i in range(B):
        m = int(min(rt[i], tlen))
        if m >= 256:
            # ring buffer: if runtime < tlen the first m entries are in order;
            # else the ring holds the last tlen moves (order rotated — rotation
            # does not change zlib ratio materially for our purpose)
            track[i] = edge_score(moves[i, :m] if rt[i] <= tlen else moves[i])
        if hi[i] - lo[i] + 1 >= 32:
            tstr[i] = edge_score(tape[i, lo[i]:hi[i] + 1])
    return track, tstr


def fitness_for(cond, out, track, tstr):
    rt = out["runtime"].cpu().numpy().astype(np.float64)
    halted = out["halted"].cpu().numpy()
    if cond == "depth":
        f = np.where(halted, rt, 0.0)
    elif cond == "depthXtrack":
        f = np.where(halted, rt * (0.1 + track), 0.0)
    elif cond == "trackonly":
        # structure WITHOUT a depth/halt requirement: score the head-track of EVERY
        # machine that ran >=256 steps (halting or not). Structured infinite
        # computation (quasi-periodic walkers, nested sweeps) is exactly what an
        # uncontaminated edge-of-chaos signal should surface, and it ignites at gen 0
        # (no need to first find a deep halter). A tiny rt tiebreaker breaks ties.
        f = track + 1e-7 * np.minimum(rt, 1000.0)
    else:
        raise ValueError(cond)
    return torch.tensor(f, device=DEV, dtype=torch.float32)


def evolve(cond, n, L, Tmax, batch, gens, seed, topk=8):
    g = torch.Generator(device=DEV).manual_seed(seed)
    tabs = random_tms(batch, n, g)
    out = run_tms_traced(tabs, n, L, Tmax)
    track, tstr = struct_scores(out)
    fit = fitness_for(cond, out, track, tstr)
    flat = flatten(tabs, n)
    stats = dict(runtime=out["runtime"].cpu().numpy(), track=track, tape_s=tstr,
                 halted=out["halted"].cpu().numpy())
    nelite = max(2, batch // 5)
    trace = []
    t0 = time.time()
    for gen in range(gens):
        order = torch.argsort(fit, descending=True)
        elite = flat[order[:nelite]]
        reps = -(-batch // nelite)
        parents = elite.repeat(reps, 1)[:batch]
        child = mutate_tms(unflatten(parents, n), g, n)
        cout = run_tms_traced(child, n, L, Tmax)
        ctrack, ctstr = struct_scores(cout)
        cfit = fitness_for(cond, cout, ctrack, ctstr)
        cflat = flatten(child, n)
        allflat = torch.cat([elite, cflat]); allfit = torch.cat([fit[order[:nelite]], cfit])
        order2 = torch.argsort(allfit, descending=True)
        keep = order2[:batch]
        flat = allflat[keep]; fit = allfit[keep]
        # track stats arrays for the kept population (elite stats from prev, child stats new)
        prev_idx = order[:nelite].cpu().numpy()
        allstats = {k: np.concatenate([stats[k][prev_idx], v]) for k, v in
                    [("runtime", cout["runtime"].cpu().numpy()), ("track", ctrack),
                     ("tape_s", ctstr), ("halted", cout["halted"].cpu().numpy())]}
        kn = keep.cpu().numpy()
        stats = {k: v[kn] for k, v in allstats.items()}
        b = int(torch.argmax(fit))
        trace.append(dict(gen=gen, best_fit=float(fit[b]), best_rt=int(stats["runtime"][b]),
                          best_track=round(float(stats["track"][b]), 3),
                          best_tape=round(float(stats["tape_s"][b]), 3)))
        if gen % max(1, gens // 10) == 0 or gen == gens - 1:
            print(f"    [{cond}] gen {gen:4d}: fit={trace[-1]['best_fit']:.1f} rt={trace[-1]['best_rt']} "
                  f"track={trace[-1]['best_track']:.3f} tape={trace[-1]['best_tape']:.3f} "
                  f"[{time.time()-t0:.0f}s]")
    # final winners: top-K by fitness with full stats
    order = torch.argsort(fit, descending=True)[:topk]
    winners = []
    for r, i in enumerate(order.cpu().numpy()):
        winners.append(dict(rank=r, fit=float(fit[i]), runtime=int(stats["runtime"][i]),
                            track=round(float(stats["track"][i]), 3),
                            tape_s=round(float(stats["tape_s"][i]), 3),
                            halted=bool(stats["halted"][i]),
                            genome=flat[i].cpu().numpy().tolist()))
    return winners, trace


def rerun_full_trace(genome, n, L, Tmax_verify):
    """Re-run ONE machine with full head-position history (for rendering + the
    re-execute-archived-winners rule). Returns dict incl. positions (runtime,)."""
    tabs = unflatten(torch.tensor([genome], device=DEV), n)
    write, move, nxt = tabs
    tape = torch.zeros((1, L), dtype=torch.uint8, device=DEV)
    head = torch.full((1,), L // 2, dtype=torch.long, device=DEV)
    state = torch.zeros((1,), dtype=torch.long, device=DEV)
    pos = []
    rt = 0; halted = False; escaped = False
    for t in range(Tmax_verify):  # Tmax_verify == evolution Tmax: non-halters cap here
        sym = int(tape[0, head[0]])
        w, d, nx = int(write[0, state[0], sym]), int(move[0, state[0], sym]), int(nxt[0, state[0], sym])
        tape[0, head[0]] = w
        pos.append(int(head[0]))
        nh = int(head[0]) + d
        rt += 1
        if nx == n:
            halted = True
            break
        if nh < 0 or nh >= L:
            escaped = True
            break
        head[0] = nh; state[0] = nx
    return dict(runtime=rt, halted=halted, escaped=escaped,
                positions=np.array(pos, dtype=np.int32), tape=tape[0].cpu().numpy())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--L", type=int, default=8192)
    ap.add_argument("--Tmax", type=int, default=100000)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--gens", type=int, default=150)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--conds", type=str, default="depth,depthXtrack,trackonly")
    ap.add_argument("--out", type=str, default="runs/dstruct")
    args = ap.parse_args()
    if args.smoke:
        args.n, args.L, args.Tmax, args.batch, args.gens = 4, 512, 2000, 512, 12
    os.makedirs(args.out, exist_ok=True)
    print(f"gpu_depthstruct | device={dev_info()} | n={args.n} L={args.L} Tmax={args.Tmax} "
          f"batch={args.batch} gens={args.gens} seed={args.seed}")
    print("  QUESTION: does a TM-appropriate structure term (head-track edge-of-chaos, the expEE gap closed)")
    print("  change WHAT depth-evolution finds — deep+structured machines vs plain deep counters?\n")
    results = {}
    for cond in args.conds.split(","):
        print(f"  === condition: {cond} ===")
        winners, trace = evolve(cond, args.n, args.L, args.Tmax, args.batch, args.gens, args.seed)
        # re-execute winners (audit rule) + save full traces for rendering
        for wnr in winners[:4]:
            # re-run to the SAME Tmax used in evolution so (runtime, halted) must match
            # exactly — for a halter it confirms the depth; for a non-halter both cap at Tmax.
            rr = rerun_full_trace(wnr["genome"], args.n, args.L, args.Tmax)
            wnr["rerun_runtime"] = rr["runtime"]; wnr["rerun_halted"] = rr["halted"]
            wnr["rerun_ok"] = (rr["runtime"] == wnr["runtime"]) and (rr["halted"] == wnr["halted"])
            np.save(os.path.join(args.out, f"{cond}_top{wnr['rank']}_positions.npy"), rr["positions"])
            np.save(os.path.join(args.out, f"{cond}_top{wnr['rank']}_tape.npy"), rr["tape"])
        results[cond] = dict(winners=winners, trace=trace[-1])
        w0 = winners[0]
        print(f"    BEST: rt={w0['runtime']} track={w0['track']} tape={w0['tape_s']} "
              f"rerun_ok={w0.get('rerun_ok')}\n")
        json.dump(results, open(os.path.join(args.out, "dstruct_results.json"), "w"), indent=1)
    print("  === SUMMARY (best per condition) ===")
    for cond, r in results.items():
        w0 = r["winners"][0]
        print(f"    {cond:12s} rt={w0['runtime']:7d}  track={w0['track']:.3f}  tape={w0['tape_s']:.3f}  "
              f"rerun_ok={w0.get('rerun_ok')}")
    print("\n  READ: depth-only finding high-rt/low-track machines while depthXtrack finds high-rt/")
    print("  mid-track machines = the structure term steers WHAT is found (then RENDER the position")
    print("  traces before any verdict). depthXtrack collapsing to shallow runtimes = structure taxes")
    print("  depth (the expEE landscape gate, now measured with an uncontaminated signal).")
