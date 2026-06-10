"""
gpu_exp2_qd.py — QUALITY-DIVERSITY (MAP-Elites) vs the LANDSCAPE WALL, at GPU scale.

THE WALL THIS ATTACKS (taxonomy #4, the ONLY hard wall marked "FIX: unknown"). expEE found that on the PROGRAM space
(n-state Turing machines from a blank tape) deep/structured computation lives at ISOLATED, mutation-fragile NEEDLES: a
single objective (depth = runtime-to-halt) + plain evolution STALLED at 43 steps (vs BB(5)=47,176,870) because almost any
mutation to a deep halter breaks halting (fitness->0), so the landscape is maximally rugged and hill-climbing can't move.

The RIGHT tool for rugged, deceptive landscapes is QUALITY-DIVERSITY: don't hill-climb one objective — maintain a DIVERSE
ARCHIVE of elites binned by BEHAVIOR, and let stepping-stones in one niche seed progress in another (MAP-Elites). It
illuminates the space instead of climbing it. The GPU lets the archive be filled by MILLIONS of batched TM evaluations —
the scale the 4060 cannot reach. Three conditions share an identical evaluation budget (fair compare):
  MAP-ELITES   — niche archive over a 2D behavior descriptor (tape-span used, ones written); in-niche objective = depth.
  EVOLUTION    — expEE's single-objective depth hill-climb (the baseline that stalled at 43).
  SAMPLING     — random TMs (the floor).
RESULT either way is a clean wall finding: QD reaches deeper than evolution (a CROSSING of the landscape wall) OR QD also
stalls (the wall is real even for the method designed to beat rugged landscapes). Deep halters are KNOWN champions, so this
is a WALL experiment, not a novelty claim — but the archive's structured-deep machines are saved for inspection (a halter
that is deep AND has low-complexity structured space-time would be moonshot-adjacent).

Run small (4060):  python gpu_exp2_qd.py --smoke
Run scale (4090):  python gpu_exp2_qd.py --n 5 --batch 8192 --gens 4000 --Tmax 100000 --grid 28 --out runs/exp2_n5
"""
from __future__ import annotations
import argparse, time, os, json, zlib
import numpy as np
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"
HALTW = None  # set to n at runtime: next-state == n means HALT


def dev_info():
    if DEV == "cuda":
        p = torch.cuda.get_device_properties(0)
        return f"{p.name} {p.total_memory/1e9:.0f}GB"
    return "cpu"


# ---------------------------------------------------------------------------
# Batched n-state, 2-symbol Turing machines on a bounded tape of length L.
# Genome tensors: write (B,n,2) in {0,1}; move (B,n,2) in {-1,+1}; nxt (B,n,2) in {0..n-1, n=HALT}.
# ---------------------------------------------------------------------------
def random_tms(B, n, gen):
    write = (torch.rand((B, n, 2), generator=gen, device=DEV) < 0.5).to(torch.uint8)
    move = torch.where(torch.rand((B, n, 2), generator=gen, device=DEV) < 0.5,
                       torch.tensor(1, device=DEV), torch.tensor(-1, device=DEV)).to(torch.int64)
    nxt = torch.randint(0, n, (B, n, 2), generator=gen, device=DEV)
    halt = torch.rand((B, n, 2), generator=gen, device=DEV) < (1.0 / (2 * n))   # ~1 HALT edge per machine
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


def run_tms(tabs, n, L, Tmax, check_every=256):
    """Run a batch of TMs from a blank tape. Returns dict of per-machine outcomes (all GPU tensors)."""
    write, move, nxt = tabs
    B = write.shape[0]; bidx = torch.arange(B, device=DEV)
    tape = torch.zeros((B, L), dtype=torch.uint8, device=DEV)
    head = torch.full((B,), L // 2, dtype=torch.long, device=DEV)
    state = torch.zeros((B,), dtype=torch.long, device=DEV)
    halted = torch.zeros((B,), dtype=torch.bool, device=DEV)
    escaped = torch.zeros((B,), dtype=torch.bool, device=DEV)
    runtime = torch.zeros((B,), dtype=torch.long, device=DEV)
    lo = head.clone(); hi = head.clone()
    for t in range(Tmax):
        active = ~halted
        sym = tape[bidx, head].long()
        w = write[bidx, state, sym]
        d = move[bidx, state, sym]
        nx = nxt[bidx, state, sym]
        tape[bidx, head] = torch.where(active, w, tape[bidx, head])
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
    ones = tape.long().sum(dim=1)                # Σ-style: ones written on the tape (long: no uint8 overflow)
    span = (hi - lo + 1)                         # tape cells visited
    return dict(runtime=runtime, halted=true_halt, escaped=escaped, ones=ones, span=span, tape=tape)


# ---------------------------------------------------------------------------
# Genome flatten/unflatten (for the archive, which lives on GPU as dense arrays).
# ---------------------------------------------------------------------------
def flatten(tabs, n):
    write, move, nxt = tabs
    mv01 = (move > 0).to(torch.long)             # -1->0, +1->1
    return torch.cat([write.reshape(write.shape[0], -1).long(),
                      mv01.reshape(mv01.shape[0], -1),
                      nxt.reshape(nxt.shape[0], -1)], dim=1)   # (B, 6n)


def unflatten(flat, n):
    B = flat.shape[0]; sz = n * 2
    write = flat[:, 0:sz].reshape(B, n, 2).to(torch.uint8)
    move = torch.where(flat[:, sz:2 * sz].reshape(B, n, 2) > 0,
                       torch.tensor(1, device=DEV), torch.tensor(-1, device=DEV))
    nxt = flat[:, 2 * sz:3 * sz].reshape(B, n, 2)
    return write, move, nxt


# ---------------------------------------------------------------------------
# Descriptors -> archive cell. 2D behavior: (log2 span, ones), each binned into `grid`.
# ---------------------------------------------------------------------------
def cell_ids(out, grid, L):
    span = out["span"].float().clamp(min=1)
    ones = out["ones"].float()
    b1 = (torch.log2(span) / np.log2(L) * grid).long().clamp(0, grid - 1)
    omax = max(1.0, float(ones.max()))
    b2 = (ones / (omax + 1e-9) * grid).long().clamp(0, grid - 1)
    return b1 * grid + b2


def fitness(out):
    # depth objective: runtime for TRUE halters, else 0 (escaped / non-halting score 0, as in expEE)
    return torch.where(out["halted"], out["runtime"].float(), torch.zeros_like(out["runtime"].float()))


# ---------------------------------------------------------------------------
# MAP-Elites.
# ---------------------------------------------------------------------------
def map_elites(n, L, Tmax, batch, gens, grid, seed, out=None):
    gsize = 6 * n; ncells = grid * grid
    g = torch.Generator(device=DEV).manual_seed(seed)
    arch_fit = torch.full((ncells,), -1.0, device=DEV)
    arch_gen = torch.zeros((ncells, gsize), dtype=torch.long, device=DEV)

    def insert(flat, fit, cells):
        order = torch.argsort(fit)               # ascending -> last scatter (highest fit) wins per cell
        cand_fit = torch.full((ncells,), -1.0, device=DEV)
        cand_gen = torch.zeros((ncells, gsize), dtype=torch.long, device=DEV)
        cand_fit[cells[order]] = fit[order]
        cand_gen[cells[order]] = flat[order]
        better = cand_fit > arch_fit
        arch_fit[better] = cand_fit[better]
        arch_gen[better] = cand_gen[better]
        return int(better.sum())

    # seed the archive with random machines
    tabs = random_tms(batch, n, g)
    o = run_tms(tabs, n, L, Tmax)
    insert(flatten(tabs, n), fitness(o), cell_ids(o, grid, L))
    best_trace = [float(arch_fit.max())]
    for gen in range(gens):
        filled = (arch_fit > -1).nonzero(as_tuple=True)[0]
        if len(filled) == 0:
            tabs = random_tms(batch, n, g)
        else:
            pick = filled[torch.randint(0, len(filled), (batch,), generator=g, device=DEV)]
            tabs = mutate_tms(unflatten(arch_gen[pick], n), g, n)
        o = run_tms(tabs, n, L, Tmax)
        insert(flatten(tabs, n), fitness(o), cell_ids(o, grid, L))
        best_trace.append(float(arch_fit.max()))
        if gen % max(1, gens // 12) == 0 or gen == gens - 1:
            print(f"  ME gen {gen:5d}: best depth {int(arch_fit.max()):8d} | coverage {int((arch_fit>-1).sum()):4d}/{ncells}")
    return arch_fit, arch_gen, best_trace


def evolution(n, L, Tmax, batch, gens, seed):
    """expEE baseline: single-objective depth hill-climb with elitism (the method that stalled at 43)."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    tabs = random_tms(batch, n, g)
    fit = fitness(run_tms(tabs, n, L, Tmax))
    flat = flatten(tabs, n)
    best = [float(fit.max())]
    nelite = max(2, batch // 5)
    for gen in range(gens):
        order = torch.argsort(fit, descending=True)
        elite = flat[order[:nelite]]
        reps = -(-batch // nelite)
        parents = elite.repeat(reps, 1)[:batch]
        child = mutate_tms(unflatten(parents, n), g, n)
        cfit = fitness(run_tms(child, n, L, Tmax))
        cflat = flatten(child, n)
        # keep best `batch` of elite+children
        allflat = torch.cat([elite, cflat]); allfit = torch.cat([fit[order[:nelite]], cfit])
        order2 = torch.argsort(allfit, descending=True)
        flat = allflat[order2[:batch]]; fit = allfit[order2[:batch]]
        best.append(float(fit.max()))
    return float(fit.max()), best


def sampling(n, L, Tmax, total, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    best = 0
    done = 0
    while done < total:
        b = min(8192, total - done)
        o = run_tms(random_tms(b, n, g), n, L, Tmax)
        best = max(best, int(fitness(o).max()))
        done += b
    return best


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--L", type=int, default=4096, help="bounded tape length")
    ap.add_argument("--Tmax", type=int, default=20000, help="depth cap in the search loop")
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--gens", type=int, default=800)
    ap.add_argument("--grid", type=int, default=24, help="MAP-Elites bins per behavior axis")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    if args.smoke:
        args.n, args.L, args.Tmax, args.batch, args.gens, args.grid = 4, 512, 2000, 512, 20, 12

    print(f"gpu_exp2_qd | device={dev_info()} | n={args.n} L={args.L} Tmax={args.Tmax} batch={args.batch} "
          f"gens={args.gens} grid={args.grid}\n")
    t0 = time.time()

    # fair budget: ME and EVO each do (gens+1)*batch evals; sampling matches.
    budget = (args.gens + 1) * args.batch
    print("  SAMPLING (random TMs, matched budget):")
    s_best = sampling(args.n, args.L, args.Tmax, budget, args.seed)
    print(f"    deepest halter = {s_best} steps   [{time.time()-t0:.0f}s]\n")

    print("  EVOLUTION (single-objective depth hill-climb = the expEE baseline):")
    e_best, e_trace = evolution(args.n, args.L, args.Tmax, args.batch, args.gens, args.seed)
    print(f"    deepest halter = {int(e_best)} steps   [{time.time()-t0:.0f}s]\n")

    print("  MAP-ELITES (quality-diversity over (span, ones) niches):")
    arch_fit, arch_gen, m_trace = map_elites(args.n, args.L, args.Tmax, args.batch, args.gens, args.grid, args.seed, args.out)
    m_best = int(arch_fit.max()); coverage = int((arch_fit > -1).sum())
    print(f"    deepest halter = {m_best} steps | archive coverage {coverage}/{args.grid**2}   [{time.time()-t0:.0f}s]\n")

    # inspect the deepest machine: structure (compressibility) of its space-time
    best_cell = int(arch_fit.argmax())
    best_tab = unflatten(arch_gen[best_cell:best_cell + 1], args.n)
    o = run_tms(best_tab, args.n, args.L, max(args.Tmax, m_best + 1))
    print("  === RESULT ===")
    print(f"    sampling {s_best}   evolution {int(e_best)}   MAP-Elites {m_best}   (deepest halter, steps)")
    verdict = ("MAP-Elites CROSSED the landscape wall (deeper than evolution)" if m_best > 1.3 * max(1, e_best)
               else "MAP-Elites did NOT beat evolution — the landscape wall holds even for QD")
    print(f"    VERDICT: {verdict}")
    print(f"    deepest machine: runtime={int(o['runtime'][0])} ones={int(o['ones'][0])} span={int(o['span'][0])}")
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "qd_result.json"), "w") as fh:
            json.dump(dict(n=args.n, sampling=s_best, evolution=int(e_best), map_elites=m_best, coverage=coverage,
                           m_trace=m_trace[::max(1, len(m_trace)//200)], e_trace=e_trace[::max(1, len(e_trace)//200)]), fh, indent=2)
        np.save(os.path.join(args.out, "archive_fitness.npy"), arch_fit.cpu().numpy().reshape(args.grid, args.grid))
        np.save(os.path.join(args.out, "deepest_tape.npy"), o["tape"][0].cpu().numpy())
    print("\n  READ: a large MAP-Elites>>evolution gap = QD's stepping-stones cross the rugged BB needle-landscape where")
    print("  hill-climbing stalls (a movable wall). A small gap = the landscape wall is fundamental to the discovery")
    print("  PROCESS, not just to naive search. Either is a clean, honest result on the one wall marked 'FIX: unknown'.")
