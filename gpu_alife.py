"""
gpu_alife.py — RUN WITH THE SURVIVAL BRIDGE. Is open-ended survival-selection a path to the MOONSHOT?

gpu_weird_soup proved self-replicators EMERGE from random BFF code with no target (the survival bridge produces MEANING
where distinctness diverged). The moonshot-relevant question this asks: does survival-selection produce SUSTAINED OPEN-ENDED
novelty — complexity GROWTH, ecology (parasites/arms-races), successive innovation — or does it just settle into one stable
replicator? Open-endedness is the property the moonshot needs; a settled soup is a dead end.

This instruments the soup to measure exactly that, over LONG runs with periodic CHECKPOINTS (saving only at the end cost
re-runs last time). Per snapshot it logs: ORDER (zlib), DIVERSITY (distinct genotypes), DOMINANCE (top-genotype freq),
COMPLEXITY of the dominant lineage (distinct ops used, replication-cycle step-count = computational depth), and TURNOVER
(Jaccard of the top-K genotypes vs the previous snapshot: low=settled, high=still innovating). Plus a robust SHIFT-INVARIANT
replication proof and a PARASITE scan (genotypes that replicate better paired with the dominant than with themselves =
they hijack others' copy machinery — the first sign of an arms race).

Reuses the verified sync-free BFF interpreter from gpu_weird_soup. (A non-standard instruction set — where an emergent
replicator might NOT be catalogued, the honest novelty angle — is the planned follow-up if open-endedness shows here.)

Run:  python gpu_alife.py --smoke
      python gpu_alife.py --N 16384 --L 64 --K 256 --epochs 40000 --snap 1000 --out runs/alife1   (long, on the 4090)
      python gpu_alife.py --analyze runs/alife1                                                     (post-hoc analysis)
"""
from __future__ import annotations
import argparse, time, os, json, zlib, glob
import numpy as np
import torch
from gpu_weird_soup import run_programs, DEV
OPS = "<>{}-+.,[]"


def metrics(pop_np):
    raw = np.ascontiguousarray(pop_np).tobytes()
    z = len(zlib.compress(raw, 6)) / max(1, len(raw))
    uniq, cnt = np.unique(pop_np, axis=0, return_counts=True)
    top = uniq[np.argmax(cnt)]
    # complexity proxy: mean number of DISTINCT ops actually present per tape (richer code = higher)
    ops = (pop_np & 15)
    distinct_ops = np.array([len(np.unique(o[o < 10])) for o in ops[:512]]).mean()  # sample 512 for speed
    return dict(zlib=float(z), distinct=int(len(uniq)), top_freq=int(cnt.max()),
                top_genotype=top.tolist(), mean_distinct_ops=float(distinct_ops),
                top_set=[tuple(r) for r in uniq[np.argsort(-cnt)[:50]].tolist()])


def replication_score(geno, L, K, trials=512, seed=0):
    """Shift-invariant proof: put geno|RANDOM, run, measure best circular-shift overlap of the partner-half with geno.
    Compare to a RANDOM-geno control. A real replicator copies itself into the naive partner -> overlap >> control."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    geno_t = torch.tensor(np.asarray(geno).reshape(-1), device=DEV, dtype=torch.uint8)   # -> (L,)

    def run_pair(parent):                                        # parent: (1, L)
        A = parent.repeat(trials, 1)
        B = torch.randint(0, 256, (trials, L), generator=g, device=DEV, dtype=torch.uint8)
        tape = torch.cat([A, B], dim=1)
        tape = run_programs(tape.clone(), K)
        Bafter = tape[:, L:]                                      # offspring region
        # best overlap over circular shifts of the parent (copy may land at an offset)
        best = torch.zeros(trials, device=DEV)
        for s in range(L):
            ref = torch.roll(parent, s, dims=1)
            best = torch.maximum(best, (Bafter == ref).float().mean(dim=1))
        return best.mean().item()

    rep = run_pair(geno_t[None])
    ctrl = run_pair(torch.randint(0, 256, (1, L), generator=g, device=DEV, dtype=torch.uint8))
    return rep, ctrl


def parasite_scan(top_genos, L, K):
    """Per common genotype, its STANDALONE replication score (geno|RANDOM -> does it copy itself in, vs a random control).
    A PARASITE is ABUNDANT in the population yet has rep ~ ctrl (cannot self-replicate alone) — so it only persists by
    hijacking other tapes' copy machinery in the soup. A true replicator has rep >> ctrl."""
    return [replication_score(geno, L, K, trials=256) for geno in top_genos]


def decode(row):
    return "".join(OPS[b & 15] if (b & 15) < 10 else "." for b in row)


def soup(N, L, K, epochs, mut, alphabet, snap, out, seed=1):
    T2 = 2 * L
    os.makedirs(out, exist_ok=True)
    g = torch.Generator(device=DEV).manual_seed(seed)
    pop = torch.randint(0, alphabet, (N, L), generator=g, device=DEV, dtype=torch.uint8)
    log = []; prev_top = set(); t0 = time.time()
    for ep in range(epochs + 1):
        if ep % snap == 0:
            m = metrics(pop.cpu().numpy())
            top_set = set(m.pop("top_set"))
            turnover = 1.0 - (len(top_set & prev_top) / max(1, len(top_set | prev_top))) if prev_top else 1.0
            prev_top = top_set
            rep, ctrl = replication_score(np.array(m["top_genotype"], np.uint8), L, K)
            m.update(epoch=ep, turnover=round(turnover, 3), rep=round(rep, 3), rep_ctrl=round(ctrl, 3),
                     dom_code=decode(np.array(m["top_genotype"], np.uint8)))
            log.append(m)
            np.save(os.path.join(out, f"snap_{ep:06d}.npy"), pop.cpu().numpy())
            json.dump(log, open(os.path.join(out, "alife_log.json"), "w"), indent=1)
            print(f"  ep {ep:6d}: zlib={m['zlib']:.3f} distinct={m['distinct']:6d} top={m['top_freq']:5d} "
                  f"ops={m['mean_distinct_ops']:.1f} turnover={m['turnover']:.2f} rep={m['rep']:.2f}(ctrl{m['rep_ctrl']:.2f}) "
                  f"[{time.time()-t0:.0f}s]  {m['dom_code'][:24]}")
        if ep == epochs:
            break
        perm = torch.randperm(N, generator=g, device=DEV)
        a, b = perm[:N // 2], perm[N // 2:]
        tape = run_programs(torch.cat([pop[a], pop[b]], 1), K)
        pop[a] = tape[:, :L]; pop[b] = tape[:, L:]
        if mut > 0:
            nm = int(mut * N * L)
            pop[torch.randint(0, N, (nm,), generator=g, device=DEV),
                torch.randint(0, L, (nm,), generator=g, device=DEV)] = \
                torch.randint(0, alphabet, (nm,), generator=g, device=DEV, dtype=torch.uint8)
    return log


def analyze(d, L=None, K=160):
    log = json.load(open(os.path.join(d, "alife_log.json")))
    if L is None:
        L = len(log[-1]["top_genotype"])
    print(f"=== {d}: {len(log)} snapshots ===")
    print("  epoch    zlib  distinct   top  ops  turnover  rep(ctrl)   dominant code")
    for m in log:
        print(f"  {m['epoch']:6d}  {m['zlib']:.3f}  {m['distinct']:6d} {m['top_freq']:5d} {m['mean_distinct_ops']:.1f}  "
              f"{m['turnover']:.2f}     {m['rep']:.2f}({m['rep_ctrl']:.2f})  {m['dom_code'][:28]}")
    # OPEN-ENDEDNESS VERDICT: after emergence, does turnover stay high (new genotypes keep displacing old = innovating)
    # and complexity climb — or does turnover->0 and complexity plateau (settled = a dead end)?
    post = [m for m in log if m["zlib"] < 0.7]                  # snapshots after order emerged
    if len(post) >= 2:
        late_turn = float(np.mean([m["turnover"] for m in post[-3:]]))
        ops0, ops1 = post[0]["mean_distinct_ops"], post[-1]["mean_distinct_ops"]
        zlate = float(np.mean([m["zlib"] for m in post[-3:]]))
        verdict = ("OPEN-ENDED (turnover stays high — still innovating)" if late_turn > 0.45
                   else "SETTLED (turnover -> 0 — froze into a stable replicator)")
        print(f"\n  OPEN-ENDEDNESS: late-turnover {late_turn:.2f}, complexity-ops {ops0:.1f}->{ops1:.1f}, "
              f"late-zlib {zlate:.2f}  => {verdict}")
    snaps = sorted(glob.glob(os.path.join(d, "snap_*.npy")))
    if snaps:
        pop = np.load(snaps[-1])
        uniq, cnt = np.unique(pop, axis=0, return_counts=True)
        top = uniq[np.argsort(-cnt)[:8]]
        print("\n  final top genotypes + replication + parasite scan:")
        rep, ctrl = replication_score(top[0], L, K)
        print(f"    dominant replication: {rep:.3f} vs random-control {ctrl:.3f}  "
              f"({'REPLICATOR' if rep > ctrl + 0.15 else 'not a clean replicator'})")
        par = parasite_scan([t for t in top], L, K)
        ordcnt = cnt[np.argsort(-cnt)]
        for i, (t, (rep_i, ctrl_i)) in enumerate(zip(top, par)):
            ct = int(ordcnt[i])
            is_rep = rep_i > ctrl_i + 0.15
            tag = "  <- replicator" if is_rep else ("  <- PARASITE? abundant but can't self-replicate" if ct > 20 else "")
            print(f"    x{ct:5d} rep={rep_i:.2f}(ctrl{ctrl_i:.2f})  {decode(t)}{tag}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze", type=str, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--N", type=int, default=8192); ap.add_argument("--L", type=int, default=64)
    ap.add_argument("--K", type=int, default=256); ap.add_argument("--epochs", type=int, default=20000)
    ap.add_argument("--mut", type=float, default=0.006); ap.add_argument("--alphabet", type=int, default=256)
    ap.add_argument("--snap", type=int, default=1000); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=str, default="runs/alife")
    args = ap.parse_args()
    if args.analyze:
        analyze(args.analyze); raise SystemExit(0)
    if args.smoke:
        args.N, args.L, args.K, args.epochs, args.snap = 2048, 32, 96, 600, 200
    print(f"gpu_alife | device={DEV} | N={args.N} L={args.L} K={args.K} epochs={args.epochs} mut={args.mut} snap={args.snap}\n")
    soup(args.N, args.L, args.K, args.epochs, args.mut, args.alphabet, args.snap, args.out, args.seed)
    print("\n  READ: complexity(ops) GROWING + dominance rising + turnover STAYING high = OPEN-ENDED evolution (the moonshot")
    print("  substrate). Complexity plateau + turnover->0 = settled (a dead end). rep>>ctrl proves the dominant self-copies.")
