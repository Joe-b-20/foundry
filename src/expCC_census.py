"""
expCC_census.py — ANTI-OCCAM: invert the project's deepest bias (minimal description) and census the WHOLE space of
correct length-generalizing digit-serial adders. The project's expB rewarded the SHORTEST program -> carry (3 ops); every
result since prefers minimality. Question: is carry the UNIQUE length-general adder, or is there a ZOO of baroque-but-correct
ones humans never wrote because they had no reason to? Anti-Occam = look for the LARGEST genuinely-distinct correct adder.

Method (a superoptimizer-style CENSUS with a PROOF of length-gen, not a search for one program):
 - VM (same shape as expB): persistent register C threaded across LSB-first positions; per position a straight-line program
   over inputs {A,B,C} and constants {0,1,BASE} computes an OUTPUT digit and the NEXT register C'. Position-independent =>
   any CORRECT-as-a-finite-transducer program length-generalizes by construction.
 - Bottom-up enumerate DISTINCT value-functions over the finite domain (a,b,c), observational-equivalence dedup (expU trick),
   tracking the minimal op-DAG (shared subexpressions) per function.
 - A candidate adder = a pair (OUT-function, C'-function) + initial register c0. It is a CORRECT adder FOR ALL LENGTHS iff
   its encoded-carry transducer BISIMULATES the true 2-state carry transducer (out=(a+b+k)%B, carry=(a+b+k)//B). The
   bisimulation is computed over the finite reachable product => a PROOF (two bisimilar FSMs agree on all inputs of every
   length), in the project's exact-verification spirit. No sampling.
 - ANTI-OCCAM readout: among all DISTINCT correct adders, the size spectrum (min..max joint op-count) and the distinct CARRY
   ENCODINGS (incl. redundant-state ones). If the max is just carry-with-detours, correctness+length-gen ALONE pins carry
   (Occam wasn't doing the work); if a genuinely larger distinct algorithm exists, that's a "procedure humans skipped".
Run: python expCC_census.py   (pure numpy; small base for a near-complete census)
"""
from __future__ import annotations
import argparse
from collections import deque
import numpy as np

def build_pool(B, CMAX, VCAP, ops, G, Ncap, verbose=True):
    cs = CMAX + 1
    ai, bi, ci = np.meshgrid(np.arange(B), np.arange(B), np.arange(cs), indexing="ij")
    A = ai.ravel().astype(np.int64); Bv = bi.ravel().astype(np.int64); C = ci.ravel().astype(np.int64)
    leaves = [("A", A), ("B", Bv), ("C", C), ("0", np.zeros_like(A)), ("1", np.ones_like(A)), ("K", np.full_like(A, B))]
    arrs, opsin, size, descr, isop = [], [], [], [], []
    seen = {}
    for nm, arr in leaves:
        seen[arr.tobytes()] = len(arrs)
        arrs.append(arr); opsin.append(frozenset()); size.append(0); descr.append(nm); isop.append(False)

    def apply(op, x, y):
        with np.errstate(divide="ignore", invalid="ignore"):
            if op == "+":  return x + y
            if op == "-":  return x - y
            if op == "*":  return x * y
            if op == "//": return np.where(y != 0, np.floor_divide(x, y), 0)
            if op == "%":  return np.where(y != 0, np.mod(x, y), x)
            if op == ">=": return (x >= y).astype(np.int64)
            if op == "mn": return np.minimum(x, y)
            if op == "mx": return np.maximum(x, y)
        raise ValueError(op)

    frontier = list(range(len(arrs)))
    capped = False
    for gen in range(1, G + 1):
        newly = []
        snapshot = len(arrs)                      # combine frontier x (pool as it was at gen start) — NOT the growing list
        for i in frontier:
            for j in range(snapshot):
                for op in ops:
                    r = apply(op, arrs[i], arrs[j])
                    if np.abs(r).max() > VCAP:
                        continue
                    key = r.tobytes()
                    cand_ops = opsin[i] | opsin[j]
                    if key in seen:
                        fid = seen[key]
                        cs_ = len(cand_ops | {fid})
                        if isop[fid] and cs_ < size[fid]:
                            opsin[fid] = cand_ops | {fid}; size[fid] = cs_
                            descr[fid] = f"({descr[i]}{op}{descr[j]})"
                        continue
                    fid = len(arrs)
                    seen[key] = fid
                    arrs.append(r); opsin.append(cand_ops | {fid}); size.append(len(cand_ops | {fid}))
                    descr.append(f"({descr[i]}{op}{descr[j]})"); isop.append(True)
                    newly.append(fid)
                    if len(arrs) >= Ncap:
                        capped = True; break
                if capped: break
            if capped: break
        if verbose:
            print(f"    gen {gen}: pool={len(arrs)}  (+{len(newly)})" + ("  [CAPPED]" if capped else ""), flush=True)
        frontier = newly
        if capped or not newly:
            break
    return arrs, opsin, size, descr, isop, (B, cs)


def correct_adder(out3, cp3, c0, B, CMAX, state_limit=8):
    """Bisimulate the candidate (encoded-carry) transducer against the true 2-state carry transducer.
    Returns (#encoded states) if it is a correct adder for ALL lengths, else None. PROOF: bisimilar FSMs agree on every
    input of every length, so finite-product consistency => exact length-generalization."""
    if not (0 <= c0 <= CMAX):
        return None
    mp = {c0: 0}; dq = deque([c0])
    while dq:
        c = dq.popleft(); k = mp[c]
        for a in range(B):
            for b in range(B):
                s = a + b + k
                if out3[a, b, c] != s % B:
                    return None
                nc = cp3[a, b, c]
                if not (0 <= nc <= CMAX):
                    return None
                tk = s // B                      # true next carry in {0,1}
                if nc in mp:
                    if mp[nc] != tk:
                        return None
                else:
                    mp[nc] = tk; dq.append(nc)
        if len(mp) > state_limit:
            return None
    return len(mp)


def census(B=5, CMAX=11, VCAP=24, rich=True, G=4, Ncap=1600):
    ops = ["+", "-", "*", "//", "%", ">=", "mn", "mx"] if rich else ["+", "-", "*", ">=", "mn", "mx"]
    tag = "RICH (incl. // %)" if rich else "LEAN (no // %)"
    print(f"\n===== CENSUS  base={B}  ops={tag}  Gmax={G}  Ncap={Ncap} =====", flush=True)
    arrs, opsin, size, descr, isop, (B_, cs) = build_pool(B, CMAX, VCAP, ops, G, Ncap)
    arr3 = [a.reshape(B, B, cs) for a in arrs]
    seeds = [0, 1, B]
    # prefilter OUT candidates: some seed c0 makes the FIRST column correct (a+b mod B), necessary for any correct adder
    out_cands = []
    for fid in range(len(arrs)):
        for c0 in seeds:
            if 0 <= c0 <= CMAX and np.all(arr3[fid][:, :, c0] == (np.add.outer(np.arange(B), np.arange(B)) % B)):
                out_cands.append((fid, c0))
    print(f"    pool={len(arrs)}   viable OUT-candidates (correct first column for some c0)={len(out_cands)}", flush=True)

    found = {}   # (out_fid, cp_fid, c0) -> (states, joint_size)
    for (ofid, c0) in out_cands:
        o3 = arr3[ofid]
        for cfid in range(len(arrs)):
            st = correct_adder(o3, arr3[cfid], c0, B, CMAX)
            if st is not None:
                joint = len(opsin[ofid] | opsin[cfid])
                found[(ofid, cfid, c0)] = (st, joint)
    if not found:
        print("    NO correct adder found in this pool.", flush=True)
        return found, descr, size, opsin

    distinct_fn = set((o, c) for (o, c, _) in found)
    joints = [v[1] for v in found.values()]
    states = [v[0] for v in found.values()]
    print(f"    CORRECT ADDERS: {len(found)} (out,cp,c0) triples;  {len(distinct_fn)} distinct (OUT-fn,C'-fn) machines", flush=True)
    print(f"    joint op-count spectrum: min={min(joints)}  max={max(joints)};  #encoded-carry-states seen: {sorted(set(states))}", flush=True)

    # show the MINIMAL adder (carry) and the MAXIMAL distinct one (anti-Occam), and the distinct state-count encodings
    items = sorted(found.items(), key=lambda kv: (kv[1][1], kv[1][0]))
    (o, c, c0), (st, js) = items[0]
    print(f"    MINIMAL  joint={js} states={st} c0={c0}:  OUT={descr[o]}   C'={descr[c]}", flush=True)
    (o, c, c0), (st, js) = items[-1]
    print(f"    MAXIMAL  joint={js} states={st} c0={c0}:  OUT={descr[o]}   C'={descr[c]}", flush=True)
    # distinct by #states (redundant-encoding adders): one exemplar per state-count
    by_states = {}
    for (o, c, c0), (st, js) in found.items():
        by_states.setdefault(st, (o, c, c0, js))
    print(f"    distinct encodings by #carry-states:", flush=True)
    for st in sorted(by_states):
        o, c, c0, js = by_states[st]
        print(f"      {st} states (joint {js}, c0={c0}): OUT={descr[o]}  C'={descr[c]}", flush=True)
    return found, descr, size, opsin


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=int, default=5)
    ap.add_argument("--G", type=int, default=4)
    ap.add_argument("--Ncap", type=int, default=1600)
    args = ap.parse_args()
    print("ANTI-OCCAM census of correct length-generalizing digit-serial adders (bisimulation PROOF of length-gen).")
    census(B=args.base, rich=True, G=args.G, Ncap=args.Ncap)
    census(B=args.base, rich=False, G=args.G + 1, Ncap=args.Ncap)   # lean set forces larger programs; allow one more gen
    print("\n  READ: if MAX joint size is just carry-with-detours and all machines bisimulate the 2-state carry, then")
    print("  CORRECTNESS+LENGTH-GEN (not Occam) is what pins addition; anti-Occam reveals only re-encodings, no new algorithm.")
