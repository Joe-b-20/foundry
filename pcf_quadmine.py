"""
pcf_quadmine.py — RE-MINE the existing PCF survivor sets with a BIGGER NET (phase 3, local CPU).

Tier-2 of the identity campaign: every past GPU sweep saved its survivors; this re-mines
them retroactively with (a) an EXTENDED constant battery, (b) QUADRATIC integer relations
PSLQ([1, C, C^2, v, vC, vC^2]) — catching v = (p+qC+rC^2)/(s+tC+uC^2) identity classes the
Mobius-only stage 2 was blind to — and (c) mpmath.identify() on the most interesting
unknowns (inverse symbolic computation). Runs on the LOCAL 20-core box (pod CPUs are slow).

Inputs: any stage1_survivors.npz / stage1_blindsample.npz produced by gpu_pcf_hunt.py
(poly / n6 / p4 / gen6 families all supported).
Discipline: a quadratic hit is reported ONLY if it re-verifies at --dps digits; rational
trap (pslq([v,1])) still applied; >=3-constant simultaneous matches still rejected; known
forms (Apery, RM 8/(7zeta3), 30/pi^2, 4/pi) tagged as controls/rediscoveries.

Run:  MPMATH_NOGMPY=1 python pcf_quadmine.py --inputs runs_pod/phase2/pcf_n6/stage1_survivors.npz \
          runs_pod/phase2/pcf_n6big/stage1_survivors.npz --procs 18 --dps 220 --out runs/quadmine
"""
from __future__ import annotations
import argparse, os, json, time
import numpy as np

os.environ.setdefault("MPMATH_NOGMPY", "1")


def _constants(mp):
    return {
        "pi": mp.pi, "e": mp.e, "catalan": mp.catalan, "zeta3": mp.zeta(3),
        "gamma": mp.euler, "log2": mp.log(2), "log3": mp.log(3),
        "zeta5": mp.zeta(5), "zeta7": mp.zeta(7),
        "sqrt2": mp.sqrt(2), "phi": (1 + mp.sqrt(5)) / 2,
    }


KNOWN_FORMS = {
    (5, 27, 51, 34): "6/zeta3 [Apery]", (1, 5, 9, 6): "8/(7zeta3) [RM 2021]",
    (-5, -27, -51, -34): "mirror Apery", (-1, -5, -9, -6): "mirror RM",
    (3, 11, 11, 0): "30/pi^2 [Apery z2]", (1, 2, 0, 0): "4/pi [Brouncker]",
}


def _eval_pcf(A, B, mp, family, terms=900):
    p_prev, q_prev = mp.mpf(1), mp.mpf(0)
    p, q = mp.mpf(int(A[0])), mp.mpf(1)
    for n in range(1, terms + 1):
        An = mp.mpf(0)
        for k in range(len(A)):
            An += int(A[k]) * mp.mpf(n) ** k
        if family == "n6":
            Bn = -mp.mpf(n) ** 6
        elif family == "n4":
            Bn = -mp.mpf(n) ** 4
        elif family == "p4":
            Bn = mp.mpf(n) ** 4
        elif family == "gen6":
            Bn = mp.mpf(0)
            for k in range(len(B)):
                Bn += int(B[k]) * mp.mpf(n) ** k
        else:  # poly (deg<=2 B)
            Bn = mp.mpf(int(B[0]) + int(B[1]) * n + int(B[2]) * n * n)
        p, p_prev = An * p + Bn * p_prev, p
        q, q_prev = An * q + Bn * q_prev, q
        if n % 4 == 0 and q != 0:
            s = q
            p, q, p_prev, q_prev = p / s, q / s, p_prev / s, q_prev / s
    if q == 0:
        return None
    return p / q


def _mine_chunk(args):
    import os as _os
    _os.environ["MPMATH_NOGMPY"] = "1"
    import mpmath as mp
    A_list, B_list, family, dps_verify = args
    mp.mp.dps = 60
    consts = _constants(mp)
    out = []
    for A, B in zip(A_list, B_list):
        v = _eval_pcf(A, B, mp, family)
        if v is None or not mp.isfinite(v) or abs(v) > 1e8:
            continue
        # rational trap
        rr = mp.pslq([v, mp.mpf(1)], maxcoeff=10**7, maxsteps=10**4)
        if rr and rr[0] != 0:
            continue
        hits = []
        for cname, C in consts.items():
            # Mobius (4-term) first — cheap
            rel = mp.pslq([mp.mpf(1), C, v, v * C], maxcoeff=10**5, maxsteps=8000)
            if rel and any(rel) and (rel[2] != 0 or rel[3] != 0):
                hits.append((cname, "mobius", [int(x) for x in rel], max(abs(int(x)) for x in rel)))
                continue
            # QUADRATIC (6-term): 1, C, C^2, v, vC, vC^2.
            # TRAP (the quadratic analog of the rational-Mobius trap, caught in smoke):
            # for an ALGEBRAIC constant, the minimal polynomial makes
            # rel = [0,0,0, m0,m1,m2] a VALUE-INDEPENDENT relation (e.g. 1+phi-phi^2=0
            # multiplies v). Guard 1: skip algebraic battery members for the quad pass.
            # Guard 2: require the v-coefficient polynomial rel[3]+rel[4]C+rel[5]C^2
            # to be numerically nonzero (the relation must actually constrain v).
            if cname in ("sqrt2", "phi", "sqrt3"):
                continue
            rel6 = mp.pslq([mp.mpf(1), C, C * C, v, v * C, v * C * C], maxcoeff=10**4, maxsteps=12000)
            if rel6 and any(rel6) and any(rel6[3:]):
                vden = rel6[3] + rel6[4] * C + rel6[5] * C * C
                if abs(vden) > mp.mpf(10) ** -10:
                    hits.append((cname, "quad", [int(x) for x in rel6], max(abs(int(x)) for x in rel6)))
        if not hits or len(hits) >= 3:
            continue
        hits.sort(key=lambda h: h[3])
        cname, kind, rel, height = hits[0]
        # re-verify at high precision
        mp.mp.dps = dps_verify
        v2 = _eval_pcf(A, B, mp, family, terms=1400)
        C2 = _constants(mp)[cname]
        if kind == "mobius":
            resid = rel[0] + rel[1] * C2 + rel[2] * v2 + rel[3] * v2 * C2
        else:
            resid = (rel[0] + rel[1] * C2 + rel[2] * C2 * C2 + rel[3] * v2
                     + rel[4] * v2 * C2 + rel[5] * v2 * C2 * C2)
        ok = abs(resid) < mp.mpf(10) ** (-(dps_verify - 12))
        mp.mp.dps = 60
        tag = KNOWN_FORMS.get(tuple(int(x) for x in (list(A) + [0, 0, 0, 0])[:4]), "")
        out.append(dict(A=[int(x) for x in A], B=[int(x) for x in B], family=family,
                        value=mp.nstr(v, 30), constant=cname, kind=kind, relation=rel,
                        height=height, verified=bool(ok), known=tag))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--procs", type=int, default=18)
    ap.add_argument("--dps", type=int, default=220)
    ap.add_argument("--limit", type=int, default=0, help="cap per input (0=all)")
    ap.add_argument("--out", type=str, default="runs/quadmine")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    from multiprocessing import Pool
    t0 = time.time()
    all_results = []
    for path in args.inputs:
        d = np.load(path, allow_pickle=False)
        A, B, V = d["A"], d["B"], d["V"]
        family = str(d["family"][0]) if "family" in d.files else "poly"
        n = len(A)
        if args.limit and n > args.limit:
            idx = np.linspace(0, n - 1, args.limit).astype(int)
            A, B, V = A[idx], B[idx], V[idx]
            n = args.limit
        print(f"  [{path}] family={family} candidates={n:,}")
        sz = max(1, -(-n // (args.procs * 6)))
        chunks = [(A[s:s + sz].tolist(), B[s:s + sz].tolist(), family, args.dps)
                  for s in range(0, n, sz)]
        with Pool(args.procs) as pool:
            for i, r in enumerate(pool.imap_unordered(_mine_chunk, chunks)):
                all_results.extend(r)
                if (i + 1) % 10 == 0 or i + 1 == len(chunks):
                    print(f"    chunk {i+1}/{len(chunks)}  hits {len(all_results)}  [{time.time()-t0:.0f}s]")
        json.dump(all_results, open(os.path.join(args.out, "quadmine_hits.json"), "w"), indent=1)
    # summary
    from collections import Counter
    ver = [r for r in all_results if r["verified"]]
    novel = [r for r in ver if not r["known"]]
    quad = [r for r in ver if r["kind"] == "quad"]
    print(f"\n  QUADMINE done: {len(all_results)} raw hits, {len(ver)} verified at {args.dps} digits")
    print(f"  by constant: {dict(Counter(r['constant'] for r in ver))}")
    print(f"  by kind:     {dict(Counter(r['kind'] for r in ver))}")
    print(f"  KNOWN forms among verified: {len([r for r in ver if r['known']])}; quad-relation hits: {len(quad)}")
    print(f"  verified hits with NO known-form tag: {len(novel)} (tag absence != novelty — literature-check each)")
    for r in novel[:25]:
        print(f"    [{r['kind']}] {r['constant']:8s} A={r['A']} B={r['B']} v={r['value'][:24]} rel={r['relation']} h={r['height']}")
    json.dump(dict(n_raw=len(all_results), n_verified=len(ver),
                   by_constant=dict(Counter(r['constant'] for r in ver)),
                   by_kind=dict(Counter(r['kind'] for r in ver)),
                   untagged_verified=novel),
              open(os.path.join(args.out, "quadmine_summary.json"), "w"), indent=1)
    print(f"  wrote {args.out}/quadmine_hits.json + quadmine_summary.json")


if __name__ == "__main__":
    main()
