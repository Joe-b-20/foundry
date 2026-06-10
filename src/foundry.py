"""
foundry.py — THE CONJECTURE FOUNDRY (phase 5): a closed-loop, self-driving identity
hunter where SEARCH UNIVERSES are the organisms.

Joe's spec (2026-06-10), implemented with corrections: candidate-formula families are
genomes; fitness is multi-objective (verified relations, low height, delta/convergence
quality, POCKET coherence, proof-liftability, novelty-after-reference-subtraction,
minus rational-trap and duplicate penalties); and the key move — HITS MUTATE THE
SEARCH SPACE: a universe that produces verified structure spawns parameterized
neighbor-universes (factor shifts, exponent splits, A-factorization constraints,
index maps). Plus the two weird modes: UNKNOWN-CONSTANT clustering (cross-PSLQ among
high-delta unnamed values; discovered latent constants join the Mobius table for later
generations — the basis self-modifies) and PRIME-INDEXED PCFs (genome field
index_map='primes': A,B evaluated at the n-th prime — a family class nobody sweeps).

GENOME (a universe, not a single PCF):
  object_type: PCF (extensible)
  A_form:  ('dense', deg, crange)            — dense integer poly grid
           ('factored', (alpha,beta)-range, quad-crange) — (alpha n+beta)(quadratic)
  B_form:  ('monomial', sign, k)             — sign*n^k
           ('shifted', sign, k, c)           — sign*(n+c)^k
           ('split', sign, k1, k2, c)        — sign*n^k1*(n+c)^k2
           ('dense', deg, crange)
  index_map: 'id' | 'primes'
  terms, near_thresh, battery (inherited + latent K's)
POCKET = >=2 verified, non-reference-equivalent hits in one universe sharing B-form
and constant class. The moonshot target is a pocket, not a hit.

Run:  python src/foundry.py --smoke
      python src/foundry.py --gens 12 --out runs/foundry      (pod: GPU sweeps + CPU PSLQ)
"""
from __future__ import annotations
import argparse, json, os, time, itertools, random
import numpy as np

os.environ.setdefault("MPMATH_NOGMPY", "1")
import torch

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------- primes (for index_map)
def _primes(n):
    sieve = np.ones(max(15, int(n * (np.log(n + 2) + np.log(np.log(n + 3))) * 1.2)), bool)
    sieve[:2] = False
    for i in range(2, int(len(sieve) ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = False
    return np.nonzero(sieve)[0][:n].astype(np.float64)

PRIMES = _primes(3000)


# ---------------------------------------------------------------- GPU PCF evaluator
def eval_universe(Ac, Bfun, terms, index_map="id", want_delta=True):
    """Ac: (M,4) float64 A-coeffs [c0..c3]. Bfun(t)->float B value at index t.
    index_map 'primes': step t uses x=p_t (t-th prime) in A and B."""
    M = Ac.shape[0]
    p_prev = torch.ones(M, dtype=torch.float64, device=DEV)
    q_prev = torch.zeros(M, dtype=torch.float64, device=DEV)
    p = Ac[:, 0].clone(); q = torch.ones(M, dtype=torch.float64, device=DEV)
    val_prev = torch.full((M,), float("nan"), dtype=torch.float64, device=DEV)
    conv = torch.zeros(M, dtype=torch.bool, device=DEV)
    div = torch.zeros(M, dtype=torch.bool, device=DEV)
    logq = torch.zeros(M, dtype=torch.float64, device=DEV)
    last_err = torch.full((M,), float("nan"), dtype=torch.float64, device=DEV)
    for t in range(1, terms + 1):
        x = float(PRIMES[t - 1]) if index_map == "primes" else float(t)
        An = Ac[:, 0] + Ac[:, 1] * x + Ac[:, 2] * x * x + Ac[:, 3] * (x ** 3)
        Bn = torch.full_like(An, Bfun(x))
        p_new = An * p + Bn * p_prev
        q_new = An * q + Bn * q_prev
        p_prev, q_prev, p, q = p, q, p_new, q_new
        s = q.abs().clamp(min=1e-300)
        logq = logq + torch.log(s)
        p, q, p_prev, q_prev = p / s, q / s, p_prev / s, q_prev / s
        if t > 8:
            qs = torch.where(q.abs() < 1e-300, torch.full_like(q, 1e-300), q)
            v = p / qs
            d = (v - val_prev).abs()
            conv = conv | ((d < 1e-13) & torch.isfinite(v))
            div = div | ~torch.isfinite(v)
            last_err = torch.where(torch.isfinite(v), d, last_err)
            val_prev = v
    value = p / torch.where(q.abs() < 1e-300, torch.full_like(q, 1e-300), q)
    div = div | ~torch.isfinite(value)
    delta = -1.0 - torch.log(last_err.clamp(min=1e-308)) / logq.clamp(min=1.0)
    return value, conv & ~div, delta, logq


def a_grid(genome):
    f = genome["A_form"]
    if f[0] == "dense":
        deg, cr = f[1], f[2]
        r = np.arange(-cr, cr + 1, dtype=np.float64)
        if deg == 3:
            g = np.array(np.meshgrid(r, r, r, r, indexing="ij")).reshape(4, -1).T[:, ::-1]
        else:
            g = np.array(np.meshgrid(r, r, r, indexing="ij")).reshape(3, -1).T[:, ::-1]
            g = np.concatenate([g, np.zeros((len(g), 1))], 1)
        return np.ascontiguousarray(g)
    if f[0] == "factored":      # (alpha n + beta)(u n^2 + v n + w)
        ar, qr = f[1], f[2]
        rows = []
        rng_ab = range(-ar, ar + 1)
        rng_q = range(-qr, qr + 1)
        for al, be in itertools.product(rng_ab, repeat=2):
            if al == 0 and be == 0:
                continue
            for u, v, w in itertools.product(rng_q, repeat=3):
                if u == 0 and v == 0 and w == 0:
                    continue
                # (al n + be)(u n^2 + v n + w) = al*u n^3 + (al*v+be*u) n^2 + (al*w+be*v) n + be*w
                rows.append([be * w, al * w + be * v, al * v + be * u, al * u])
        return np.array(rows, dtype=np.float64)
    raise ValueError(f)


def b_fun(genome):
    f = genome["B_form"]
    if f[0] == "monomial":
        s, k = f[1], f[2]
        return (lambda x: s * x ** k), f"{s:+d}n^{k}"
    if f[0] == "shifted":
        s, k, c = f[1], f[2], f[3]
        return (lambda x: s * (x + c) ** k), f"{s:+d}(n{c:+d})^{k}"
    if f[0] == "split":
        s, k1, k2, c = f[1], f[2], f[3], f[4]
        return (lambda x: s * (x ** k1) * (x + c) ** k2), f"{s:+d}n^{k1}(n{c:+d})^{k2}"
    raise ValueError(f)


# ---------------------------------------------------------------- constants / table
def make_battery(latent):
    import mpmath as mp
    mp.mp.dps = 40
    bat = {"pi": mp.pi, "e": mp.e, "catalan": mp.catalan, "zeta3": mp.zeta(3),
           "gamma": mp.euler, "log2": mp.log(2), "zeta5": mp.zeta(5),
           "pi^2": mp.pi ** 2, "pi^3": mp.pi ** 3,
           "primezeta2": mp.primezeta(2), "primezeta3": mp.primezeta(3)}
    for k, v in latent.items():
        bat[k] = mp.mpf(v)
    return bat


def mobius_table(battery, R=12):
    coefs = np.arange(-R, R + 1)
    P, Q, Rr, S = (x.ravel().astype(np.float64) for x in
                   np.meshgrid(coefs, coefs, coefs, coefs, indexing="ij"))
    tv = []
    for cname, C in battery.items():
        Cf = float(C)
        den = Rr * Cf + S
        ok = np.abs(den) > 1e-9
        t = (P[ok] * Cf + Q[ok]) / den[ok]
        tv.append(np.unique(np.round(t[np.isfinite(t) & (np.abs(t) < 1e6)], 12)))
    return np.unique(np.concatenate(tv))


# ---------------------------------------------------------------- CPU verify + score
def verify_hits(cands, genome, battery, dps=150):
    """cands: list of (Acoeffs, v_float). PSLQ vs battery, rational trap, re-verify."""
    import mpmath as mp
    out = []
    Bf, _ = b_fun(genome)
    for A, vf in cands:
        mp.mp.dps = 60
        v = _eval_mp(A, Bf, genome, mp, terms=max(500, genome["terms"]))
        if v is None or not mp.isfinite(v):
            continue
        rr = mp.pslq([v, mp.mpf(1)], maxcoeff=10**7, maxsteps=8000)
        if rr and rr[0] != 0:
            out.append(dict(A=[int(x) for x in A], status="rational"))
            continue
        hits = []
        for cname, C in battery.items():
            rel = mp.pslq([mp.mpf(1), C, v, v * C], maxcoeff=10**5, maxsteps=8000)
            if rel and any(rel) and (rel[2] != 0 or rel[3] != 0):
                hits.append((cname, [int(x) for x in rel], max(abs(int(x)) for x in rel)))
        if not hits or len(hits) >= 3:
            out.append(dict(A=[int(x) for x in A], status="none", value=mp.nstr(v, 30)))
            continue
        hits.sort(key=lambda h: h[2])
        cname, rel, height = hits[0]
        mp.mp.dps = dps
        v2 = _eval_mp(A, Bf, genome, mp, terms=min(2800, max(1200, dps * 9)))
        C2 = make_battery({k: battery[k] for k in [] })  # rebuild fresh high-prec battery below
        C2 = make_battery({})[cname] if cname in make_battery({}) else mp.mpf(float(battery[cname]))
        resid = rel[0] + rel[1] * C2 + rel[2] * v2 + rel[3] * v2 * C2
        ok = abs(resid) < mp.mpf(10) ** (-(dps - 12))
        mp.mp.dps = 60
        out.append(dict(A=[int(x) for x in A], status="hit", constant=cname, relation=rel,
                        height=int(height), verified=bool(ok), value=mp.nstr(v, 30)))
    return out


def _eval_mp(A, Bf, genome, mp, terms=900):
    pp, qp = mp.mpf(1), mp.mpf(0)
    p, q = mp.mpf(int(A[0])), mp.mpf(1)
    for t in range(1, terms + 1):
        x = int(PRIMES[t - 1]) if genome["index_map"] == "primes" else t
        An = mp.mpf(int(A[0]) + int(A[1]) * x + int(A[2]) * x * x + int(A[3]) * x ** 3)
        Bn = mp.mpf(Bf(float(x)))
        p, pp = An * p + Bn * pp, p
        q, qp = An * q + Bn * qp, q
        if t % 4 == 0 and q != 0:
            s = q
            p, q, pp, qp = p / s, q / s, pp / s, qp / s
    return None if q == 0 else p / q


def reference_subtract(hit, ref_values, mp):
    """Mobius-equivalence of VALUES: hit is 'known' if PSLQ finds a small relation
    [1, v_ref, v, v*v_ref] for any reference value (catches sign/scale/shift/reparam)."""
    v = mp.mpf(hit["value"])
    for rname, rv in ref_values.items():
        rel = mp.pslq([mp.mpf(1), rv, v, v * rv], maxcoeff=10**4, maxsteps=6000)
        if rel and any(rel) and (rel[2] != 0 or rel[3] != 0):
            if max(abs(x) for x in rel) <= 5000:
                return rname
    return None


def liftability(delta, genome, logq_r2):
    s = 0.0
    if delta is not None and delta > 0:
        s += min(1.0, delta)                     # linear-forms-in-1-and-C signature
    if genome["B_form"][0] in ("monomial", "shifted", "split"):
        s += 0.5                                  # fully factored B
    s += 0.5 * max(0.0, logq_r2)                  # regular denominator growth
    return round(s, 3)


# ---------------------------------------------------------------- the foundry loop
SEED_UNIVERSES = [
    dict(name="zeta3-home", A_form=("dense", 3, 9), B_form=("monomial", -1, 6),
         index_map="id", terms=90, near=1e-10),
    dict(name="catalan-home", A_form=("dense", 2, 9), B_form=("monomial", -2, 4),
         index_map="id", terms=120, near=1e-10),   # -n^4 ~ catalan family kappa=0 (b=-2n^4 scaled in A)
    dict(name="zeta2-home", A_form=("dense", 2, 11), B_form=("monomial", 1, 4),
         index_map="id", terms=120, near=1e-10),
    dict(name="prime-monomial", A_form=("dense", 2, 6), B_form=("monomial", -1, 2),
         index_map="primes", terms=200, near=1e-9),
    dict(name="prime-cubic", A_form=("dense", 3, 4), B_form=("monomial", -1, 6),
         index_map="primes", terms=200, near=1e-9),
]


def mutate(genome, rng, gen):
    g = json.loads(json.dumps(genome))
    g["parent"] = genome["name"]
    moves = []
    f = g["B_form"]
    roll = rng.random()
    if roll < 0.3 and f[0] == "monomial":        # exponent split  n^k -> n^k1 (n+c)^k2
        k = f[2]
        if k >= 2:
            k1 = rng.randrange(1, k)
            g["B_form"] = ("split", f[1], k1, k - k1, rng.choice([-2, -1, 1, 2]))
            moves.append("B-split")
    elif roll < 0.55:                            # shift a factor
        if f[0] == "monomial":
            g["B_form"] = ("shifted", f[1], f[2], rng.choice([-2, -1, 1, 2]))
        elif f[0] in ("shifted", "split"):
            f = list(f); f[-1] = int(f[-1] + rng.choice([-1, 1])); g["B_form"] = tuple(f)
        moves.append("B-shift")
    elif roll < 0.7:                             # A factorization constraint
        g["A_form"] = ("factored", 4, 3)
        moves.append("A-factored")
    elif roll < 0.85:                            # budget tweak
        a = list(g["A_form"])
        if a[0] == "dense":
            a[2] = int(min(14, a[2] + rng.choice([2, 3])))
        g["A_form"] = tuple(a)
        moves.append("A-budget")
    else:                                        # index-map flip (the crazy axis)
        g["index_map"] = "primes" if g["index_map"] == "id" else "id"
        g["terms"] = 200 if g["index_map"] == "primes" else 100
        moves.append("index-map")
    g["name"] = f"{genome['name']}>g{gen}:{'+'.join(moves) or 'copy'}"
    return g


def run_universe(genome, table_t, battery, ref_values, mp, max_near=400):
    Bf, bdesc = b_fun(genome)
    A = a_grid(genome)
    Ac = torch.tensor(A, dtype=torch.float64, device=DEV)
    t0 = time.time()
    NEAR_A, NEAR_V = [], []
    n_conv = 0; n_total = len(A)
    bs = 400000
    best_delta = 0.0; delta_vals = []
    for s in range(0, len(Ac), bs):
        blk = Ac[s:s + bs]
        v, conv, delta, logq = eval_universe(blk, Bf, genome["terms"], genome["index_map"])
        fin = torch.isfinite(v) & (v.abs() < 1e6) & (v.abs() > 1e-6)
        ni = (v - v.round()).abs() < 1e-7
        keep = conv & fin & ~ni
        if keep.any():
            ki = keep.nonzero(as_tuple=True)[0]
            vk = v[ki]
            n_conv += len(ki)
            pos = torch.searchsorted(table_t, vk).clamp(0, len(table_t) - 1)
            pos0 = (pos - 1).clamp(0, len(table_t) - 1)
            dist = torch.minimum((vk - table_t[pos]).abs(), (vk - table_t[pos0]).abs())
            nm = dist < genome["near"]
            if nm.any():
                idx = nm.nonzero(as_tuple=True)[0]
                NEAR_A.append(blk[ki][idx].cpu().numpy()); NEAR_V.append(vk[idx].cpu().numpy())
            dd = delta[ki]
            good = torch.isfinite(dd) & (dd > 0.3) & ~nm
            if good.any():
                gi = good.nonzero(as_tuple=True)[0][:50]
                delta_vals.extend([(float(x), blk[ki][i].cpu().numpy().tolist())
                                   for x, i in zip(dd[gi].cpu(), gi.cpu())])
                best_delta = max(best_delta, float(dd[gi].max()))
    NA = np.concatenate(NEAR_A) if NEAR_A else np.zeros((0, 4))
    NV = np.concatenate(NEAR_V) if NEAR_V else np.zeros((0,))
    if len(NV) > max_near:
        o = np.argsort(NV)
        keep = np.linspace(0, len(NV) - 1, max_near).astype(int)
        NA, NV = NA[o][keep], NV[o][keep]
    hits = verify_hits(list(zip(NA, NV)), genome, battery)
    verified = [h for h in hits if h.get("status") == "hit" and h.get("verified")]
    rational = sum(1 for h in hits if h.get("status") == "rational")
    # reference subtraction + dedup within universe by (constant, relation up to sign)
    novel, seen = [], set()
    for h in verified:
        key = (h["constant"], tuple(np.sign(h["relation"][0]) * np.array(h["relation"])))
        if key in seen:
            continue
        seen.add(key)
        ref = reference_subtract(h, ref_values, mp)
        h["matches_reference"] = ref
        if ref is None:
            novel.append(h)
    # pocket: >=2 distinct verified non-reference hits, same constant
    from collections import Counter
    cc = Counter(h["constant"] for h in novel)
    pocket = max(cc.values()) if cc else 0
    fit = (2.0 * len(novel) + 0.5 * len(verified) + 3.0 * (pocket >= 2)
           + 0.3 * best_delta + liftability(best_delta, genome, 1.0)
           - 0.02 * rational - 0.2 * sum(1 for h in verified if h.get("matches_reference")))
    return dict(name=genome["name"], B=bdesc, index_map=genome["index_map"],
                n_total=int(n_total), n_conv=int(n_conv), n_near=int(len(NV)),
                n_verified=len(verified), n_novel=len(novel), pocket=int(pocket),
                rational=rational, best_delta=round(best_delta, 2),
                fitness=round(float(fit), 2), secs=round(time.time() - t0, 1),
                verified_hits=verified[:12], novel_hits=novel[:12],
                top_delta=sorted(delta_vals, reverse=True)[:8])


def latent_constants(reports, mp, max_vals=14):
    """UNKNOWN-CONSTANT MODE: cross-PSLQ among top-delta unnamed values across
    universes; any small relation clusters them -> christen a latent K."""
    vals = []
    for r in reports:
        for d, A in r.get("top_delta", [])[:3]:
            vals.append((r["name"], d, A))
    vals = vals[:max_vals]
    found = {}
    mp.mp.dps = 60
    cache = {}
    def val_of(entry):
        nm, d, A = entry
        key = tuple(A) + (nm,)
        if key not in cache:
            g = next((u for u in ALL_GENOMES if u["name"] == nm), None)
            if g is None:
                return None
            Bf, _ = b_fun(g)
            cache[key] = _eval_mp(A, Bf, g, mp, terms=max(500, g["terms"]))
        return cache[key]
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            v1, v2 = val_of(vals[i]), val_of(vals[j])
            if v1 is None or v2 is None or not (mp.isfinite(v1) and mp.isfinite(v2)):
                continue
            rel = mp.pslq([mp.mpf(1), v1, v2, v1 * v2], maxcoeff=2000, maxsteps=5000)
            if rel and any(rel) and (rel[2] != 0 or rel[3] != 0):
                kname = f"K{len(found)+1}_{vals[i][0][:12]}"
                found[kname] = float(v1)
    return found


ALL_GENOMES = []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--gens", type=int, default=10)
    ap.add_argument("--pop", type=int, default=6)
    ap.add_argument("--out", type=str, default="runs/foundry")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    import mpmath as mp
    rng = random.Random(7)
    pop = [dict(u) for u in SEED_UNIVERSES]
    if args.smoke:
        args.gens = 2
        for u in pop:
            if u["A_form"][0] == "dense":
                u["A_form"] = (u["A_form"][0], u["A_form"][1], min(4, u["A_form"][2]))
    latent = {}
    # reference values: controls + known-family members (Apery, RM zeta3, Catalan kappa-family, zeta2)
    mp.mp.dps = 60
    refs = {"apery": 6 / mp.zeta(3), "rm_zeta3": 8 / (7 * mp.zeta(3)),
            "catalan_k0": 1 / (2 * mp.catalan), "zeta2": 30 / mp.pi ** 2,
            "brouncker": 4 / mp.pi}
    # the PUBLISHED Catalan family (arXiv:2210.15669): a=3n^2+(3+4k)n+(2k+1),
    # b=-2n^2(n+2k)(n+c) — add k,c in 0..3 as FAMILY references (the gen6 lesson:
    # reference single values miss family members)
    def _famval(kk, cc, terms=2500):
        pp, qp = mp.mpf(1), mp.mpf(0)
        p, q = mp.mpf(2 * kk + 1), mp.mpf(1)
        for t in range(1, terms + 1):
            An = mp.mpf(3 * t * t + (3 + 4 * kk) * t + 2 * kk + 1)
            Bn = mp.mpf(-2 * t * t * (t + 2 * kk) * (t + cc))
            p, pp = An * p + Bn * pp, p
            q, qp = An * q + Bn * qp, q
            if t % 4 == 0 and q != 0:
                s = q; p, q, pp, qp = p / s, q / s, pp / s, qp / s
        return p / q if q != 0 else mp.mpf(0)
    for kk in range(4):
        for cc in range(4):
            refs[f"cat_fam_k{kk}c{cc}"] = _famval(kk, cc)
    history = []
    for gen in range(args.gens):
        battery = make_battery(latent)
        table = mobius_table(battery)
        table_t = torch.tensor(table, dtype=torch.float64, device=DEV)
        reports = []
        for u in pop:
            ALL_GENOMES.append(u)
            r = run_universe(u, table_t, battery, refs, mp)
            reports.append(r)
            print(f"  g{gen} {r['name'][:46]:46s} B={r['B']:16s} im={r['index_map'][:2]} "
                  f"conv={r['n_conv']:>8,} near={r['n_near']:>4} ver={r['n_verified']:>3} "
                  f"novel={r['n_novel']} pocket={r['pocket']} fit={r['fitness']:>6.2f} [{r['secs']}s]", flush=True)
        # unknown-constant clustering every 3rd gen
        if gen % 3 == 2:
            newk = latent_constants(reports, mp)
            for k, v in newk.items():
                if all(abs(v - float(x)) > 1e-9 for x in latent.values()):
                    latent[k] = v
            if newk:
                print(f"  g{gen} LATENT constants christened: {list(newk)} (battery grows to {len(make_battery(latent))})", flush=True)
        history.append(dict(gen=gen, reports=reports, latent=dict(latent)))
        json.dump(history, open(os.path.join(args.out, "foundry_log.json"), "w"), indent=1)
        # selection + mutation: top half survive, each spawns one mutant
        reports_sorted = sorted(zip(pop, reports), key=lambda pr: -pr[1]["fitness"])
        keep = [p for p, _ in reports_sorted[:max(2, args.pop // 2)]]
        pop = keep + [mutate(p, rng, gen + 1) for p in keep]
        pop = pop[:args.pop]
        # add verified novel hits' refs so we don't re-celebrate them
        for _, r in reports_sorted:
            for h in r["novel_hits"]:
                refs[f"own_{len(refs)}"] = mp.mpf(h["value"])
    print("\nFOUNDRY done. Pockets, novel hits and latent constants are in foundry_log.json")


if __name__ == "__main__":
    main()
