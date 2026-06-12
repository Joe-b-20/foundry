"""Calibration: rediscover Karatsuba (R=3) from outcome in the bilinear
mold, through the generic pipeline.

Two islands, the pattern validated on sorting networks and PCFs:
- blank: exact-first lexicographic rank (exact, -l1, -R, -dl). Finds the
  naive R=4 fast; tends to sit on it.
- pressure: scalar blended rank -(l1 + 0.45 R + 0.02 dl) — a near-miss
  R=3 outranks nothing exact lexicographically, but stays in the
  population pool, so the island can cross the R=4 -> R=3 valley through
  WRONG intermediates (the superoptimization lesson; wrong candidates
  cannot leave — only exact ones migrate, and verification is exact).
Migration: exact candidates only, every 20 generations, both ways.

PASS (predeclared): exact, verified R=3 in 3/3 seeds; recognition reported
either KNOWN(=karatsuba variant) by canonical key or UNRESOLVED with the
proven-optimal-rank note (flattening bound, computed exactly in
domains/bilinear_shelf.py). R<3 would CONTRADICT the proven bound and
means our own machinery is broken.

Run from repo root:  python3 -m scripts.run_karatsuba
"""

import dataclasses
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from engine import judge, registry
from engine.proposers import EvolutionProposer
from engine.recorder import Recorder
from domains import bilinear_shelf


@dataclass
class KaratsubaSpec:
    domain: str = "bilinear"
    domain_params: tuple = ()           # default target polymul2
    pop_size: int = 64
    generations: int = 4000
    migrate_every: int = 20
    init_rank: int = 4
    seed: int = 0
    cost_rules: str = "primary: multiplications R; secondary: dl; exact-only"
    schema: str = "karatsubaspec-v0"


def run_one(seed):
    t0 = time.time()
    spec = KaratsubaSpec(seed=seed)
    run_id = f"bilinear-karatsuba-s{seed}-{int(t0)}"
    rec = Recorder(Path("runs") / run_id, run_id)
    rec.event("foreman", "runspec", payload=dataclasses.asdict(spec),
              reason="predeclaration; per-candidate logging off at this "
                     "throughput — improvements/migrations/verdicts logged")
    pack, mold = registry.build("bilinear", {})
    shelf = bilinear_shelf.build_shelf(pack, mold)
    bounds = bilinear_shelf.BOUNDS["polymul2"]

    import random
    islands = {
        "blank": EvolutionProposer(pop_size=spec.pop_size),
        "pressure": EvolutionProposer(
            pop_size=spec.pop_size,
            key=lambda sc: sc[1] + 0.45 * sc[2] + 0.02 * sc[3]),
    }
    rngs = {nm: random.Random((seed << 8) + i)
            for i, nm in enumerate(islands)}

    evals = 0
    first_r4 = None
    found = None          # (gen, island, tidy)
    for gen in range(spec.generations):
        for nm, prop in islands.items():
            ctx = {"rng": rngs[nm], "mold": mold,
                   "batch": spec.pop_size, "init_length": spec.init_rank}
            results = []
            for cand in prop.propose(ctx):
                tidy, sc, _cost = judge.score(pack, mold, cand)
                results.append((tidy, sc))
                evals += 1
                if sc[0] == 1:
                    r = -sc[2]
                    if r == 4 and first_r4 is None:
                        first_r4 = (gen, nm)
                        rec.event("judge", "naive_rediscovered",
                                  payload={"gen": gen, "island": nm})
                    if r <= 3 and found is None:
                        found = (gen, nm, tidy)
            prop.feedback(results)
        if found:
            break
        if gen % spec.migrate_every == 0:
            for src in islands.values():
                b = src.best()
                if b and b[1][0] == 1:
                    for dst in islands.values():
                        if dst is not src:
                            dst.absorb([b])
        if gen % 200 == 0:
            rec.event("foreman", "gen_summary", payload={
                "gen": gen, "evals": evals,
                "bests": {nm: list(p.best()[1]) if p.best() else None
                          for nm, p in islands.items()}})

    report = {"run_id": run_id, "seed": seed, "evals": evals,
              "seconds": round(time.time() - t0, 2),
              "naive_r4_first": first_r4, "found_r3": bool(found)}
    if found:
        gen, nm, tidy = found
        ok, det = pack.verify_trusted(mold, tidy)
        key = mold.canonical_key(tidy)
        match = next((e for e in shelf if e["canonical"] == key), None)
        r = mold.native_cost(tidy)["mults"]
        if r < bounds["rank"]:
            label = "CONTRADICTS-PROVEN-BOUND (suspect our machinery!)"
        elif match:
            label = f"KNOWN = {match['name']} ({match['citation']})"
        else:
            label = ("UNRESOLVED: rank-3 exact, matches the proven optimal "
                     "rank (flattening bound, computed exactly); form not "
                     "identical to shelf variants under the small symmetry "
                     "group — wider canonicalization is future recognizer work")
        report.update({
            "gen": gen, "island": nm, "verified": ok,
            "pretty": mold.pretty(tidy), "mults": r,
            "certificate": det.get("certificate") if ok else det,
            "recognition": label})
        rec.event("judge", "verify_on_write", payload=det,
                  outcome="pass" if ok else "FAIL")
        rec.event("recognizer", "verdict", payload={"label": label})
    rec.event("foreman", "report", payload=report)
    (rec.run_dir / "report.json").write_text(json.dumps(report, indent=2))
    rec.close()
    return report


def main():
    results = [run_one(s) for s in (0, 1, 2)]
    print()
    ok_all = True
    for r in results:
        ok = r["found_r3"] and r.get("verified")
        ok_all &= ok
        print(f"seed {r['seed']}: {'OK' if ok else 'NOT FOUND'} "
              f"gen={r.get('gen')} island={r.get('island')} "
              f"evals={r['evals']} t={r['seconds']}s "
              f"naive_first={r['naive_r4_first']}")
        if r.get("pretty"):
            print(f"   {r['pretty']}")
            print(f"   -> {r['recognition']}")
    print(f"\nPASS: {ok_all} (predeclared bar: 3/3 seeds)")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
