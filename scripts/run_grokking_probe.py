"""Grokking probe: does the doctor recommend ABANDON before a slow-but-
reachable search reorganizes? (Joe's question, 2026-06-12.)

Setup designed to MAXIMIZE the doctor's chance to wrongly quit:
- reachable targets (planted, in the mold's op set — a path provably exists)
- an early min_gens (200) so the doctor diagnoses through the whole search
- a deceptive conjunction plant (out = product of two sub-motifs: until
  BOTH sub-motifs assemble, the product carries little signal — a
  building-block plateau, the closest thing to grokking in program space)
- plus several random hard plants (len 7) for breadth

The doctor here is OBSERVE-ONLY: it logs the verdict it WOULD give at every
diagnose point but never acts. We then ask: for each run, was there a
would-abandon verdict at any gen STRICTLY BEFORE the find? If yes, the
plateau-only doctor would have killed a real discovery.

Run from repo root:  python3 -m scripts.run_grokking_probe
"""

import json
import random
import sys
import time
from pathlib import Path

from engine import judge, registry
from engine.doctor import WallDoctor
from engine.proposers import EvolutionProposer
from engine.recorder import Recorder

# deceptive conjunction: out = (x ^ y) * (x + y); neither factor alone
# resembles the product -> a building-block plateau
DECEPTIVE_PLANT = (("XOR", 2, 0, 1), ("ADD", 3, 0, 1), ("MUL", 0, 2, 3))


def one_run(rec, label, pack_kwargs, generations=6000, min_gens=200,
            window=400, diagnose_every=50, pop_size=64, seed=0):
    pack, mold = registry.build("bitmixer", pack_kwargs)
    baselines = pack.baselines()
    total_bits = pack.w * len(pack.corpus)
    doctor = WallDoctor(min_gens, window, chance_margin=0.03)
    prop = EvolutionProposer(pop_size=pop_size)
    rng = random.Random(seed)
    ctx = {"rng": rng, "mold": mold, "batch": pop_size,
           "init_length": max(4, len(pack.reveal() or (0,)))}

    best_frac, best_cand = 0.0, None
    found_gen = None
    first_would_abandon = None
    diag_trace = []
    for gen in range(generations):
        results = []
        for cand in prop.propose(ctx):
            tidy, sc, _ = judge.score(pack, mold, cand)
            results.append((tidy, sc))
            frac = sc[1] / total_bits
            if frac > best_frac:
                best_frac, best_cand = frac, tidy
            if sc[0] == 1 and found_gen is None:
                found_gen = gen
        prop.feedback(results)
        doctor.observe(gen, best_frac)
        if gen and gen % diagnose_every == 0:
            hb = pack.heldout_frac(best_cand) if best_cand else None
            v = doctor.diagnose(baselines, heldout_best=hb)
            diag_trace.append({"gen": gen, "corpus": round(best_frac, 4),
                               "heldout": round(hb, 4) if hb else None,
                               "verdict": v["recommendation"] if v else None})
            if v and v["recommendation"] == "abandon-target" \
                    and first_would_abandon is None:
                first_would_abandon = gen
        if found_gen is not None:
            break

    killed = (first_would_abandon is not None
              and (found_gen is None or first_would_abandon < found_gen))
    out = {"label": label, "found_gen": found_gen,
           "first_would_abandon_gen": first_would_abandon,
           "would_have_killed_before_find": killed,
           "best_frac": round(best_frac, 4),
           "chance_plus": round(max(baselines.values()), 4),
           "trace": diag_trace}
    rec.event("probe", "run", payload={k: out[k] for k in out if k != "trace"})
    return out


def main():
    t0 = time.time()
    rec = Recorder(Path("runs") / f"grokking-probe-{int(t0)}",
                   f"grokking-probe-{int(t0)}")
    rows = []
    # deceptive conjunction plant, 3 search seeds
    for s in (0, 1, 2):
        r = one_run(rec, f"deceptive-conj s{s}",
                    {"target": "planted", "seed": 0, "plant": DECEPTIVE_PLANT},
                    seed=s)
        rows.append(r)
        print(f"deceptive s{s}: found@{r['found_gen']} "
              f"would_abandon@{r['first_would_abandon_gen']} "
              f"KILLED={r['would_have_killed_before_find']} "
              f"best={r['best_frac']} chance+={r['chance_plus']}")
    # random hard plants (len 7), distinct plant per seed
    for s in (0, 1, 2):
        r = one_run(rec, f"hard-rand s{s}",
                    {"target": "planted", "seed": 100 + s, "planted_len": 7},
                    seed=s)
        rows.append(r)
        print(f"hard-rand s{s}: found@{r['found_gen']} "
              f"would_abandon@{r['first_would_abandon_gen']} "
              f"KILLED={r['would_have_killed_before_find']} "
              f"best={r['best_frac']} chance+={r['chance_plus']}")

    n_killed = sum(r["would_have_killed_before_find"] for r in rows)
    summary = {"runs": len(rows), "killed_before_find": n_killed,
               "seconds": round(time.time() - t0, 2), "rows": rows}
    rec.event("probe", "summary",
              payload={k: summary[k] for k in
                       ("runs", "killed_before_find", "seconds")})
    (rec.run_dir / "report.json").write_text(json.dumps(summary, indent=2))
    rec.close()
    print(f"\n{n_killed}/{len(rows)} runs the plateau-only doctor would have "
          f"killed before the find. ({summary['seconds']}s)")
    # one detailed trace for the record
    ex = next((r for r in rows if r["found_gen"]), rows[0])
    print(f"\nexample trace [{ex['label']}], find@{ex['found_gen']}:")
    for d in ex["trace"][:14]:
        print(f"  gen {d['gen']:>4}: corpus={d['corpus']} "
              f"heldout={d['heldout']} verdict={d['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
