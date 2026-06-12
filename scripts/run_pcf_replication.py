"""Calibration B, part 1: replicate the parent's Foundry-v1/v2 result
through the generic engine — seeded mutation walks from Catalan kappa-family
members that reach OTHER published family members, with per-generation
controls that void the run if they ever fail.

Parent result being replicated (its commits b088db9/23a53f1): catalan
B-mutants autonomously walked to adjacent published Table-7/kappa members;
controls 18/18 C+ after the v2 precision fix; clean control-gated null.

PASS here = (a) every control verifies in every generation (C+ throughout),
(b) at least one MUTANT (not a seed) classifies KNOWN(=ref) for a reference
that is not a seed — "walked to an adjacent published member",
(c) any UNRESOLVED-novel-flag is listed for review (with full Table-7 refs
loaded we expect zero, per the parent's v2).

Run from repo root:  python3 -m scripts.run_pcf_replication
"""

import dataclasses
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from domains.pcf import PCFPack
from engine.molds_pcf import PCFMold
from engine.recorder import Recorder


@dataclass
class WalkSpec:
    seeds: tuple = ("kappa(k=0,c=0)", "kappa(k=1,c=0)",
                    "kappa(k=0,c=1)", "kappa(k=1,c=1)")
    generations: int = 14
    children_per_gen: int = 8
    parents_kept: int = 4
    seed: int = 0
    screen_dps: int = 60
    verify_dps: int = 250
    cost_rules: str = ("primary: verified Mobius match (250-digit, "
                       "conjecture-grade); quality: delta high, dl low")
    schema: str = "pcfwalk-v0"


def fitness(label_rec, mold, cand):
    verified = 1 if label_rec.get("status") == "verified" else 0
    d = label_rec.get("delta")
    return (verified, d if d is not None else -5.0,
            -mold.native_cost(cand)["dl"])


def main():
    t0 = time.time()
    spec = WalkSpec(seed=int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    run_id = f"pcf-replication-s{spec.seed}-{int(t0)}"
    rec = Recorder(Path("runs") / run_id, run_id)
    rec.event("foreman", "runspec", payload=dataclasses.asdict(spec),
              reason="predeclaration: budgets, controls and cost rules fixed "
                     "before search")

    pack = PCFPack(spec.screen_dps, spec.verify_dps)
    mold = PCFMold()
    print("building/loading reference shelf (34 candidates, full pipeline)...")
    refs, rejects = pack.load_refs()
    rec.event("shelf", "built", payload={
        "verified": len(refs), "rejected": len(rejects),
        "rejects": rejects,
        "by_constant": {c: sum(1 for r in refs if r["constant"] == c)
                        for c in {r["constant"] for r in refs}}})
    print(f"shelf: {len(refs)} verified refs, {len(rejects)} rejects "
          f"({[r['reason'] for r in rejects][:5]}...)")

    from domains.pcf_shelf import factored_by_id
    factored = factored_by_id()
    # structural recognition map: DENSE form -> published member NAMES.
    # Factored forms are not unique (n^3*(n+2) == n^2*(n)*(n+2)), so the
    # canonical key for polynomial identity is the dense expansion — and
    # distinct published parameterizations can collide on the same
    # polynomial (kappa(1,0) == table7(1,1,0)), so values are alias LISTS.
    # Value-level reference subtraction can only name the Mobius CLASS;
    # naming the MEMBER needs form.
    structural = {}
    for rid, f in factored.items():
        structural.setdefault(mold.dense(mold.tidy(f)), []).append(rid)
    by_id = {r["id"]: r for r in refs}
    seed_ids = [s for s in spec.seeds if s in by_id and s in factored]
    assert len(seed_ids) >= 2, f"need >=2 verified seeds, got {seed_ids}"
    control_ids = ["rm-8-7zeta3", "kappa(k=0,c=0)", "kappa(k=1,c=0)"]
    controls = [(cid, factored[cid], by_id[cid]["constant"])
                for cid in control_ids if cid in by_id]
    assert len(controls) >= 3, "need 3 controls"

    rng = random.Random(spec.seed)
    cache = {}

    def classify_cached(cand):
        cand = mold.tidy(cand)
        if cand not in cache:
            cache[cand] = pack.classify(mold, cand)
        return cand, cache[cand]

    population = []
    seed_dense = set()
    for sid in seed_ids:
        cand, rec_label = classify_cached(factored[sid])
        seed_dense.add(mold.dense(cand))
        population.append((cand, fitness(rec_label, mold, cand)))

    reached, class_hits, novel_flags = set(), [], []
    controls_ok = controls_total = 0
    void = False
    for gen in range(spec.generations):
        # --- controls first: fresh full pipeline, never cached -----------
        for cid, ccand, cconst in controls:
            controls_total += 1
            crec = pack.classify(mold, ccand)
            good = (crec.get("status") == "verified"
                    and crec.get("constant") == cconst)
            controls_ok += int(good)
            rec.event("control", "check", payload={
                "gen": gen, "control": cid, "expected": cconst,
                "got": crec.get("constant"), "label": crec.get("label")},
                outcome="C+" if good else "C- VOID")
            if not good:
                void = True
        if void:
            rec.event("foreman", "stop",
                      reason="CONTROL FAILURE — batch void, halting "
                             "(parent PLAYBOOK rule: C- makes the universe "
                             "uninterpretable)")
            break
        # --- breed ---------------------------------------------------------
        parents = sorted(population, key=lambda r: r[1],
                         reverse=True)[: spec.parents_kept]
        children = []
        for _ in range(spec.children_per_gen):
            base = rng.choice(parents)[0]
            child = base
            for _ in range(rng.randint(1, 3)):
                child = mold.mutate(child, rng)
            children.append(child)
        for child in children:
            cand, lab = classify_cached(child)
            population.append((cand, fitness(lab, mold, cand)))
            dense = mold.dense(cand)
            aliases = (structural.get(dense, [])
                       if lab.get("status") == "verified" else [])
            rec.event("judge", "classify", payload={
                "gen": gen, "cand": mold.pretty(cand),
                "label": lab.get("label"), "member_by_form": aliases,
                "drop_reason": lab.get("drop_reason"),
                "constant": lab.get("constant"), "ref": lab.get("ref"),
                "delta": lab.get("delta")})
            # a member counts as reached only if its POLYNOMIAL is not a
            # seed's polynomial under any published name
            if aliases and dense not in seed_dense:
                name = " == ".join(aliases)
                if name not in reached:
                    reached.add(name)
                    rec.event("judge", "walked_to_published_member",
                              payload={"gen": gen, "member": name,
                                       "cand": mold.pretty(cand)},
                              outcome="replication signal (match BY FORM)")
                    print(f"  gen {gen}: mutant reached {name} by form "
                          f"({mold.pretty(cand)})")
            elif (not aliases and lab.get("label") == "KNOWN"):
                # verified, in a known Mobius orbit, but NOT a published
                # member's form — the parent's "boundary member" analogue
                class_hits.append({"gen": gen, "cand": mold.pretty(cand),
                                   "constant": lab.get("constant"),
                                   "orbit_rep": lab.get("ref")})
            if lab.get("label") == "UNRESOLVED-novel-flag":
                novel_flags.append({"gen": gen, "cand": mold.pretty(cand),
                                    "constant": lab.get("constant"),
                                    "rel": lab.get("rel")})
        population = sorted(population, key=lambda r: r[1],
                            reverse=True)[: spec.parents_kept * 3]
        print(f"gen {gen}: pop best {population[0][1]}, "
              f"classified {len(cache)} unique, reached {len(reached)}")

    drops = {}
    for lab in cache.values():
        if lab.get("label") == "DROPPED":
            drops[lab["drop_reason"]] = drops.get(lab["drop_reason"], 0) + 1
    report = {
        "run_id": run_id, "spec": dataclasses.asdict(spec),
        "seconds": round(time.time() - t0, 2),
        "void": void,
        "controls": {"ok": controls_ok, "total": controls_total},
        "shelf": {"verified": len(refs), "rejected": len(rejects)},
        "seeds": seed_ids,
        "unique_candidates_classified": len(cache),
        "reached_published_members_by_form": sorted(reached),
        "class_hits_known_orbit_new_form": class_hits,
        "novel_flags": novel_flags,
        "drop_histogram": drops,
        "PASS": (not void and controls_ok == controls_total
                 and len(reached) >= 1),
    }
    rec.event("foreman", "report", payload=report)
    (rec.run_dir / "report.json").write_text(json.dumps(report, indent=2))
    rec.close()
    print(json.dumps({k: report[k] for k in
                      ("controls", "reached_published_members_by_form",
                       "class_hits_known_orbit_new_form",
                       "novel_flags", "drop_histogram", "PASS", "seconds")},
                     indent=2))
    return 0 if report["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
