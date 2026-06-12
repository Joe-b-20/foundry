"""Island-ecology driver v0 (Joe's injection #1, adopted 2026-06-11).

Islands run in parallel roles and share survivors ONLY through the archive:
- blank    (A): random starts; pulls only outcome-only entries. Its whole
                lineage is discovered-from-outcome by induction.
- seeded   (B): starts from the reference shelf; pulls anything. Everything
                it touches is labelled "seeded" — improvements, not
                discoveries.
- pressure (C): Island-C rank — smallness outranks sortedness, so it may
                pass through wrong intermediates inside the island. Pulls
                only outcome-only entries, so its lineage stays
                outcome-only. Wrong candidates cannot leave: the archive
                verifies on write.
Island D (alien molds) needs a second mold to mean anything; it arrives
with domains B/C. v0 tracks provenance at island granularity — the
per-candidate lineage graph is future work.
"""

import dataclasses
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

from engine import judge, recognizer
from engine.archive import Archive
from engine.molds import ComparatorMold
from engine.proposers import EvolutionProposer
from engine.recorder import Recorder
from domains.sorting_networks import SortingNetworkPack
from domains import sorting_networks_shelf as shelf_mod

PROV = {"blank": "outcome-only", "pressure": "outcome-only", "seeded": "seeded"}
PULL = {"blank": "outcome-only", "pressure": "outcome-only", "seeded": None}


@dataclass
class IslandsSpec:
    domain: str
    domain_params: dict
    roles: tuple = ("blank", "seeded", "pressure")
    pop_size: int = 64
    generations: int = 600
    migrate_every: int = 25
    init_length: int = 15
    seed: int = 0
    target_outcome_size: int = 0      # 0 = no early stop
    cost_rules: str = "primary: comparators (size); secondary: depth"
    schema: str = "islandsspec-v0"


def run_islands(spec: IslandsSpec, runs_root="runs"):
    t0 = time.time()
    n = spec.domain_params.get("n")
    run_id = f"{spec.domain}-n{n}-islands-s{spec.seed}-{int(t0)}"
    rec = Recorder(Path(runs_root) / run_id, run_id)
    rec.event("foreman", "runspec", payload=dataclasses.asdict(spec),
              reason="predeclaration: budgets, roles and cost rules fixed before search")

    assert spec.domain == "sorting_networks", "v0 knows exactly one domain"
    pack = SortingNetworkPack(**spec.domain_params)
    mold = ComparatorMold(pack.n)
    archive = Archive(spec.domain, pack, mold)
    shelf = shelf_mod.build_shelf(pack, mold)
    bounds = shelf_mod.BOUNDS.get(pack.n)

    islands = {}
    for i, role in enumerate(spec.roles):
        seeds = [e["canonical"] for e in shelf] if role == "seeded" else []
        islands[f"{role}-{i}"] = {
            "role": role,
            "prop": EvolutionProposer(pop_size=spec.pop_size,
                                      rank="pressure" if role == "pressure" else "default",
                                      seeds=seeds),
            "rng": random.Random((spec.seed << 8) + i),
            "evals": 0,
        }

    evals_total = 0
    stop_reason = "generation budget exhausted"
    for gen in range(spec.generations):
        for name, isl in islands.items():
            ctx = {"rng": isl["rng"], "mold": mold,
                   "batch": spec.pop_size, "init_length": spec.init_length}
            cands = isl["prop"].propose(ctx)
            results = []
            for cand in cands:
                tidy, sc, _cost = judge.score(pack, mold, cand)
                results.append((tidy, sc))
            isl["prop"].feedback(results)
            isl["evals"] += len(results)
            evals_total += len(results)

        if gen % spec.migrate_every == 0:
            for name, isl in islands.items():
                best = isl["prop"].best()
                if best and best[1][0] == 1:          # correct only
                    accepted, why = archive.admit(best[0], PROV[isl["role"]],
                                                  name, gen)
                    if accepted:
                        rec.event("archive", "admit",
                                  payload={"island": name, "gen": gen,
                                           "cand": mold.pretty(mold.tidy(best[0])),
                                           "cost": mold.native_cost(mold.tidy(best[0])),
                                           "provenance": PROV[isl["role"]]},
                                  outcome="verified+stored")
            for name, isl in islands.items():
                entry = archive.sample(isl["rng"], PULL[isl["role"]])
                if entry:
                    tidy, sc, _ = judge.score(pack, mold, entry["cand"])
                    isl["prop"].absorb([(tidy, sc)])
            rec.event("foreman", "gen_summary",
                      payload={"gen": gen, "evals": evals_total,
                               "islands": {nm: {"best_score": list(isl["prop"].best()[1])
                                                if isl["prop"].best() else None}
                                           for nm, isl in islands.items()}})
            best_oo = archive.best_size("outcome-only")
            if (spec.target_outcome_size and best_oo
                    and best_oo["cost"]["comparators"] <= spec.target_outcome_size):
                stop_reason = (f"outcome-only entry reached target size "
                               f"{spec.target_outcome_size} at gen {gen}")
                break
    rec.event("foreman", "stop", reason=stop_reason)

    report = {"run_id": run_id, "spec": dataclasses.asdict(spec),
              "evals_total": evals_total,
              "seconds": round(time.time() - t0, 2),
              "stop_reason": stop_reason,
              "islands": {}}
    for name, isl in islands.items():
        b = isl["prop"].best()
        report["islands"][name] = {
            "evals": isl["evals"],
            "best": {"pretty": mold.pretty(mold.tidy(b[0])), "score": list(b[1]),
                     "cost": mold.native_cost(mold.tidy(b[0]))} if b else None}
    for prov in ("outcome-only", "seeded"):
        e = archive.best_size(prov)
        if e:
            verdict = recognizer.recognize(mold, e["cand"], shelf, bounds)
            rec.event("recognizer", "verdict",
                      payload={"provenance": prov, **verdict})
            report[f"best_{prov}"] = {
                "pretty": mold.pretty(e["cand"]), "cost": e["cost"],
                "island": e["island"], "gen": e["gen"],
                "certificate": {"level": "L1-exhaustive-in-bounds",
                                "evidence": e["verify_details"]},
                "recognition": verdict}
    n_saved = archive.save(rec.run_dir / "archive.json")
    report["archive_entries"] = n_saved
    rec.event("foreman", "report", payload=report)
    (rec.run_dir / "report.json").write_text(json.dumps(report, indent=2))
    rec.close()
    return report
