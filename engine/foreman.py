"""Foreman v0: takes a RunSpec, drives the loop, enforces budgets, logs
everything, and verifies the winner through the core runner before reporting
(verify-on-write). The foreman executes decisions; it never makes them.

The RunSpec is a plain data object on purpose — "a trained model operates
the foundry" will literally mean "a model emits RunSpecs and decisions".
"""

import dataclasses
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

from engine import judge, recognizer
from engine.molds import ComparatorMold
from engine.proposers import PROPOSERS
from engine.recorder import Recorder
from domains.sorting_networks import SortingNetworkPack
from domains import sorting_networks_shelf as shelf_mod


@dataclass
class RunSpec:
    domain: str
    domain_params: dict
    proposer: str
    proposer_params: dict = field(default_factory=dict)
    budget_candidates: int = 20_000
    batch: int = 32
    init_length: int = 12
    seed: int = 0
    # once a correct candidate exists, stop after this many batches without
    # any improvement (lets the climb keep shrinking size before stopping)
    settle_batches: int = 150
    cost_rules: str = "primary: comparators (size); secondary: depth"
    log_every_candidate: bool = True
    schema: str = "runspec-v0"


def build(spec: RunSpec):
    assert spec.domain == "sorting_networks", "v0 knows exactly one domain"
    pack = SortingNetworkPack(**spec.domain_params)
    mold = ComparatorMold(pack.n)
    proposer = PROPOSERS[spec.proposer](**spec.proposer_params)
    return pack, mold, proposer


def run(spec: RunSpec, runs_root="runs"):
    t0 = time.time()
    run_id = (f"{spec.domain}-n{spec.domain_params.get('n')}"
              f"-{spec.proposer}-s{spec.seed}-{int(t0)}")
    rec = Recorder(Path(runs_root) / run_id, run_id)
    rec.event("foreman", "runspec", payload=dataclasses.asdict(spec),
              reason="predeclaration: budgets and cost rules fixed before search")
    pack, mold, proposer = build(spec)
    rng = random.Random(spec.seed)
    ctx = {"rng": rng, "mold": mold,
           "batch": spec.batch, "init_length": spec.init_length}

    best = None          # (score_tuple, tidied_candidate, native_cost)
    used = 0
    batches_since_improve = 0
    stop_reason = "candidate budget exhausted"
    while used < spec.budget_candidates:
        cands = proposer.propose(ctx)
        results = []
        for cand in cands:
            tidy, sc, cost = judge.score(pack, mold, cand)
            results.append((tidy, sc))
            used += 1
            if spec.log_every_candidate:
                rec.event("judge", "evaluate",
                          payload={"cand": mold.pretty(tidy), "score": list(sc)})
            if best is None or sc > best[0]:
                best = (sc, tidy, cost)
                batches_since_improve = -1   # reset; incremented below
                rec.event("judge", "improvement",
                          payload={"cand": mold.pretty(tidy), "score": list(sc),
                                   "cost": cost, "candidates_used": used})
        proposer.feedback(results)
        batches_since_improve += 1
        if (best and best[0][0] == 1
                and batches_since_improve >= spec.settle_batches):
            stop_reason = (f"correct candidate held through {spec.settle_batches}"
                           " batches without improvement")
            break
    rec.event("foreman", "stop", reason=stop_reason)

    report = {"run_id": run_id, "spec": dataclasses.asdict(spec),
              "candidates_used": used,
              "seconds": round(time.time() - t0, 2),
              "stop_reason": stop_reason,
              "found_correct": bool(best and best[0][0] == 1)}
    if best:
        sc, tidy, cost = best
        report["best"] = {"pretty": mold.pretty(tidy),
                          "cand": [list(p) for p in tidy],
                          "score": list(sc), "cost": cost}
        if sc[0] == 1:
            ok, details = judge.verify_canonical(pack, mold, tidy)
            rec.event("judge", "verify_on_write", payload=details,
                      outcome="pass" if ok else "FAIL",
                      reason="fast checker never gets the final word")
            report["verified_canonical"] = ok
            report["verify_details"] = details
            if ok:
                report["certificate"] = {
                    "level": "L1-exhaustive-in-bounds",
                    "claim": f"sorts every length-{pack.n} input",
                    "evidence": (f"all {pack.total} binary vectors pass via the"
                                 " core runner; 0/1 principle extends this to"
                                 " all inputs (see pack docstring); plus"
                                 f" {len(pack._extra)} random integer vectors"),
                }
                shelf = shelf_mod.build_shelf(pack, mold)
                verdict = recognizer.recognize(
                    mold, tidy, shelf, shelf_mod.BOUNDS.get(pack.n))
                rec.event("recognizer", "verdict", payload=verdict,
                          reason="gate 3: new, known, or variant?")
                report["recognition"] = verdict
    rec.event("foreman", "report", payload=report)
    (rec.run_dir / "report.json").write_text(json.dumps(report, indent=2))
    rec.close()
    return report
