"""Archive v0: the gene pool, and the ONLY bridge between islands.

Rules baked in (RULES.md):
- verify-on-write: nothing is stored before an independent re-execution
  through the core runner (judge.verify_canonical). For sorting networks the
  corpus check is already exhaustive (0/1 principle), so the "light breaker
  pass" admission rule is satisfied by construction in this domain; the
  standalone breaker arrives with domains whose corpora are not exhaustive.
- provenance is first-class: "outcome-only" lineages (random starts, pulls
  restricted to outcome-only entries) stay separable from "seeded" lineages
  (started from the reference shelf). Discovered-from-outcome must never
  blur with scaffolded. v0 tracks provenance at island granularity; the
  per-candidate lineage graph is future work.
- admission: correct + verified + not pareto-dominated by an existing entry
  on (size, depth).
"""

import json
import time
from pathlib import Path

from engine import judge


class Archive:
    def __init__(self, domain, pack, mold):
        self.domain = domain
        self.pack = pack
        self.mold = mold
        # keyed by (provenance, canonical): domination and identity are
        # scoped WITHIN a provenance class. A seeded entry must never block
        # an outcome-only entry from being recorded — otherwise the seeded
        # lineage silently erases the discovered-from-outcome record (bug
        # found in the first n=8 islands run, 2026-06-12: Batcher 19/6 at
        # gen 0 pareto-blocked every outcome-only admission).
        self.entries = {}

    def _dominated(self, cost, provenance):
        """Dominated (or equalled) by an existing SAME-PROVENANCE entry?"""
        for (prov, _), e in self.entries.items():
            if prov != provenance:
                continue
            c = e["cost"]
            if (c["comparators"] <= cost["comparators"]
                    and c["depth"] <= cost["depth"]):
                return True
        return False

    def admit(self, cand, provenance, island, gen):
        """Returns (accepted: bool, reason: str). Verifies before storing."""
        tidy = self.mold.tidy(cand)
        if (provenance, tidy) in self.entries:
            return False, "duplicate canonical form (same provenance)"
        cost = self.mold.native_cost(tidy)
        if self._dominated(cost, provenance):
            return False, "pareto-dominated within provenance class"
        ok, details = judge.verify_canonical(self.pack, self.mold, tidy)
        if not ok:
            return False, f"verify-on-write FAILED: {details}"
        self.entries[(provenance, tidy)] = {
            "cand": tidy, "cost": cost, "provenance": provenance,
            "island": island, "gen": gen, "ts": round(time.time(), 3),
            "verified": True, "verify_details": details,
        }
        return True, "admitted"

    def elites(self, prov_filter=None):
        """Pareto front on (size, depth); optionally provenance-filtered."""
        pool = [e for e in self.entries.values()
                if prov_filter is None or e["provenance"] == prov_filter]
        front = []
        for e in pool:
            c = e["cost"]
            if not any(o is not e
                       and o["cost"]["comparators"] <= c["comparators"]
                       and o["cost"]["depth"] <= c["depth"]
                       and (o["cost"]["comparators"] < c["comparators"]
                            or o["cost"]["depth"] < c["depth"])
                       for o in pool):
                front.append(e)
        return front

    def sample(self, rng, prov_filter=None):
        front = self.elites(prov_filter)
        return rng.choice(front) if front else None

    def best_size(self, prov_filter=None):
        front = self.elites(prov_filter)
        if not front:
            return None
        return min(front, key=lambda e: (e["cost"]["comparators"],
                                         e["cost"]["depth"]))

    def save(self, path):
        out = [{**e, "cand": [list(p) for p in e["cand"]]}
               for e in self.entries.values()]
        Path(path).write_text(json.dumps(out, indent=1))
        return len(out)


if __name__ == "__main__":
    import random
    from domains.sorting_networks import SortingNetworkPack
    from domains.sorting_networks_shelf import batcher_network
    from engine.molds import ComparatorMold
    pack, mold = SortingNetworkPack(4), ComparatorMold(4)
    ar = Archive("sorting_networks", pack, mold)
    ok, why = ar.admit(batcher_network(4), "seeded", "B", 0)
    assert ok, why
    ok2, why2 = ar.admit(batcher_network(4), "seeded", "B", 1)
    assert not ok2 and "duplicate" in why2
    bad = ((0, 1), (2, 3))   # not a sorter: verify-on-write must reject
    ok3, why3 = ar.admit(bad, "outcome-only", "A", 2)
    assert not ok3 and "FAILED" in why3, why3
    assert ar.best_size()["cost"]["comparators"] == 5
    assert ar.sample(random.Random(0), "outcome-only") is None
    print("archive v0 ok:", len(ar.entries), "entry;", why2, "/", why3[:30])
