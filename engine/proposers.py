"""Proposers v0: strategies only — the edit moves come from the mold.

Contract:
    propose(ctx) -> list of candidates
    feedback(results)            # results = [(candidate, score_tuple), ...]
ctx is a dict: {"rng", "mold", "batch", "init_length"}.
Scores are tuples ordered so that bigger is better (python max works).

Proposers see only outcomes — scores and (later) counterexamples. They never
see checker internals or shelf solutions.
"""


class RandomProposer:
    name = "random"

    def propose(self, ctx):
        mold, rng = ctx["mold"], ctx["rng"]
        return [mold.random_candidate(rng, ctx["init_length"])
                for _ in range(ctx["batch"])]

    def feedback(self, results):
        pass


class HillClimber:
    """Keep one current candidate; propose a batch of mutants; adopt the best
    if it improves; restart from random after `patience` stale rounds."""

    name = "hill-climb"

    def __init__(self, patience=60):
        self.current = None
        self.score = None
        self.stale = 0
        self.patience = patience
        self.restarts = 0

    def propose(self, ctx):
        mold, rng = ctx["mold"], ctx["rng"]
        if self.current is None:
            return [mold.random_candidate(rng, ctx["init_length"])
                    for _ in range(ctx["batch"])]
        return [mold.mutate(self.current, rng) for _ in range(ctx["batch"])]

    def feedback(self, results):
        best_cand, best_score = max(results, key=lambda r: r[1])
        if self.score is None or best_score > self.score:
            self.current, self.score, self.stale = best_cand, best_score, 0
        else:
            self.stale += 1
            if self.stale >= self.patience:
                self.current, self.score, self.stale = None, None, 0
                self.restarts += 1


class EvolutionProposer:
    """Population GA. Strategy only — moves come from the mold.

    rank="default": climb correctness first, then shrink (correct, sorted,
    -size, -depth). rank="pressure": Island-C mode — among candidates,
    smallness outranks sortedness ((correct, -size, -depth, sorted)), so the
    population may pass through WRONG intermediates on the way to smaller
    correct forms. Wrong candidates never leave the island: only the archive
    bridges islands, and the archive verifies on write.
    """

    name = "evolution"

    def __init__(self, pop_size=64, mut_stack=3, crossover=0.2,
                 max_len_factor=3, rank="default", seeds=None, key=None):
        self.pop_size = pop_size
        self.mut_stack = mut_stack
        self.crossover = crossover
        self.max_len_factor = max_len_factor
        self.rank = rank
        self.key = key                # optional custom rank callable
        self.seeds = list(seeds or [])
        self.population = []          # list of (cand, score_tuple)

    def _key(self, sc):
        if self.key is not None:
            return self.key(sc)
        correct, n_sorted, neg_size, neg_depth = sc
        if self.rank == "pressure":
            return (correct, neg_size, neg_depth, n_sorted)
        return sc

    def _parent(self, rng):
        a = rng.choice(self.population)
        b = rng.choice(self.population)
        return a[0] if self._key(a[1]) >= self._key(b[1]) else b[0]

    def propose(self, ctx):
        mold, rng = ctx["mold"], ctx["rng"]
        if not self.population:
            fresh = [mold.random_candidate(rng, ctx["init_length"])
                     for _ in range(max(0, self.pop_size - len(self.seeds)))]
            return list(self.seeds) + fresh
        cap = self.max_len_factor * ctx["init_length"]
        out = []
        for _ in range(ctx["batch"]):
            child = self._parent(rng)
            if self.crossover and rng.random() < self.crossover:
                other = self._parent(rng)
                if child and other:
                    child = (child[: rng.randrange(1, len(child) + 1)]
                             + other[rng.randrange(len(other)):])
            for _ in range(rng.randrange(1, self.mut_stack + 1)):
                child = mold.mutate(child, rng)
            if len(child) > cap:
                child = child[:cap]
            out.append(child)
        return out

    def absorb(self, scored):
        """Inject already-scored candidates (e.g. archive migrants)."""
        self.population.extend(scored)
        self._truncate()

    def feedback(self, results):
        self.population.extend(results)
        self._truncate()

    def _truncate(self):
        seen = {}
        for cand, sc in self.population:
            if cand not in seen or sc > seen[cand]:
                seen[cand] = sc
        ranked = sorted(seen.items(), key=lambda kv: self._key(kv[1]),
                        reverse=True)
        self.population = ranked[: self.pop_size]

    def best(self):
        return max(self.population, key=lambda r: r[1]) if self.population else None


PROPOSERS = {"random": RandomProposer, "hill-climb": HillClimber,
             "evolution": EvolutionProposer}


if __name__ == "__main__":
    import random
    from engine.molds import ComparatorMold
    mold = ComparatorMold(3)
    ctx = {"rng": random.Random(0), "mold": mold, "batch": 8, "init_length": 4}
    hc = HillClimber(patience=2)
    batch = hc.propose(ctx)
    assert len(batch) == 8
    # fake scores: pretend the first is best
    hc.feedback([(c, (0, i == 0, 0, 0)) for i, c in enumerate(batch)])
    assert hc.current == batch[0]
    # two stale rounds -> restart
    hc.feedback([(c, (0, 0, 0, 0)) for c in batch])
    hc.feedback([(c, (0, 0, 0, 0)) for c in batch])
    assert hc.current is None and hc.restarts == 1
    print("proposers v0 ok")
