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


PROPOSERS = {"random": RandomProposer, "hill-climb": HillClimber}


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
