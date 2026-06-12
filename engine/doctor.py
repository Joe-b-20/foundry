"""Wall doctor v1: differential diagnosis for stuck searches.

Watches the best-so-far trajectory and compares the plateau level against
the domain's "chance-plus" baselines (the best one-op functions). Verdicts
always use the scoped format agreed with Joe — never absolute claims:

    "no usable signal under the current representation / search
     primitives / budget"

v1 RECOMMENDS and never kills: the operator (a human, or a predeclared
logged policy) decides. The exam (scripts/run_doctor_exam.py) tests three
directions: wrong-quit on a findable target (C1), wrong-grind on a hopeless
one (C2), and — added after Joe's grokking question, 2026-06-12 — NOT
killing a deceptive search that plateaus ABOVE chance then suddenly
reorganizes (C3). The wall taxonomy behind the smells is the parent
project's (reference/mathlab, consolidation docs).

GROKKING NOTE (measured, scripts/run_grokking_probe.py): the abandon
verdict fires only when the plateau sits at CHANCE on held-out data, not
merely when best-so-far is flat. A grokking-bound search assembling
building blocks carries above-chance partial signal during its plateau, so
it is routed to 'switch/raise budget', never 'abandon' — verified on a real
gen-2303 grok. The residual hard case (a search flat exactly AT chance then
sudden) is information-theoretically near-identical to the cryptographic
wall; no outcome-only detector separates them. Defenses: recommends-only,
confidence grading + asymmetric patience below, and the open 'motion
underneath' signal (population novelty / description-length still moving)
to be built when an at-chance grok can be manufactured to test it.
"""

SCOPED = ("no usable signal under the current representation / "
          "search primitives / budget")


class WallDoctor:
    def __init__(self, min_gens=800, plateau_window=500,
                 improve_eps=0.002, chance_margin=0.03,
                 abandon_min_gens=None):
        self.min_gens = min_gens
        self.plateau_window = plateau_window
        self.improve_eps = improve_eps
        self.chance_margin = chance_margin
        # asymmetric patience: a HIGH-confidence abandon needs the
        # at-chance plateau to persist this long. Killing a real discovery
        # costs far more than running a dead search a while longer, so the
        # bar to confidently declare a wall is deliberately higher than the
        # bar to start worrying.
        self.abandon_min_gens = (abandon_min_gens if abandon_min_gens
                                 is not None else min_gens + 2 * plateau_window)
        self.history = []          # (gen, best_frac_so_far, best_len)

    def observe(self, gen, best_frac, best_len=None):
        self.history.append((gen, best_frac, best_len))

    def diagnose(self, baselines, heldout_best=None):
        """None = no concern (keep going / solved). Otherwise a verdict
        dict with evidence and exactly one recommendation.

        heldout_best, when provided, is the best candidate's agreement on
        data the search never optimized — the overfit detector. The
        abandon decision is made on generalization, not on corpus fit
        (plateau + corpus >> heldout = memorization, not signal)."""
        if not self.history:
            return None
        gen, best, _len = self.history[-1]
        if gen < self.min_gens or best >= 1.0:
            return None
        window = [h for h in self.history
                  if h[0] >= gen - self.plateau_window]
        if not window or window[0][0] > gen - self.plateau_window + 50:
            return None                     # window not yet filled
        improvement = best - window[0][1]
        if improvement > self.improve_eps:
            return None                     # still climbing
        chance_plus = max(baselines.values())
        basis = heldout_best if heldout_best is not None else best
        evidence = {"gen": gen, "corpus_best": round(best, 4),
                    "heldout_best": (round(heldout_best, 4)
                                     if heldout_best is not None else None),
                    "window_gens": self.plateau_window,
                    "window_improvement": round(improvement, 5),
                    "chance_plus_baseline": round(chance_plus, 4),
                    "baselines": {k: round(v, 4)
                                  for k, v in baselines.items()}}
        if basis <= chance_plus + self.chance_margin:
            # confidence is asymmetric: only a long-sustained at-chance
            # plateau earns 'high'. A grok that sits at chance briefly then
            # jumps is still 'low' and survives an operator that requires
            # high confidence to accept an abandon.
            confidence = "high" if gen >= self.abandon_min_gens else "low"
            evidence["abandon_min_gens"] = self.abandon_min_gens
            return {"verdict": SCOPED,
                    "wall_smell": "learnability/pseudorandom — plateaued "
                                  "at one-op chance-plus level on held-out",
                    "evidence": evidence,
                    "confidence": confidence,
                    "recommendation": "abandon-target"}
        return {"verdict": SCOPED,
                "wall_smell": "partial signal, plateaued ABOVE chance — "
                              "representation or budget limit, NOT a wall "
                              "(grokking-compatible: keep going)",
                "evidence": evidence,
                "confidence": "n/a",
                "recommendation": "switch-representation-or-raise-budget"}


if __name__ == "__main__":
    bl = {"x^y": 0.55, "x": 0.52}
    # at chance, sustained past abandon_min_gens -> high-confidence abandon
    d = WallDoctor(min_gens=100, plateau_window=50, chance_margin=0.03,
                   abandon_min_gens=150)
    for g in range(0, 400, 10):
        d.observe(g, 0.56)
    v = d.diagnose(bl)
    assert v and v["recommendation"] == "abandon-target" \
        and v["confidence"] == "high", v
    # at chance but NOT yet sustained -> low confidence (grok still has rope)
    d_low = WallDoctor(min_gens=100, plateau_window=50, abandon_min_gens=5000)
    for g in range(0, 400, 10):
        d_low.observe(g, 0.56)
    vlow = d_low.diagnose(bl)
    assert vlow["recommendation"] == "abandon-target" \
        and vlow["confidence"] == "low", vlow
    # still climbing -> no concern
    d2 = WallDoctor(min_gens=100, plateau_window=50)
    for g in range(0, 200, 10):
        d2.observe(g, 0.5 + g * 0.002)
    assert d2.diagnose(bl) is None
    # plateaued ABOVE chance (the grokking case) -> switch, never abandon
    d3 = WallDoctor(min_gens=100, plateau_window=50)
    for g in range(0, 200, 10):
        d3.observe(g, 0.80)
    v3 = d3.diagnose(bl)
    assert v3 and v3["recommendation"].startswith("switch"), v3
    print("doctor v1 ok: high/low-confidence abandon, climbing, "
          "and above-chance-plateau (grok-safe) verdicts behave")
