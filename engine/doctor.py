"""Wall doctor v1: differential diagnosis for stuck searches.

Watches the best-so-far trajectory and compares the plateau level against
the domain's "chance-plus" baselines (the best one-op functions). Verdicts
always use the scoped format agreed with Joe — never absolute claims:

    "no usable signal under the current representation / search
     primitives / budget"

v1 RECOMMENDS and never kills: the operator (a human, or a predeclared
logged policy) decides. The C1/C2 exam (scripts/run_doctor_exam.py) tests
both failure directions: wrong-quit on a findable target and wrong-grind
on a hopeless one. The wall taxonomy behind the smells is the parent
project's (reference/mathlab, consolidation docs).
"""

SCOPED = ("no usable signal under the current representation / "
          "search primitives / budget")


class WallDoctor:
    def __init__(self, min_gens=800, plateau_window=500,
                 improve_eps=0.002, chance_margin=0.03):
        self.min_gens = min_gens
        self.plateau_window = plateau_window
        self.improve_eps = improve_eps
        self.chance_margin = chance_margin
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
            return {"verdict": SCOPED,
                    "wall_smell": "learnability/pseudorandom — plateaued "
                                  "at one-op chance-plus level",
                    "evidence": evidence,
                    "recommendation": "abandon-target"}
        return {"verdict": SCOPED,
                "wall_smell": "partial signal, plateaued above chance — "
                              "representation or budget limit",
                "evidence": evidence,
                "recommendation": "switch-representation-or-raise-budget"}


if __name__ == "__main__":
    bl = {"x^y": 0.55, "x": 0.52}
    d = WallDoctor(min_gens=100, plateau_window=50, chance_margin=0.03)
    for g in range(0, 200, 10):
        d.observe(g, 0.56)                  # flat at chance-plus
    v = d.diagnose(bl)
    assert v and v["recommendation"] == "abandon-target", v
    d2 = WallDoctor(min_gens=100, plateau_window=50)
    for g in range(0, 200, 10):
        d2.observe(g, 0.5 + g * 0.002)      # still climbing
    assert d2.diagnose(bl) is None
    d3 = WallDoctor(min_gens=100, plateau_window=50)
    for g in range(0, 200, 10):
        d3.observe(g, 0.80)                 # plateaued but well above chance
    v3 = d3.diagnose(bl)
    assert v3 and v3["recommendation"].startswith("switch"), v3
    print("doctor v1 ok: abandon / quiet / switch verdicts behave")
