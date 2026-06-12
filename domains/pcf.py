"""Domain pack: polynomial continued fractions (calibration B — replicate
the parent's strongest results through the generic engine).

Problem: find PCFs x = a(0) + K b(n)/a(n) whose value is a Mobius transform
of a battery constant. A PCF that equals a constant IS an algorithm for
computing it; quality = delta (irrationality-style convergence quality,
higher better) and description length (smaller polynomials, lower better).
Cost rules predeclared here.

Verification is conjecture-grade numerics, NOT proof — certificate wording
is "numeric-250-digit match (conjecture-grade, Ramanujan-Machine style)".
Discipline ported from the parent (its v2 control failure was a precision
bug): two stages, scaled term counts, and the three traps:
  rational trap -> drop; multi-constant (>=3) match -> drop;
  high-precision residual re-check -> drop on failure.

Controls: known members injected into every batch; ANY control failure
voids the batch (C- => uninterpretable, per the parent's PLAYBOOK lesson).
"""

import mpmath as mp

from engine import numeric


class PCFPack:
    name = "pcf"

    def __init__(self, screen_dps=60, verify_dps=250):
        self.screen_dps = screen_dps
        self.verify_dps = verify_dps
        from domains import pcf_shelf
        self._shelf_mod = pcf_shelf
        self._battery60 = pcf_shelf.battery(screen_dps)
        # order matters for speed: most-likely constants first
        self._order = ["catalan", "zeta3", "pi^2", "pi", "e", "log2",
                       "zeta5", "euler-gamma"]
        self.cost_rules = ("primary: verified Mobius match to a battery "
                           "constant (250-digit, conjecture-grade); "
                           "quality: delta higher-better, description "
                           "length lower-better")
        self.refs = None      # set by load_refs()

    # --- verification pipeline (gate 1 + certificate) ---------------------
    def verify(self, cand):
        a, b = cand
        meters = {}
        r = numeric.eval_pcf(a, b, terms=numeric.verify_terms_for(self.screen_dps),
                             dps=self.screen_dps)
        meters["terms_screen"] = r.get("terms", 0)
        if r["degenerate"] or not r["converged"]:
            return {"status": "dropped", "drop_reason": "diverged-or-degenerate",
                    "meters": meters}
        v = r["value"]
        d = numeric.delta(r["err"], r["log10_q"])
        if numeric.is_rational(v, dps=self.screen_dps):
            return {"status": "dropped", "drop_reason": "rational",
                    "delta": d, "meters": meters}
        matches = []
        for name in self._order:
            rel = numeric.mobius_match(v, self._battery60[name],
                                       dps=self.screen_dps)
            if rel:
                matches.append((name, rel))
                if len(matches) >= 3:
                    return {"status": "dropped",
                            "drop_reason": "multi-constant-trivial",
                            "delta": d, "meters": meters}
        if not matches:
            return {"status": "dropped", "drop_reason": "no-match",
                    "delta": d, "meters": meters}
        name, rel = matches[0]
        vt = numeric.verify_terms_for(self.verify_dps)
        r2 = numeric.eval_pcf(a, b, terms=vt, dps=self.verify_dps)
        meters["terms_verify"] = r2.get("terms", 0)
        if r2["degenerate"]:
            return {"status": "dropped", "drop_reason": "degenerate-at-verify",
                    "delta": d, "meters": meters}
        with mp.workdps(self.verify_dps):
            const_hi = self._shelf_mod.battery(self.verify_dps)[name]
            resid = numeric.residual(rel, r2["value"], const_hi,
                                     self.verify_dps)
            bound = mp.mpf(10) ** (-(self.verify_dps - 12))
            if resid >= bound:
                return {"status": "dropped",
                        "drop_reason": "failed-high-precision",
                        "constant": name, "rel": list(rel), "delta": d,
                        "residual_log10": float(mp.log10(resid)),
                        "meters": meters}
        return {"status": "verified", "constant": name, "rel": list(rel),
                "delta": d, "value60": mp.nstr(v, 50),
                "residual_log10": float(mp.log10(resid)) if resid > 0 else None,
                "certificate": {"level": "numeric-250-digit "
                                         "(conjecture-grade, RM-style)",
                                "evidence": f"Mobius relation {list(rel)} to "
                                            f"{name}; residual < 1e-{self.verify_dps - 12} "
                                            f"at {self.verify_dps} dps with {vt} terms"},
                "meters": meters}

    # --- recognition: reference subtraction (gate 3) ----------------------
    def load_refs(self, force=False):
        self.refs, rejects = self._shelf_mod.build_refs(self, force=force)
        return self.refs, rejects

    def reference_subtract(self, value60_str, constant):
        """KNOWN if the value is Mobius-equivalent to a same-constant
        reference value: pslq([1, rv, v, v*rv]) with small coefficients."""
        assert self.refs is not None, "call load_refs() first"
        with mp.workdps(self.screen_dps):
            v = mp.mpf(value60_str)
            for ref in self.refs:
                if ref["constant"] != constant:
                    continue
                rv = mp.mpf(ref["value60"])
                rel = mp.pslq([mp.mpf(1), rv, v, v * rv],
                              maxcoeff=10**4, maxsteps=6000)
                if rel and not (rel[2] == 0 and rel[3] == 0):
                    return ref["id"], tuple(int(x) for x in rel)
        return None, None

    def classify(self, mold, cand):
        """Full record: verify, then reference-subtract on success.
        `cand` is a MOLD candidate (factored); the numeric engine gets the
        dense expansion."""
        rec = self.verify(mold.dense(mold.tidy(cand)))
        if rec["status"] != "verified":
            return {"label": "DROPPED", **rec}
        ref_id, ref_rel = self.reference_subtract(rec["value60"],
                                                  rec["constant"])
        if ref_id:
            return {"label": "KNOWN", "ref": ref_id,
                    "ref_rel": list(ref_rel), **rec}
        return {"label": "UNRESOLVED-novel-flag", **rec}

    # --- judge contract (engine/registry.py) ----------------------------
    def gate1(self, mold, tidy):
        """Screen-grade score for generic drivers: full pipeline verdict
        folded into (verified, delta, -dl). Expensive relative to sorting's
        gate (pslq-bound) — drivers should cache by tidy form."""
        rec = self.verify(mold.dense(tidy))
        d = rec.get("delta")
        cost = mold.native_cost(tidy)
        return ((1 if rec["status"] == "verified" else 0,
                 d if d is not None else -5.0, -cost["dl"]), cost)

    def verify_trusted(self, mold, cand):
        rec = self.verify(mold.dense(mold.tidy(cand)))
        return rec["status"] == "verified", rec

    def controls(self):
        """Known members injected into every batch; failure voids the batch.
        Chosen to span constant classes."""
        ids = ["rm-8-7zeta3", "kappa(k=0,c=0)", "kappa(k=1,c=0)"]
        assert self.refs is not None, "call load_refs() first"
        by_id = {r["id"]: r for r in self.refs}
        return [(i, by_id[i]["cand"], by_id[i]["constant"])
                for i in ids if i in by_id]


if __name__ == "__main__":
    pack = PCFPack()
    rec = pack.verify(((1, 5, 9, 6), (0, 0, 0, 0, 0, 0, -1)))
    assert rec["status"] == "verified" and rec["constant"] == "zeta3", rec
    print("pcf pack ok: RM 8/(7 zeta3) verified end to end —",
          rec["certificate"]["level"], "| delta =",
          round(rec["delta"], 3) if rec["delta"] else None)
