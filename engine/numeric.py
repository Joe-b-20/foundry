"""Numeric engine for identity-style domains (PCFs). mpmath is the rented
oracle; this wrapper is the owned, metered surface — nothing in a domain
pack calls mpmath directly.

Conventions and traps are ported from the parent prototype
(reference/mathlab/src/foundry.py, gpu_pcf_hunt.py) — see TRACKER for the
porting brief. The hard-won ones:
- term count must SCALE with precision: vterms = max(1400, dps*9). PCFs
  converge ~0.15 digits/term; the parent's "Foundry v2" control failure was
  exactly a fixed term count at high dps.
- rational trap: a rational value Mobius-matches every constant; test
  pslq([v, 1]) first and drop.
- multi-constant trap: a value matching >= 3 distinct constants is residual
  triviality, not a discovery.
"""

import mpmath as mp


def eval_pcf(a_coeffs, b_coeffs, terms, dps):
    """Evaluate x = a(0) + K_{n>=1} b(n)/a(n) by the standard recurrence
        p_n = a(n) p_{n-1} + b(n) p_{n-2},   q_n likewise,
        p_{-1}=1, q_{-1}=0, p_0=a(0), q_0=1.
    Returns dict: value, err (last tail diff), log10_q, converged, terms,
    dps, degenerate. Cost meter = terms actually evaluated at this dps.
    """
    def poly(cs, n):
        acc = 0
        for c in reversed(cs):
            acc = acc * n + c
        return acc

    with mp.workdps(dps):
        p_prev2, q_prev2 = mp.mpf(1), mp.mpf(0)
        p_prev, q_prev = mp.mpf(poly(a_coeffs, 0)), mp.mpf(1)
        v_prev = None
        err = mp.inf
        for n in range(1, terms + 1):
            an, bn = poly(a_coeffs, n), poly(b_coeffs, n)
            p = an * p_prev + bn * p_prev2
            q = an * q_prev + bn * q_prev2
            p_prev2, q_prev2, p_prev, q_prev = p_prev, q_prev, p, q
            if q_prev == 0:
                return {"degenerate": True, "converged": False,
                        "terms": n, "dps": dps}
            if n % 8 == 0 or n == terms:
                v = p_prev / q_prev
                if v_prev is not None:
                    err = abs(v - v_prev)
                v_prev = v
        v = p_prev / q_prev
        if not mp.isfinite(v):
            return {"degenerate": True, "converged": False,
                    "terms": terms, "dps": dps}
        log10_q = float(mp.log10(abs(q_prev))) if q_prev != 0 else 0.0
        converged = bool(mp.isfinite(err) and err < mp.mpf(10) ** (-(dps // 3)))
        return {"value": v, "err": err, "log10_q": log10_q,
                "converged": converged, "terms": terms, "dps": dps,
                "degenerate": False}


def delta(err, log10_q):
    """Irrationality-quality score (Ramanujan-Machine style):
    delta = -1 - log(err)/log(q). Higher is better; >0.3 notable."""
    if err == 0 or log10_q <= 0:
        return None
    try:
        return float(-1 - mp.log10(err) / log10_q)
    except (ValueError, OverflowError):
        return None


def is_rational(v, dps=60, maxcoeff=10**7, maxsteps=10**4):
    """Rational trap. True if pslq finds p*v + q*1 = 0 with p != 0.
    Exact zero is rational (and pslq refuses zero vectors)."""
    if v == 0:
        return True
    with mp.workdps(dps):
        rel = mp.pslq([v, mp.mpf(1)], maxcoeff=maxcoeff, maxsteps=maxsteps)
    return rel is not None and rel[0] != 0


def mobius_match(v, const, dps=60, maxcoeff=10**5, maxsteps=8000):
    """Find integers (r0,r1,r2,r3), r0 + r1*C + r2*v + r3*v*C = 0,
    i.e. v = -(r0 + r1*C) / (r2 + r3*C). None if no relation."""
    with mp.workdps(dps):
        rel = mp.pslq([mp.mpf(1), const, v, v * const],
                      maxcoeff=maxcoeff, maxsteps=maxsteps)
    if rel is None or (rel[2] == 0 and rel[3] == 0):
        return None
    return tuple(int(r) for r in rel)


def residual(rel, v, const, dps):
    """|r0 + r1*C + r2*v + r3*v*C| at the CURRENT precision."""
    with mp.workdps(dps):
        r = (rel[0] + rel[1] * const + rel[2] * v + rel[3] * v * const)
        return abs(r)


def verify_terms_for(dps):
    """The v2 lesson: scale evaluation depth with target precision."""
    return max(1400, int(dps * 9))


if __name__ == "__main__":
    # control: the parent's flagship rediscovery, RM 8/(7*zeta3):
    # a(n) = (2n+1)(3n^2+3n+1) -> coeffs (1,5,9,6); b(n) = -n^6
    a = (1, 5, 9, 6)
    b = (0, 0, 0, 0, 0, 0, -1)
    r = eval_pcf(a, b, terms=600, dps=60)
    assert r["converged"] and not r["degenerate"], r
    with mp.workdps(60):
        target = 8 / (7 * mp.zeta(3))
        assert abs(r["value"] - target) < mp.mpf(10) ** -40, \
            (r["value"], target)
        rel = mobius_match(r["value"], mp.zeta(3))
    assert rel is not None, "Mobius match failed on the control"
    assert not is_rational(r["value"])
    d = delta(r["err"], r["log10_q"])
    print("numeric ok: RM 8/(7 zeta3) control verified at 60 dps;",
          "rel =", rel, "delta =", round(d, 3) if d else None)
