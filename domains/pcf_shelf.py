"""PCF reference shelf: battery constants + published generator families.

Families (citations):
- Table-7 family and Catalan kappa family of arXiv:2210.15669 (Ramanujan
  Machine project), parameterizations as implemented by the parent
  prototype (reference/mathlab/src/foundry.py:436-471):
    table7(i,j,mu):  a(n) = j(2i-j+2) + (4i+3) n + 3 n^2
                     b(n) = -2 n (n+j-1) (n+2i-j+1) (n+mu)
    kappa(k,c):      a(n) = (2k+1) + (3+4k) n + 3 n^2
                     b(n) = -2 n^2 (n+2k) (n+c)
- rm-8-7zeta3: a(n) = (2n+1)(3n^2+3n+1), b(n) = -n^6 — the Ramanujan-
  Machine conjecture 8/(7 zeta3), rediscovered from outcome by the parent
  (its commit c104290), used here as the zeta3-class control.

Every shelf member is verified through OUR pipeline at build time (Mobius
match at screen precision, residual re-check at 250 digits with scaled term
count). Members that fail to verify are excluded and recorded as such —
nothing enters the shelf on citation alone. Algebraic constants (sqrt2,
phi) are deliberately NOT in the battery (algebraic trap).
"""

import json
from pathlib import Path

import mpmath as mp

CACHE = Path(__file__).with_name("pcf_refs.json")


def battery(dps):
    with mp.workdps(dps):
        return {
            "catalan": +mp.catalan, "zeta3": mp.zeta(3), "pi^2": mp.pi ** 2,
            "pi": +mp.pi, "e": +mp.e, "log2": mp.log(2),
            "zeta5": mp.zeta(5), "euler-gamma": +mp.euler,
        }


def _pmul(p, q):
    out = [0] * (len(p) + len(q) - 1)
    for i, a in enumerate(p):
        for j, b in enumerate(q):
            out[i + j] += a * b
    return out


def _poly_prod(factors, scale):
    acc = [scale]
    for f in factors:
        acc = _pmul(acc, f)
    return tuple(acc)


def table7(i, j, mu):
    a = (j * (2 * i - j + 2), 4 * i + 3, 3)
    b = _poly_prod([[0, 1], [j - 1, 1], [2 * i - j + 1, 1], [mu, 1]], -2)
    return (a, b)


def kappa(k, c):
    a = (2 * k + 1, 3 + 4 * k, 3)
    b = _poly_prod([[0, 0, 1], [2 * k, 1], [c, 1]], -2)
    return (a, b)


# factored-mold forms of the same members (for walks: moves step along
# the family's own parameters)
def table7_f(i, j, mu):
    return ((j * (2 * i - j + 2), 4 * i + 3, 3),
            (-2, 1, tuple(sorted((j - 1, 2 * i - j + 1, mu)))))


def kappa_f(k, c):
    return ((2 * k + 1, 3 + 4 * k, 3), (-2, 2, tuple(sorted((2 * k, c)))))


def rm_f():
    return ((1, 5, 9, 6), (-1, 6, ()))


def factored_by_id():
    out = {"rm-8-7zeta3": rm_f(),
           "apery-zeta3": ((5, 27, 51, 34), (-1, 6, ()))}
    for k in range(0, 3):
        for c in range(0, 3):
            out[f"kappa(k={k},c={c})"] = kappa_f(k, c)
    for i in range(1, 4):
        for j in range(0, i // 2 + 2):
            for mu in range(0, 3):
                out[f"table7(i={i},j={j},mu={mu})"] = table7_f(i, j, mu)
    return out


def candidate_refs():
    """(id, family, params, cand) for every member we attempt to verify."""
    out = [("rm-8-7zeta3", "rm", {},
            ((1, 5, 9, 6), (0, 0, 0, 0, 0, 0, -1))),
           # Apery's zeta(3) fraction (value 6/zeta3): a = 34n^3+51n^2+27n+5,
           # b = -n^6. Apery 1979 ("Irrationalite de zeta(2) et zeta(3)");
           # CF form per van der Poorten 1979 ("A proof that Euler missed").
           ("apery-zeta3", "apery-1979", {},
            ((5, 27, 51, 34), (0, 0, 0, 0, 0, 0, -1)))]
    for k in range(0, 3):
        for c in range(0, 3):
            out.append((f"kappa(k={k},c={c})", "kappa-2210.15669",
                        {"k": k, "c": c}, kappa(k, c)))
    for i in range(1, 4):
        for j in range(0, i // 2 + 2):
            for mu in range(0, 3):
                out.append((f"table7(i={i},j={j},mu={mu})",
                            "table7-2210.15669",
                            {"i": i, "j": j, "mu": mu}, table7(i, j, mu)))
    return out


def build_refs(pack, force=False):
    """Verify every candidate reference through the pack's full pipeline.
    Returns (verified_refs, rejects). Cached to pcf_refs.json."""
    if CACHE.exists() and not force:
        data = json.loads(CACHE.read_text())
        refs = [{**r, "cand": (tuple(r["cand"][0]), tuple(r["cand"][1]))}
                for r in data["refs"]]
        return refs, data["rejects"]
    refs, rejects = [], []
    for rid, family, params, cand in candidate_refs():
        rec = pack.verify(cand)
        if rec["status"] == "verified":
            refs.append({"id": rid, "family": family, "params": params,
                         "cand": cand, "constant": rec["constant"],
                         "rel": rec["rel"], "delta": rec["delta"],
                         "value60": rec["value60"]})
        else:
            rejects.append({"id": rid, "reason": rec["drop_reason"]})
    CACHE.write_text(json.dumps(
        {"refs": [{**r, "cand": [list(r["cand"][0]), list(r["cand"][1])]}
                  for r in refs],
         "rejects": rejects}, indent=1))
    return refs, rejects


if __name__ == "__main__":
    assert kappa(0, 0) == ((1, 3, 3), (0, 0, 0, 0, -2))
    a, b = table7(1, 1, 0)
    assert a == (3, 7, 3) and b[0] == 0, (a, b)
    names = [r[0] for r in candidate_refs()]
    assert len(names) == len(set(names))
    print(f"pcf shelf ok: {len(names)} candidate refs "
          f"(verification happens in pack build, see domains/pcf.py)")
