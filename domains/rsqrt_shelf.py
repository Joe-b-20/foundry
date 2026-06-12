"""Reference shelf for fast inverse square root (float32).

Citations (web-checked 2026-06-12):
- quake-newton: magic 0x5F3759DF + one standard Newton step
  y*(1.5 - 0.5*x*y*y). Quake III Arena (1999; source public 2005);
  see the Wikipedia "Fast inverse square root" article for history.
- lomont-newton: magic 0x5F375A86 — C. Lomont, "Fast Inverse Square
  Root" (2003): optimal constant for the standard structure, error bound
  1.751302e-3 over normal floats.
- (noted, constants NOT carried from memory): Moroz, Walczyk, Cieslinski
  (arXiv:1802.06302; Computation 2019; Entropy 2021) modify the
  Newton-step coefficients for ~2x better accuracy at the same cost. If
  our search lands near their numbers, exact values get fetched and cited
  before any KNOWN label is applied.

Every entry is self-verified at build time: exhaustive max-rel-error over
the pack's scope through our own pipeline. The Lomont bound doubles as a
consistency check (our measured error on a sub-scope must not exceed the
published global bound by more than float noise).

No proven LOWER bound exists for arbitrary <= 9-op programs — the
published constants are optimal for THE standard structure only. That gap
is the headroom this domain is being hunted for.
"""

import struct


def _fbits(v):
    return struct.unpack("<I", struct.pack("<f", v))[0]


def trick_newton(magic, half=0.5, three_halves=1.5):
    """The standard structure: seed via magic constant, one Newton step."""
    return ((("SHR32", 5, 0, 2),      # t0 = x >> 1
             ("SUB32", 6, 1, 5),      # t1 = magic - t0     (seed y)
             ("FMUL", 7, 0, 6),       # t2 = x*y
             ("FMUL", 7, 7, 6),       # t2 = x*y*y
             ("FMUL", 7, 7, 3),       # t2 = half*x*y*y
             ("FSUB", 7, 4, 7),       # t2 = three_halves - t2
             ("FMUL", 0, 6, 7)),      # out = y * t2
            (magic, 1, _fbits(half), _fbits(three_halves)))


def trick_seed_only(magic):
    """Just the reinterpret trick, no Newton (the 2-op cost point)."""
    return ((("SHR32", 5, 0, 2), ("SUB32", 0, 1, 5)), (magic, 1, 0, 0))


CONSTRUCTIONS = [
    ("quake-newton", trick_newton(0x5F3759DF),
     "Quake III Arena 1999 (source 2005); Wikipedia: Fast inverse square root"),
    ("lomont-newton", trick_newton(0x5F375A86),
     "Lomont 2003, 'Fast Inverse Square Root' — optimal constant for the "
     "standard structure; bound 1.751302e-3"),
    ("quake-seed-only", trick_seed_only(0x5F3759DF),
     "the reinterpret seed alone (cost point; same sources)"),
]

LOMONT_PUBLISHED_BOUND = 1.751302e-3


def build_shelf(pack, mold):
    entries = []
    for name, cand, citation in CONSTRUCTIONS:
        tidy = mold.tidy(cand)
        ok, det = pack.verify_trusted(mold, tidy)
        assert ok, f"shelf entry {name} failed verification: {det}"
        entries.append({"name": name, "cand": tidy,
                        "skeleton": tuple(i for i in tidy[0]),
                        "consts": tidy[1],
                        "max_rel_err": det["max_rel_err"],
                        "cost": mold.native_cost(tidy),
                        "citation": citation})
    by = {e["name"]: e for e in entries}
    assert by["lomont-newton"]["max_rel_err"] <= by["quake-newton"]["max_rel_err"], \
        "literature says Lomont's constant beats Quake's"
    assert by["lomont-newton"]["max_rel_err"] <= LOMONT_PUBLISHED_BOUND * 1.001, \
        "our measured error exceeds the published bound — suspect our verifier"
    return entries


if __name__ == "__main__":
    from domains.rsqrt import RsqrtPack
    from engine.molds_float import FloatProgMold
    pack, mold = RsqrtPack(lo_exp=-2, hi_exp=2), FloatProgMold()
    shelf = build_shelf(pack, mold)
    for e in shelf:
        print(f"  {e['name']}: E={e['max_rel_err']:.6e} ops={e['cost']['ops']}")
    print("rsqrt shelf ok: all entries self-verified; Lomont bound respected")
