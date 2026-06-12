"""Reference shelf + bounds for bilinear decompositions (polymul2).

Shelf entries are written from their textbook definitions and SELF-VERIFIED
against the pack (exact tensor identity) at build time. Citations:
- naive-4: the schoolbook method (definitional).
- karatsuba-3 (both sign variants): Karatsuba & Ofman, "Multiplication of
  multidigit numbers on automata", 1962.

Lower bound: rank(polymul2 tensor) >= 3 by the standard FLATTENING (slice
span) argument, COMPUTED EXACTLY HERE rather than quoted: the three output
slices S_k (2x2 integer matrices) of any rank-R decomposition lie in the
span of R rank-one matrices, so R >= dim span{S_0,S_1,S_2}; integer row
reduction below shows that dimension is 3. With the Karatsuba witness
(R=3), rank = 3 exactly — so R=3 is PROVEN optimal for this target and
the recognizer's CONTRADICTS-PROVEN-BOUND logic applies below it.
"""

from domains.bilinear import BilinearPack, TARGETS
from engine.molds_bilinear import BilinearMold

NAIVE4 = (((1, 0), (1, 0), (1, 0, 0)), ((1, 0), (0, 1), (0, 1, 0)),
          ((0, 1), (1, 0), (0, 1, 0)), ((0, 1), (0, 1), (0, 0, 1)))
KARATSUBA_PP = (((1, 0), (1, 0), (1, -1, 0)),
                ((1, 1), (1, 1), (0, 1, 0)),
                ((0, 1), (0, 1), (0, -1, 1)))
KARATSUBA_MM = (((1, 0), (1, 0), (1, 1, 0)),
                ((1, -1), (1, -1), (0, -1, 0)),
                ((0, 1), (0, 1), (0, 1, 1)))

CONSTRUCTIONS = [
    ("naive-4", NAIVE4, "schoolbook (definitional)"),
    ("karatsuba-3 (+,+)", KARATSUBA_PP, "Karatsuba & Ofman 1962"),
    ("karatsuba-3 (-,-)", KARATSUBA_MM, "Karatsuba & Ofman 1962"),
]


def integer_rank(rows):
    """Exact rank by fraction-free Gaussian elimination over the integers."""
    rows = [list(r) for r in rows]
    rank, col, ncols = 0, 0, len(rows[0])
    while rank < len(rows) and col < ncols:
        piv = next((r for r in range(rank, len(rows)) if rows[r][col]), None)
        if piv is None:
            col += 1
            continue
        rows[rank], rows[piv] = rows[piv], rows[rank]
        for r in range(rank + 1, len(rows)):
            if rows[r][col]:
                a, b = rows[rank][col], rows[r][col]
                rows[r] = [a * x - b * y for x, y in zip(rows[r], rows[rank])]
        rank += 1
        col += 1
    return rank


def flattening_lower_bound(target_name):
    t = TARGETS[target_name]
    slices = [[t[(i * 2 + j) * 3 + k] for i in range(2) for j in range(2)]
              for k in range(3)]
    return integer_rank(slices)


def build_shelf(pack, mold):
    entries = []
    for name, cand, citation in CONSTRUCTIONS:
        tidy = mold.tidy(cand)
        ok, det = pack.verify_trusted(mold, tidy)
        assert ok, f"shelf construction {name} FAILED self-check: {det}"
        entries.append({"name": name, "canonical": mold.canonical_key(tidy),
                        "cost": mold.native_cost(tidy), "citation": citation})
    return entries


BOUNDS = {"polymul2": {
    "rank": 3,
    "lower_bound": "flattening/slice-span bound, computed exactly here "
                   "(see flattening_lower_bound)",
    "witness": "Karatsuba & Ofman 1962 (R=3), self-verified",
    "status": "proven optimal",
}}


if __name__ == "__main__":
    pack, mold = BilinearPack(), BilinearMold()
    shelf = build_shelf(pack, mold)
    assert len(shelf) == 3
    lb = flattening_lower_bound("polymul2")
    assert lb == 3, lb
    assert min(e["cost"]["mults"] for e in shelf) == 3 == BOUNDS["polymul2"]["rank"]
    # the two Karatsuba sign-variants are genuinely different canonical keys
    # (the symmetry group used is only the target's a<->b and reversal)
    keys = {e["name"]: e["canonical"] for e in shelf}
    print("bilinear shelf ok: 3 constructions self-verified; "
          f"flattening lower bound = {lb} -> R=3 proven optimal "
          f"(distinct canonical keys: {len(set(keys.values()))})")
