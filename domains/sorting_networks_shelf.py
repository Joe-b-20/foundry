"""Reference shelf + known-bounds table for sorting networks.

Shelf entries are GENERATED constructions (bubble, insertion, Batcher
odd-even mergesort): built from their definitions and self-verified against
the pack's checker at build time, so no network listing is trusted from
memory. The bounds table carries published optimality results WITH
citations (RULES.md: bounds numbers only with citations).

Citations (checked 2026-06-12 against the primary papers + the Wikipedia
"Sorting network" summary table):
- Size optimality n<=8: R. W. Floyd & D. E. Knuth; documented in Knuth,
  The Art of Computer Programming vol. 3, sec. 5.3.4.
- Size optimality n=9,10: Codish, Cruz-Filipe, Frank, Schneider-Kamp,
  "Twenty-Five Comparators is Optimal when Sorting Nine Inputs (and
  Twenty-Nine for Ten)" (2014).
- Size optimality n=11,12: J. Harder, "An Answer to the Bose-Nelson Sorting
  Problem for 11 and 12 Channels" (2020), SAT-based.
- Depth optimality n<=8: Knuth TAOCP vol. 3; n=9,10: I. Parberry (1989);
  n=11..16: Bundala & Zavodny, "Optimal Sorting Networks" (2014,
  arXiv:1310.6271) — closed Knuth's 1973 open problem via SAT.
"""

SRC_SIZE = {
    range(2, 9): "Floyd-Knuth (Knuth TAOCP v3 5.3.4)",
    range(9, 11): "Codish-Cruz-Filipe-Frank-Schneider-Kamp 2014",
    range(11, 13): "Harder 2020",
}
SRC_DEPTH = {
    range(2, 9): "Knuth TAOCP v3 5.3.4",
    range(9, 11): "Parberry 1989",
    range(11, 13): "Bundala-Zavodny 2014 (arXiv:1310.6271)",
}

_SIZES = {2: 1, 3: 3, 4: 5, 5: 9, 6: 12, 7: 16, 8: 19, 9: 25, 10: 29,
          11: 35, 12: 39}
_DEPTHS = {2: 1, 3: 3, 4: 3, 5: 5, 6: 5, 7: 6, 8: 6, 9: 7, 10: 7,
           11: 8, 12: 8}


def _src(table, n):
    for rng, s in table.items():
        if n in rng:
            return s
    return None


BOUNDS = {
    n: {"size": _SIZES[n], "size_status": "proven optimal",
        "size_source": _src(SRC_SIZE, n),
        "depth": _DEPTHS[n], "depth_status": "proven optimal",
        "depth_source": _src(SRC_DEPTH, n)}
    for n in _SIZES
}


# --- generated constructions (self-verified, never from memory) -----------

def bubble_network(n):
    return tuple((i, i + 1) for p in range(n - 1) for i in range(n - 1 - p))


def insertion_network(n):
    return tuple((i - 1, i) for p in range(1, n) for i in range(p, 0, -1))


def batcher_network(n):
    """Batcher odd-even mergesort. Generated for the next power of two with
    high-index sentinel wires, then comparators touching sentinel wires are
    dropped (a sentinel holds +inf and sits above all real wires; a
    comparator (i, j) with j a sentinel never moves the real value at i, so
    dropping it is sound). Self-verified by build_shelf regardless."""
    p2 = 1
    while p2 < n:
        p2 *= 2
    pairs = []
    p = 1
    while p < p2:
        k = p
        while k >= 1:
            for j in range(k % p, p2 - k, 2 * k):
                for i in range(0, min(k, p2 - j - k)):
                    if (i + j) // (2 * p) == (i + j + k) // (2 * p):
                        pairs.append((i + j, i + j + k))
            k //= 2
        p *= 2
    return tuple((a, b) for (a, b) in pairs if b < n)


CONSTRUCTIONS = [
    ("bubble", bubble_network, "definitional construction (exchange sort)"),
    ("insertion", insertion_network, "definitional construction"),
    ("batcher-odd-even", batcher_network,
     "K. E. Batcher, 'Sorting networks and their applications', 1968"),
]


def build_shelf(pack, mold):
    """Returns verified shelf entries for pack.n. Raises if any construction
    fails the pack's own checker — the shelf must never contain an unsorted
    'known' network. Entries whose canonical forms collide are merged with
    an alias note."""
    entries = []
    for name, gen, citation in CONSTRUCTIONS:
        cand = gen(pack.n)
        n_ok, total = pack.fast_score(cand)
        assert n_ok == total, f"shelf construction {name} FAILED self-check"
        canonical = mold.tidy(cand)
        cost = mold.native_cost(canonical)
        dup = next((e for e in entries if e["canonical"] == canonical), None)
        if dup:
            dup["aliases"] = dup.get("aliases", []) + [name]
            continue
        entries.append({"name": name, "canonical": canonical, "cost": cost,
                        "citation": citation})
    return entries


if __name__ == "__main__":
    from domains.sorting_networks import SortingNetworkPack
    from engine.molds import ComparatorMold
    for n in range(2, 9):
        pack, mold = SortingNetworkPack(n), ComparatorMold(n)
        shelf = build_shelf(pack, mold)   # raises if any construction is wrong
        assert BOUNDS[n]["size"] <= min(e["cost"]["comparators"] for e in shelf)
    # spot checks
    assert len(bubble_network(4)) == 6 and len(batcher_network(4)) == 5
    assert BOUNDS[6] == {"size": 12, "size_status": "proven optimal",
                         "size_source": "Floyd-Knuth (Knuth TAOCP v3 5.3.4)",
                         "depth": 5, "depth_status": "proven optimal",
                         "depth_source": "Knuth TAOCP v3 5.3.4"}
    print("shelf ok: constructions self-verified for n=2..8; bounds table cited")
