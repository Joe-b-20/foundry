"""Recognizer v0: is this candidate something we already know?

v0 pipeline (sorting-network grade; grows with the engine):
  1. tidy + exact canonical match against the shelf -> KNOWN(name)
  2. cost-profile comparison against shelf entries (same size+depth)
  3. bounds check: headroom vs the cited optimality table

Honesty rules baked in:
- The label NEW is never emitted by v0. A small shelf cannot support it;
  "not in shelf" is UNRESOLVED with notes (RULES.md: NEW needs a full
  shelf + literature check).
- A candidate strictly below a *proven* optimal bound is labelled
  CONTRADICTS-PROVEN-BOUND: that means OUR verification or shelf is buggy,
  and the right response is to suspect ourselves, loudly.
"""


def recognize(mold, cand, shelf, bounds=None):
    tidy = mold.tidy(cand)
    cost = mold.native_cost(tidy)
    notes = []

    for e in shelf:
        if e["canonical"] == tidy:
            return {"label": "KNOWN", "name": e["name"],
                    "citation": e["citation"], "cost": cost,
                    "notes": ["exact canonical match"]
                            + ([f"aliases: {e['aliases']}"] if e.get("aliases") else [])}

    same_cost = [e["name"] for e in shelf
                 if e["cost"] == cost]
    if same_cost:
        notes.append(f"same size+depth as {same_cost} but different wiring")

    label = "UNRESOLVED"
    if bounds:
        gap = cost["comparators"] - bounds["size"]
        if gap < 0 and "proven" in bounds["size_status"]:
            label = "CONTRADICTS-PROVEN-BOUND"
            notes.append(f"size {cost['comparators']} is BELOW the proven "
                         f"optimum {bounds['size']} ({bounds['size_source']}) "
                         "-> suspect our verifier/shelf first")
        elif gap == 0:
            notes.append(f"size matches the {bounds['size_status']} "
                         f"({bounds['size_source']})")
        else:
            notes.append(f"size is +{gap} above the {bounds['size_status']} "
                         f"{bounds['size']} ({bounds['size_source']}) "
                         "-> headroom exists")
        dgap = cost["depth"] - bounds["depth"]
        if dgap <= 0:
            notes.append(f"depth {cost['depth']} vs optimal {bounds['depth']} "
                         f"({bounds['depth_source']})")
        else:
            notes.append(f"depth is +{dgap} above optimal {bounds['depth']} "
                         f"({bounds['depth_source']})")
    if label == "UNRESOLVED":
        notes.append("not in shelf; NEW would need a full literature check")
    return {"label": label, "cost": cost, "notes": notes}


if __name__ == "__main__":
    from domains.sorting_networks import SortingNetworkPack
    from domains.sorting_networks_shelf import build_shelf, BOUNDS, batcher_network
    from engine.molds import ComparatorMold
    pack, mold = SortingNetworkPack(4), ComparatorMold(4)
    shelf = build_shelf(pack, mold)
    v = recognize(mold, batcher_network(4), shelf, BOUNDS[4])
    assert v["label"] == "KNOWN" and v["name"] == "batcher-odd-even", v
    odd = ((0, 3), (1, 2), (0, 1), (2, 3), (1, 2))     # optimal-size, odd wiring
    v2 = recognize(mold, odd, shelf, BOUNDS[4])
    assert v2["label"] == "UNRESOLVED" and any("matches" in s for s in v2["notes"]), v2
    v3 = recognize(mold, ((0, 1), (2, 3)), shelf, BOUNDS[4])
    assert v3["label"] == "CONTRADICTS-PROVEN-BOUND", v3
    print("recognizer v0 ok:", v["label"], "/", v2["label"], "/", v3["label"])
