"""Independent optimality certificates for tiny n by exhaustion.

Claim shape: "no comparator network with <= L comparators sorts all
length-n inputs" — established by enumerating EVERY comparator sequence of
length L over the n-choose-2 pairs and checking each against all 2^n binary
inputs (complete by the 0/1 principle).

Lemma (why checking only length L suffices for all k <= L): a comparator
applied to an already-sorted array changes nothing (for (i, j) with i < j,
m[i] <= m[j] already holds), so appending comparators to a sorting network
keeps it sorting. A length-k sorter would therefore extend to a length-L
sorter; if no length-L sorter exists, none shorter exists either.

This independently confirms the published optimal sizes (Floyd-Knuth via
Knuth TAOCP v3 5.3.4) for n = 2, 3, 4 with our own machinery. n = 5 needs
10^8 sequences — that is the SAT-proposer's job, not brute force's.

Run from repo root:  python3 -m scripts.certify_tiny_optimality
"""

import itertools
import json
import sys
import time
from pathlib import Path

from domains.sorting_networks import SortingNetworkPack
from domains.sorting_networks_shelf import BOUNDS, batcher_network


def none_of_length_sorts(pack, pairs, L):
    """Exhaustively check that no length-L sequence sorts. Returns count."""
    count = 0
    for seq in itertools.product(pairs, repeat=L):
        count += 1
        n_ok, total = pack.fast_score(seq)
        if n_ok == total:
            raise AssertionError(f"counterexample: {seq} sorts n={pack.n}")
    return count


def main():
    t0 = time.time()
    out_dir = Path("runs") / f"certify_tiny-{int(t0)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    certs = []
    for n in (2, 3, 4):
        pack = SortingNetworkPack(n)
        pairs = list(itertools.combinations(range(n), 2))
        published = BOUNDS[n]["size"]
        L = published - 1
        enumerated = none_of_length_sorts(pack, pairs, L) if L > 0 else 0
        witness = batcher_network(n)
        n_ok, total = pack.fast_score(witness)
        assert n_ok == total and len(witness) == published, \
            f"witness for n={n} has size {len(witness)}, expected {published}"
        certs.append({
            "n": n,
            "claim": (f"the optimal sorting-network size for n={n} is exactly "
                      f"{published}"),
            "lower_bound": {"method": "exhaustive enumeration over candidate "
                                      "space + monotone-extension lemma",
                            "length_checked": L,
                            "sequences_enumerated": enumerated,
                            "input_check": f"all 2^{n} binary vectors "
                                           "(complete by the 0/1 principle)"},
            "upper_bound": {"witness": "batcher-odd-even (self-verified)",
                            "size": len(witness)},
            "level": "L1-exhaustive-candidate-space (our own machinery; an "
                     "external re-checkable artifact would make it L2)",
            "matches_published": {"value": published,
                                  "source": BOUNDS[n]["size_source"]},
        })
        print(f"n={n}: no size-{L} sorter among {enumerated} sequences; "
              f"witness of size {len(witness)} exists -> optimal = {published} "
              f"(matches {BOUNDS[n]['size_source']})")
    (out_dir / "certificates.json").write_text(json.dumps(
        {"seconds": round(time.time() - t0, 2), "certificates": certs}, indent=2))
    print(f"certificates written to {out_dir}/certificates.json "
          f"in {round(time.time() - t0, 2)}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
