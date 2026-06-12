"""Judge: a thin, domain-blind dispatcher. Each pack provides its own
gates, built on engine services (core runner for combinatorial domains,
the numeric engine for identity domains):

    gate1(mold, tidy)          -> (score_tuple, cost_dict)
        the cheap search-time gate; score tuples order best-last
    verify_trusted(mold, cand) -> (ok, details)
        the trust path — nothing is archived or reported without it
        (verify-on-write); the fast gate never gets the final word

History: judge v0 hardwired the sorting-network gates; the PCF pack made
the shape generic (see TRACKER 2026-06-12, step 4 part 1 lesson)."""


def score(pack, mold, cand):
    tidy = mold.tidy(cand)
    sc, cost = pack.gate1(mold, tidy)
    return tidy, sc, cost


def verify_canonical(pack, mold, cand):
    return pack.verify_trusted(mold, cand)
