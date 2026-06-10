"""
expI_basechoice.py — LET THE MODEL CHOOSE ITS OWN NUMBER REPRESENTATION (the BASE axis).

Established law (TRACKER 2026-06-04, "representation-dependent"): a digit-serial net
length-generalizes division by a single divisor d IFF d DIVIDES the base (then the
remainder update rem'=(rem*base+a_t)%d is base-modular -> a clean carry-like FSM).
Base 10 -> only /2,/5; base 12 -> /2,/3,/4,/6.

Here the BASE is treated as a CHOICE the SYSTEM makes autonomously, by SEARCH, to maximize
length-generalization on a given divisor SET. For each set we search candidate bases and pick
the one with the best MEAN exact len-gen (eval to ~w16). We PREDICT the winner from the law
first (the base whose factors cover every divisor in the set), then confirm.

Reuses expD_divfixed.py's training verbatim (continuous NeuralMealy, fixed-divisor division,
mixed widths {1..5}, TINY state_dim=4 hidden=64). Training is SHORT per divisor.

Run: wsl bash -lc 'bash /home/joebachir20/math_lab/run.sh expI_basechoice.py'
"""
from __future__ import annotations
import argparse
from math import gcd

import torch

import expA_mealy as E
import expD_divfixed as F   # reuse train_fixed / predict_fixed / lengen_fixed verbatim

DEVICE = E.DEVICE

# ----------------------------------------------------------------------------
# Search configuration
# ----------------------------------------------------------------------------
CANDIDATE_BASES = [3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16]
TRAIN_WIDTHS = (1, 2, 3, 4, 5)
# Eval to ~w16 (the headline long-generalization regime). w1..5 are in-distribution.
EVAL_WIDTHS = (1, 2, 3, 4, 6, 8, 12, 16)
LONG_WIDTHS = (8, 12, 16)   # widths that actually test length-gen (well beyond train)

DIVISOR_SETS = [
    {2, 4, 8},
    {3, 9},
    {7},
    {2, 3, 4, 6},
    {2, 5},
]


def valid_base_for_set(base: int, dset) -> bool:
    """A base is a valid REPRESENTATION for the set only if every divisor is a single
    digit, i.e. d < base for all d (skip a base if any divisor >= base)."""
    return all(d < base for d in dset)


def covers(base: int, dset) -> bool:
    """Law: the set is FULLY learnable at `base` iff every divisor divides the base."""
    return all(base % d == 0 for d in dset)


def predict_base(dset):
    """Predicted winner from the law: the smallest VALID candidate base that COVERS the
    whole set (every divisor divides it). Returns None if the law says no candidate base
    can make the whole set learnable (an honest predicted-partial)."""
    for b in sorted(CANDIDATE_BASES):
        if valid_base_for_set(b, dset) and covers(b, dset):
            return b
    return None


# ----------------------------------------------------------------------------
# Train one fixed divisor at one base, return its len-gen report (reuses expD verbatim)
# ----------------------------------------------------------------------------
def train_and_eval_divisor(base: int, dfix: int, steps: int, eval_widths, n_eval: int):
    torch.manual_seed(0)
    model = E.NeuralMealy(base=base, state_dim=4, hidden=64)
    F.train_fixed(model, base, dfix, steps, train_widths=TRAIN_WIDTHS)
    rep = F.lengen_fixed(F.predict_fixed(model, base, dfix), base, dfix,
                         eval_widths, n=n_eval)
    return rep


def mean_longgen(rep) -> float:
    """Mean exact len-gen over the LONG widths (the part that tests generalization)."""
    return sum(rep[w] for w in LONG_WIDTHS) / len(LONG_WIDTHS)


# ----------------------------------------------------------------------------
# Search: for a divisor set, sweep valid candidate bases, pick best mean long-gen
# ----------------------------------------------------------------------------
def search_set(dset, steps: int, n_eval: int):
    name = "{" + ",".join(str(d) for d in sorted(dset)) + "}"
    pred = predict_base(dset)
    print(f"\n################ SET {name}  (predicted base from law: {pred}) ################")
    if pred is None:
        print(f"  LAW PREDICTION: NO candidate base in {CANDIDATE_BASES} covers all of {name} "
              f"(needs a valid base divisible by lcm of the set). Predicting a PARTIAL: search "
              f"will pick the least-bad base.")

    per_base = {}   # base -> {"per_div": {d: rep}, "mean": float}
    for base in sorted(CANDIDATE_BASES):
        if not valid_base_for_set(base, dset):
            continue
        per_div = {}
        for d in sorted(dset):
            tag = "d|base PASS-predicted" if base % d == 0 else "d!|base FAIL-predicted"
            print(f"\n  -- base {base}, /{d}  ({tag}) --")
            rep = train_and_eval_divisor(base, d, steps, EVAL_WIDTHS, n_eval)
            print("     len-gen: " + "  ".join(f"w{w}:{rep[w]:.3f}" for w in EVAL_WIDTHS))
            per_div[d] = rep
        mean_over_set = sum(mean_longgen(per_div[d]) for d in dset) / len(dset)
        per_base[base] = {"per_div": per_div, "mean": mean_over_set}
        covtag = "(covers set)" if covers(base, dset) else ""
        print(f"  == base {base}: MEAN long-gen over {name} = {mean_over_set:.3f} {covtag}")

    chosen = max(per_base, key=lambda b: per_base[b]["mean"])
    print(f"\n  >>> SET {name}: SEARCH-CHOSEN base = {chosen} "
          f"(mean long-gen {per_base[chosen]['mean']:.3f}); predicted = {pred}; "
          f"{'MATCH' if chosen == pred else 'MISMATCH'}")
    return {"name": name, "predicted": pred, "chosen": chosen, "per_base": per_base}


# ----------------------------------------------------------------------------
# Punchline: per-divisor len-gen for {3,7,9} at its best chosen base vs base 10
# ----------------------------------------------------------------------------
def punchline_3_7_9(steps: int, n_eval: int):
    print("\n################ PUNCHLINE: {3,7,9} chosen-base vs base 10 ################")
    rows = {}
    for d in (3, 7, 9):
        # Best valid candidate base in the SEARCH SET that covers this divisor (d|base, d<base).
        in_set_base = next((b for b in sorted(CANDIDATE_BASES) if d < b and b % d == 0), None)
        # If the fixed search set has none, fall back to the law-NATURAL covering base
        # (smallest multiple of d that is > d, i.e. 2*d) so the punchline is still testable.
        # Flagged honestly as outside the search candidate set.
        chosen_base = in_set_base if in_set_base is not None else 2 * d
        in_set = in_set_base is not None
        # base 10 baseline
        print(f"\n  -- /{d} at BASE 10 (baseline; 10%{d}={10 % d}) --")
        rep10 = train_and_eval_divisor(10, d, steps, EVAL_WIDTHS, n_eval)
        print("     base10 len-gen: " + "  ".join(f"w{w}:{rep10[w]:.3f}" for w in EVAL_WIDTHS))
        loc = "in search set" if in_set else "OUTSIDE search set (law-natural 2*d)"
        print(f"\n  -- /{d} at CHOSEN base {chosen_base} ({chosen_base}%{d}=0; {loc}) --")
        repC = train_and_eval_divisor(chosen_base, d, steps, EVAL_WIDTHS, n_eval)
        print(f"     base{chosen_base} len-gen: " +
              "  ".join(f"w{w}:{repC[w]:.3f}" for w in EVAL_WIDTHS))
        rows[d] = {"chosen_base": chosen_base, "in_set": in_set,
                   "base10": rep10, "chosen": repC}
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2500,
                    help="training steps per divisor (d|base fits fast)")
    ap.add_argument("--n_eval", type=int, default=400,
                    help="samples per width in len-gen eval")
    ap.add_argument("--only", type=str, default="all",
                    help="'all', 'sets', or 'punch' to run a subset")
    ap.add_argument("--set", type=int, default=-1,
                    help="run only DIVISOR_SETS[i] (for parallel launching); -1 = all sets")
    args = ap.parse_args()

    print(f"device={DEVICE}  steps/divisor={args.steps}  n_eval={args.n_eval}")
    print(f"candidate bases: {CANDIDATE_BASES}")
    print(f"train widths: {TRAIN_WIDTHS}; eval widths: {EVAL_WIDTHS}; long widths: {LONG_WIDTHS}")
    print("LAW: net length-generalizes /d iff d|base. Searching base to MAXIMIZE mean long-gen.\n")

    # Print the law-based predictions up front (predict-then-test).
    print("==== LAW-BASED PREDICTIONS (before any training) ====")
    for dset in DIVISOR_SETS:
        nm = "{" + ",".join(str(d) for d in sorted(dset)) + "}"
        pred = predict_base(dset)
        valids = [b for b in CANDIDATE_BASES if valid_base_for_set(b, dset)]
        print(f"  {nm:12s} -> predicted base {pred}   (valid candidate bases: {valids})")
    print("=====================================================")

    set_results = []
    if args.only in ("all", "sets"):
        sets_to_run = ([DIVISOR_SETS[args.set]] if args.set >= 0 else DIVISOR_SETS)
        for dset in sets_to_run:
            set_results.append(search_set(dset, args.steps, args.n_eval))

    punch = None
    if args.only in ("all", "punch"):
        punch = punchline_3_7_9(args.steps, args.n_eval)

    # ---- Final compact summary tables ----
    print("\n\n=================== FINAL SUMMARY ===================")
    if set_results:
        print("SET                | predicted | chosen | mean-long-gen@chosen")
        for r in set_results:
            mean_c = r["per_base"][r["chosen"]]["mean"]
            print(f"  {r['name']:16s} | {str(r['predicted']):9s} | {r['chosen']:6d} | {mean_c:.3f}")
    if punch:
        print("\nPUNCHLINE {3,7,9}: per-divisor mean long-gen (w8,w12,w16)")
        print("  div | chosen base       | base10 long-gen | chosen-base long-gen")
        for d in (3, 7, 9):
            row = punch[d]
            b10 = mean_longgen(row["base10"])
            cb = row["chosen_base"]
            note = "" if row["in_set"] else " (outside search set)"
            cgen = f"{mean_longgen(row['chosen']):.3f}"
            print(f"   /{d}  | {str(cb) + note:17s} | {b10:.3f}           | {cgen}")
    print("====================================================")
