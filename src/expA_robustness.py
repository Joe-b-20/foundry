"""
expA_robustness.py — Is "small state generalizes, big state overfits length" robust,
or was the d=3 subtraction failure a one-off bad minimum?

Trains the neural Mealy machine across (op, state_dim, seed), trains each, extracts
the FSM, and reports the EXTRACTED FSM's exact accuracy at width 20 (trained only on
width 3) plus the number of extracted states. Prints a per-cell grid and per-dim
success rate (fraction of seeds with w20 == 1.000).
"""
from __future__ import annotations
import torch
import expA_mealy as E
import core_data as cd


TRAIN_WIDTHS = None   # set via CLI; None = single width 3

def trial(op, d, seed, steps=4000, base=10):
    torch.manual_seed(seed)
    m = E.NeuralMealy(base=base, state_dim=d)
    E.train(m, base=base, train_width=3, op=op, steps=steps, log_every=10**9,
            train_widths=TRAIN_WIDTHS)
    _, fsm_predict, info = E.extract_fsm(m, base=base, probe_width=3)
    rep = cd.length_gen_report(fsm_predict, op, base=base,
                               widths=(3, 6, 12, 20), n_per_width=500)
    return rep[20], rep[6], info["n_states"]


if __name__ == "__main__":
    import sys, argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("ops", nargs="*", default=["add", "sub"])
    ap.add_argument("--dims", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--train_widths", type=int, nargs="+", default=None)
    args = ap.parse_args()
    ops = args.ops
    TRAIN_WIDTHS = tuple(args.train_widths) if args.train_widths else None
    dims = args.dims
    seeds = list(range(args.seeds))
    print(f"(train_widths={TRAIN_WIDTHS or 'single width 3'})")
    for op in ops:
        print(f"\n===== op={op} — extracted-FSM exact acc @ width 20 (train width 3) =====")
        print("dim | " + " ".join(f"seed{s}" for s in seeds) + " | gen-rate")
        for d in dims:
            cells, ok = [], 0
            for s in seeds:
                acc20, acc6, nst = trial(op, d, s)
                tag = f"{acc20:.2f}({nst})"
                cells.append(f"{tag:>8}")
                if acc20 > 0.999:
                    ok += 1
            print(f" {d}  | " + " ".join(cells) + f" | {ok}/{len(seeds)}")
    print("\n(cell = w20_exact_acc(num_extracted_states); gen-rate = seeds reaching 1.0)")
