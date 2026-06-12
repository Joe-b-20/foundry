"""expQ_seeds.py — robustness of SORTING self-discovery across seeds (always bubble sort, exact len-gen?).
Saves seed 0. Run: python expQ_seeds.py --seeds 0,1,2,3 --iters 100"""
import argparse, torch
import expQ_sort as Q


def run_seed(seed, iters, wmax):
    torch.manual_seed(seed)
    model = Q.Controller(hidden=48)
    model, info, _ = Q.selfdiscover(model, iters=iters, wmax=wmax, seed=seed, log_every=10 ** 9, verbose=False)
    locked = info["locked"]
    lengths = (5, 12, 20, 30, 50)
    rep = Q.verify_table(locked, lengths=lengths, n=200) if locked else {L: 0.0 for L in lengths}
    label = Q.label_table(locked) if locked else "NONE (not locked)"
    return model, locked, label, rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--wmax", type=int, default=5)
    args = ap.parse_args()
    print(f"device={Q.DEVICE}  sorting self-discovery robustness  iters={args.iters}  train len<=%d" % args.wmax)
    print("seed | algorithm | length-gen of discovered policy (exact sort, trained len<=%d)" % args.wmax)
    for s in [int(x) for x in args.seeds.split(",")]:
        model, locked, label, rep = run_seed(s, args.iters, args.wmax)
        rstr = " ".join(f"L{L}:{rep[L]:.3f}" for L in rep)
        print(f"  {s}  | {label} | {rstr}")
        if s == 0 and locked:
            torch.save(model.state_dict(), "runs/expQ_sort.pt")
            print("       saved runs/expQ_sort.pt")
