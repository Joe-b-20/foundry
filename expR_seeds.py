"""expR_seeds.py — robustness of SELECTION-sort self-discovery across seeds. Saves seed 0.
Run: python expR_seeds.py --seeds 0,1,2,3 --iters 100"""
import argparse, torch
import expR_selection as R


def run_seed(seed, iters, wmax):
    torch.manual_seed(seed)
    model = R.Controller(hidden=48)
    model, info, _ = R.selfdiscover(model, iters=iters, wmax=wmax, seed=seed, log_every=10 ** 9, verbose=False)
    locked = info["locked"]
    lengths = (5, 12, 20, 30, 50)
    rep = R.verify_table(locked, lengths=lengths, n=200) if locked else {L: 0.0 for L in lengths}
    label = R.label_table(locked) if locked else "NONE (not locked)"
    return model, locked, label, rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--wmax", type=int, default=5)
    args = ap.parse_args()
    print(f"device={R.DEVICE}  selection-sort self-discovery robustness  iters={args.iters}  train len<={args.wmax}")
    print("seed | algorithm | length-gen of discovered policy")
    for s in [int(x) for x in args.seeds.split(",")]:
        model, locked, label, rep = run_seed(s, args.iters, args.wmax)
        rstr = " ".join(f"L{L}:{rep[L]:.3f}" for L in rep)
        print(f"  {s}  | {label} | {rstr}")
        if s == 0 and locked:
            torch.save(model.state_dict(), "runs/expR_selection.pt")
            print("       saved runs/expR_selection.pt")
