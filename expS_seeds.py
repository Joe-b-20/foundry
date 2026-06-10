"""expS_seeds.py — robustness of FACTORIZATION self-discovery across seeds (always trial division?). Saves seed 0.
Run: python expS_seeds.py --seeds 0,1,2,3 --iters 100"""
import argparse, torch
import expS_factor as S


def run_seed(seed, iters, train_hi):
    torch.manual_seed(seed)
    model = S.Controller(hidden=48)
    model, info, _ = S.selfdiscover(model, iters=iters, train_hi=train_hi, seed=seed, log_every=10 ** 9, verbose=False)
    locked = info["locked"]
    mags = (2, 4, 6, 8)
    rep = S.verify_table(locked, mags=mags, n=200) if locked else {m: 0.0 for m in mags}
    label = S.label_table(locked) if locked else "NONE (not locked)"
    return model, locked, label, rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--train_hi", type=int, default=199)
    args = ap.parse_args()
    print(f"device={S.DEVICE}  factorization self-discovery robustness  iters={args.iters}  train n in [2,{args.train_hi}]")
    print("seed | algorithm | exact factorization within O(sqrt n) budget")
    for s in [int(x) for x in args.seeds.split(",")]:
        model, locked, label, rep = run_seed(s, args.iters, args.train_hi)
        rstr = " ".join(f"1e{m}:{rep[m]:.3f}" for m in rep)
        print(f"  {s}  | {label} | {rstr}")
        if s == 0 and locked:
            torch.save(model.state_dict(), "runs/expS_factor.pt")
            print("       saved runs/expS_factor.pt")
