"""expP_seeds.py — robustness of Newton-based isqrt self-discovery across seeds. Saves seed 0."""
import argparse, torch
import expP_newton as P


def run_seed(seed, iters, mult, add):
    torch.manual_seed(seed)
    model = P.Controller(hidden=64)
    model, info, _ = P.selfdiscover(model, iters=iters, mult=mult, add=add, seed=seed,
                                    log_every=10 ** 9, verbose=False)
    widths = (1, 3, 5, 8, 12, 20, 30)
    rep = {w: P.greedy_acc(model, w, mult, add, n=150, seed=1000 + w) for w in widths}
    return model, info["locked"], rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3")
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--mult", type=int, default=12)
    ap.add_argument("--add", type=int, default=12)
    args = ap.parse_args()
    print(f"device={P.DEVICE}  Newton-isqrt self-discovery robustness  iters={args.iters}")
    print("seed | locked  | length-gen (exact isqrt within budget)")
    for s in [int(x) for x in args.seeds.split(",")]:
        model, locked, rep = run_seed(s, args.iters, args.mult, args.add)
        rstr = " ".join(f"w{w}:{rep[w]:.3f}" for w in rep)
        print(f"  {s}  |  {str(locked):6s} | {rstr}")
        if s == 0 and locked:
            torch.save(model.state_dict(), "runs/expP_newton.pt")
            print("       saved runs/expP_newton.pt")
