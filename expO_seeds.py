"""
expO_seeds.py — robustness of isqrt self-discovery across seeds (does it ALWAYS discover binary search,
length-gen exact?). Compact summary, mirrors expK gcd_seeds / expM_seeds.
Run: python expO_seeds.py --seeds 0,1,2,3 --iters 120
"""
import argparse, torch
import expO_isqrt as O


def run_seed(seed, iters, mult, add):
    torch.manual_seed(seed)
    model = O.Controller(hidden=64)
    model, info, _ = O.selfdiscover(model, iters=iters, mult=mult, add=add, seed=seed,
                                    log_every=10 ** 9, verbose=False)
    widths = (1, 3, 5, 8, 12, 20, 30)
    rep = {w: O.greedy_acc(model, w, mult, add, n=150, seed=1000 + w) for w in widths}
    return info["locked"], rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3")
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--mult", type=int, default=12)
    ap.add_argument("--add", type=int, default=12)
    args = ap.parse_args()
    print(f"device={O.DEVICE}  isqrt self-discovery robustness  iters={args.iters}  cap={args.mult}*w+{args.add}")
    print("seed | locked | algorithm        | length-gen (exact isqrt within budget)")
    algo_name = {"AVG": "BINARY SEARCH", "NEXT": "LINEAR SCAN", None: "NONE"}
    for s in [int(x) for x in args.seeds.split(",")]:
        locked, rep = run_seed(s, args.iters, args.mult, args.add)
        rstr = " ".join(f"w{w}:{rep[w]:.3f}" for w in rep)
        print(f"  {s}  |  {str(locked):4s} | {algo_name[locked]:14s} | {rstr}")
