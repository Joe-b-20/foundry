"""Robustness: run the muldigit=repeated-add self-discovery across seeds; report discovered body
+ exact length-gen to w30. (No traces; outcome-only.)"""
import random, torch
import expM_muldigit as M

base = 10
WIDTHS = (1, 2, 3, 4, 6, 8, 12, 16, 20, 25, 30)
for seed in (0, 1, 2, 3):
    torch.manual_seed(seed)
    model = M.Controller(hidden=64)
    model, info, _ = M.selfdiscover(model, base=base, iters=80, M=2048, wtrain_max=4,
                                    seed=seed, log_every=1000, verbose=False)
    b = info["body"]
    bod = None if b is None else " ".join(M.INSTRS[i] + ("*" if lp else "") for i, lp in b)
    rep = M.greedy_acc(model, "mul", base, WIDTHS, n=200, seed=123)
    acc = "  ".join(f"w{w}:{rep[w]:.3f}" for w in WIDTHS)
    print(f"seed {seed}: discovered@it{info['first_loop_hit'] or '?'} body=[{bod}]")
    print(f"         len-gen {acc}")
    if seed == 0:
        torch.save(model.state_dict(), "runs/expM_muldigit.pt")
