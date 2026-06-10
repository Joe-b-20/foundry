"""Robustness for addition-from-counting: across seeds, report the discovered body, the NEURAL greedy
length-gen, and the EXTRACTED-body interpreted length-gen (exact, by construction). No traces."""
import random, torch
import expM_add as A

base = 10
WIDTHS = (1, 2, 3, 4, 6, 8, 12, 16, 20, 30)


def interp_lengen(body, widths, n=200, seed=777):
    rep = {}
    for w in widths:
        rng = random.Random(seed + w); ok = 0
        for _ in range(n):
            a, b, d = A.make_problem("add", w, base, rng)
            _, _, got, halted = A.interpret("add", body, a, b, d, base, cap=(base + 4) * (w + 2) + 60)
            ok += (halted and got == a + b)
        rep[w] = ok / n
    return rep


for seed in (0, 1, 2, 3):
    torch.manual_seed(seed)
    model = A.Controller(hidden=64)
    model, info, _ = A.selfdiscover(model, base=base, iters=150, M=2048, wtrain_max=5,
                                    warmup=18, seed=seed, log_every=1000, verbose=False)
    b = info["body"]
    if b is None:
        print(f"seed {seed}: NO body discovered"); continue
    bod = " ".join(A.INSTRS[i] + ("*" if lp else "") for i, lp, g in b)
    neural = A.greedy_acc(model, "add", base, WIDTHS, n=200, seed=123)
    interp = interp_lengen(b, WIDTHS)
    print(f"seed {seed}: body=[{bod}]  (discovered it {info['first_loop_hit']})")
    print("   neural greedy: " + "  ".join(f"w{w}:{neural[w]:.3f}" for w in WIDTHS))
    print("   extracted prog:" + "  ".join(f"w{w}:{interp[w]:.3f}" for w in WIDTHS))
    if seed == 0:
        torch.save(model.state_dict(), "runs/expM_add.pt")
