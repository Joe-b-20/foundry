"""Evaluate a self-discovered expJ controller checkpoint: per-op exact length-gen, per-divisor
division (does it cross the base-coprime WALL?), and the minimized discovered programs."""
import argparse, random
import torch
from expG_controller import Controller, controller_run, make_problem, INSTRS
import expA_mealy as E

DEVICE = E.DEVICE


def per_op_lengen(model, base=10, widths=(1, 2, 3, 4, 6, 8, 12, 16, 20, 30), n=300):
    print("PER-OP exact length-gen (greedy; the model emits AND runs the program):")
    for op in ("mul", "div"):
        rep = {}
        for w in widths:
            rng = random.Random(99 + w)
            ok = 0
            for _ in range(n):
                A, B, D = make_problem(op, w, base, rng)
                got, _, halted = controller_run(model, op, A, B, D, base)
                exp = A * B if op == "mul" else A // D
                ok += (halted and got == exp)
            rep[w] = ok / n
        print(f"  {op}: " + "  ".join(f"w{w}:{rep[w]:.3f}" for w in widths))


def per_divisor(model, base=10, w=12, n=400):
    print(f"DIVISION by each divisor (w={w}; * = base-coprime 'WALL' divisor, unlearnable per-pass):")
    line = []
    for d in range(2, base):
        rng = random.Random(1000 + d)
        ok = 0
        for _ in range(n):
            A = rng.randint(0, base ** w - 1)
            got, _, halted = controller_run(model, "div", A, 0, d, base)
            ok += (halted and got == A // d)
        star = "" if base % d == 0 else "*"
        line.append(f"/{d}{star}:{ok/n:.3f}")
    print("  " + "  ".join(line))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/expJ_both.pt")
    ap.add_argument("--hidden", type=int, default=64)
    args = ap.parse_args()
    model = Controller(hidden=args.hidden).to(DEVICE)
    model.load_state_dict(torch.load(args.ckpt, map_location=DEVICE))
    model.eval()
    print(f"loaded {args.ckpt}\n")
    per_op_lengen(model)
    print()
    per_divisor(model)
