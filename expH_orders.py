"""
expH_orders.py — train the unified 4-op model under DIFFERENT curriculum ORDERS.

Question (user): does the ORDER ops are introduced change (a) accuracy, (b) what is
learned, (c) the internal mechanism/geometry? Prior interp (TRACKER 2026-06-05) found the
add-first dynamic model puts CARRY on a clean axis and SUPERIMPOSES borrow as a rotated
direction in the same scratchpad, while CO-TRAIN (no order) partitions carry/borrow onto
separate clean axes. That yields a sharp, pre-registered hypothesis:

  H1 (order->geometry): the op introduced FIRST claims the clean axis-aligned code; later
     ops are superimposed/rotated in the shared scratchpad subspace.
  H2 (mechanism invariance): regardless of order, the MECHANISM is the same (carry=causal
     binary bit, borrow=borrow-not-negate, op-code=selector, mult-carry=b-entangled); only
     the geometry (which dims, axis vs rotated) changes.
  H3 (accuracy): with rehearsal (dynamic-blended) every order should reach ~the same final
     accuracy and hit the SAME walls (coprime div); genuinely unsure if leading with the
     hard/entangled op (mul) or the partially-walled op (div) hurts.

Controlled experiment: SAME architecture (d=8 h=96), SAME init (torch seed 0), SAME budget
and data seeds, SAME regime (dynamic-blended: focus-0.6 rehearsal intro phases + deficit-
weighted balancing). The ONLY thing that differs between runs is `order`. add-first == the
existing dynamic model's order, so it doubles as a replication check.

Run: bash run.sh expH_orders.py [--smoke] [--orders addfirst subfirst mulfirst reverse]
"""
from __future__ import annotations
import argparse
import torch
import expF_unified as F

# first-op spans {add, sub, mul, div} across these four orders
ORDERS = {
    "addfirst": ["add", "sub", "mul", "div"],   # == existing dynamic model (replication)
    "subfirst": ["sub", "add", "mul", "div"],   # sharpest test: borrow before carry
    "mulfirst": ["mul", "add", "sub", "div"],   # lead with the entangled/hard op
    "reverse":  ["div", "mul", "sub", "add"],   # div-first AND add-last
}

if __name__ == "__main__":
    import os
    os.makedirs("runs", exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps_phase", type=int, default=8000)
    ap.add_argument("--steps_balance", type=int, default=10000)
    ap.add_argument("--focus", type=float, default=0.60)
    ap.add_argument("--orders", nargs="+", default=list(ORDERS))
    ap.add_argument("--seed", type=int, default=0,
                    help="init seed; !=0 suffixes the checkpoint name (for robustness/seed checks)")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    base = 10
    if args.smoke:
        args.steps_phase = 600; args.steps_balance = 800
    sfx = "" if args.seed == 0 else f"_s{args.seed}"

    # neural division restricted to learnable divisors {1,2,5} (matches the dynamic regime;
    # coprime divisors are an architectural wall handled by composition, not training).
    F.DIV_POOL = [1, 2, 5]
    print(f"device={F.DEVICE}  steps_phase={args.steps_phase} steps_balance={args.steps_balance}")
    print(f"  [neural div restricted to {F.DIV_POOL}]")

    for tag in args.orders:
        order = ORDERS[tag]
        print(f"\n{'#'*78}\n# ORDER {tag}:  {' -> '.join(order)}\n{'#'*78}")
        torch.manual_seed(args.seed)               # SAME init for every order (per seed)
        m = F.UnifiedMealy(base=base, state_dim=8, hidden=96)
        print(f"  params: {sum(p.numel() for p in m.parameters())}  seed={args.seed}")
        F.train_dynamic_blended(m, args.steps_phase, args.steps_balance,
                                base=base, focus=args.focus, order=order)
        # does order change accuracy? compact per-op length-gen
        F.lengen_table(m, base, widths=(1, 2, 4, 8, 12, 20))
        path = f"runs/expH_order_{tag}{sfx}.pt"
        torch.save(m.state_dict(), path)
        print(f"saved {path}")
