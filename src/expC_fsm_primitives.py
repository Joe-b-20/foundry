"""
expC_fsm_primitives.py — ground Exp C's multiplication on the ACTUALLY-EXTRACTED
carry FSM. Everything reduces to the one discovered algorithm (carry):
  - add(x,y)       := the extracted 2-state carry FSM (verified exact, any length)
  - muldigit(x,d)  := repeated addition (d copies of x) using that same add FSM
  - full mult      := loop over B's digits of  add(acc, shl(muldigit(A,Bj), j))
So a tiny neural net's discovered carry, extracted to a finite-state machine, composes
(hierarchically) into exact arbitrary-length multiplication.
"""
from __future__ import annotations
import torch
import expA_mealy as E


def load_fsm_primitives(base=10, ckpt="runs/expA_mealy_d1.pt"):
    model = E.NeuralMealy(base=base, state_dim=1)
    model.load_state_dict(torch.load(ckpt, map_location=E.DEVICE))
    model.to(E.DEVICE).eval()
    # extract the carry FSM (pure-python predict over the discrete tables)
    _, add_fsm, info = E.extract_fsm(model, base=base, probe_width=3)

    def add_fn(x, y):
        return add_fsm(x, y)                 # extracted carry-FSM addition

    def muldigit_fn(x, d):
        acc = 0
        for _ in range(d):                   # single-digit multiply = repeated addition
            acc = add_fn(acc, x)
        return acc

    desc = f"add=extracted carry FSM ({info['n_states']} states); muldigit=repeated-add"
    return add_fn, muldigit_fn, desc


if __name__ == "__main__":
    add_fn, muldigit_fn, desc = load_fsm_primitives()
    print(desc)
    # sanity: the extracted FSM must add and (via repeated add) single-digit-multiply exactly
    import random
    rng = random.Random(0)
    for _ in range(2000):
        a = rng.randint(0, 10**6); b = rng.randint(0, 10**6)
        assert add_fn(a, b) == a + b, (a, b, add_fn(a, b))
    for _ in range(2000):
        a = rng.randint(0, 10**6); d = rng.randint(0, 9)
        assert muldigit_fn(a, d) == a * d, (a, d)
    print("extracted-FSM add and repeated-add muldigit are EXACT on 2000 random checks each.")
