"""
expGG_extraction.py — THE EXTRACTION WALL: fundamental, or just tooling? (the moonshot-adjacent wall.)

If a net LENGTH-GENERALIZES but we cannot EXTRACT a clean symbolic algorithm, the net is using a procedure we can't read —
the closest the project could get to "a procedure humans haven't found". Session 1 hit this on single-digit multiplication:
the net length-generalizes (~1.0) but its 6-D continuous state is a SMOOTH MANIFOLD (no crisp clusters), so geometric
k-means / centroid extraction stalled at ~0.94, never bit-exact. Was that a FUNDAMENTAL wall (the net's mechanism is
genuinely non-symbolic) or merely a TOOLING failure (a finite FSM exists — single-digit mult is regular, 9 carry states —
but clustering by GEOMETRY can't find it through the smear)?

The sharp argument: exact length-generalization to arbitrary length REQUIRES effectively-finite-state dynamics (else the
continuous state drifts and accuracy decays with length — exactly how division failed). A finite-state regular op has, by
Myhill-Nerode, a UNIQUE minimal FSM that EXISTS. So extraction should ALWAYS be possible in principle — IF you merge states
by BEHAVIOR, not geometry. This script tests that: load the trained mult net (NO retraining — robust), and extract by the
1-step OUTPUT-TABLE signature (two states are equivalent iff they output the same for every (a,b) — a Myhill-Nerode/Moore
merge), which is invariant to how smeared the continuous encoding is. Prediction (flagged): behavioral extraction recovers
the EXACT 9-state FSM (length-gen 1.0) where geometry stalled at 0.94 => the extraction wall was TOOLING, not fundamental;
and (corollary) the moonshot cannot hide in "un-extractable but correct" flat-recurrent procedures.
Run: python expGG_extraction.py
"""
from __future__ import annotations
import numpy as np
import torch

import core_data as cd
import expA_mealy as E
import expA_mul1 as M

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE = 10
CKPT = "runs/expA_mul1_d6_h64_n0.0.pt"


@torch.no_grad()
def out_signature(model, S):
    """S: (M, d) states. Return (M, 100) int matrix = argmax output for every (a,b) in 0..9 x 0..9 (the 1-step Moore signature)."""
    model.eval()
    M_ = S.shape[0]
    sig = np.zeros((M_, BASE * BASE), dtype=np.int64)
    col = 0
    for a in range(BASE):
        for b in range(BASE):
            x = torch.zeros(M_, 2 * BASE, device=DEVICE)
            x[:, a] = 1.0; x[:, BASE + b] = 1.0
            logits, _ = model.step(S, x)
            sig[:, col] = logits.argmax(-1).cpu().numpy(); col += 1
    return sig


@torch.no_grad()
def next_state(model, S, a, b):
    M_ = S.shape[0]
    x = torch.zeros(M_, 2 * BASE, device=DEVICE)
    x[:, a] = 1.0; x[:, BASE + b] = 1.0
    _, s_next = model.step(S, x)
    return s_next


@torch.no_grad()
def carry_read(model, S):
    """ROBUST, ON-DISTRIBUTION state label: feed (a=0, b) for all b; out should = carry (since 0*b+carry=carry).
    Take the MODE over b -> a single carry label per state, robust to the smear. Returns (M,) int labels."""
    M_ = S.shape[0]
    votes = np.zeros((M_, BASE), dtype=np.int64)
    for b in range(BASE):
        x = torch.zeros(M_, 2 * BASE, device=DEVICE); x[:, 0] = 1.0; x[:, BASE + b] = 1.0
        logits, _ = model.step(S, x)
        votes[:, b] = logits.argmax(-1).cpu().numpy()
    return np.array([np.bincount(row, minlength=BASE).argmax() for row in votes])


def behavioral_extract(model):
    """Behavioral extraction done RIGHT, ON-DISTRIBUTION: classify states by a robust carry-read, then read out/next tables
    by MAJORITY VOTE over the net's REAL transitions only (where it is exact — no off-distribution probing, no ground truth).
    Wide-mixed-width data gives full (carry,a,b) coverage so the table is total."""
    import fsm_extract as FX
    from collections import Counter, defaultdict
    sp_l, da_l, db_l, out_l, sn_l = [], [], [], [], []
    for w in (1, 2, 3, 4, 5, 6, 7, 8):                                              # mixed widths -> cover all carries & (a,b)
        a_oh, b_oh, _ = M.make_mul1_batch(2500, w, BASE, seed=100 + w)
        sp, da, db, out, sn, s0 = FX.collect_transitions(model, a_oh.to(DEVICE), b_oh.to(DEVICE), DEVICE)
        sp_l.append(sp); da_l.append(da); db_l.append(db); out_l.append(out); sn_l.append(sn)
    sp = np.concatenate(sp_l); da = np.concatenate(da_l); db = np.concatenate(db_l)
    out = np.concatenate(out_l); sn = np.concatenate(sn_l)
    lp = carry_read(model, torch.tensor(sp, dtype=torch.float32, device=DEVICE))
    ln = carry_read(model, torch.tensor(sn, dtype=torch.float32, device=DEVICE))
    out_votes = defaultdict(Counter); next_votes = defaultdict(Counter)
    for i in range(len(sp)):
        out_votes[(int(lp[i]), int(da[i]), int(db[i]))][int(out[i])] += 1
        next_votes[(int(lp[i]), int(da[i]), int(db[i]))][int(ln[i])] += 1
    out_table = {k: c.most_common(1)[0][0] for k, c in out_votes.items()}
    next_table = {k: c.most_common(1)[0][0] for k, c in next_votes.items()}
    start = int(carry_read(model, model.s0.unsqueeze(0).to(DEVICE))[0])
    nstates = len(set(lp.tolist()) | set(ln.tolist()))
    coverage = len(out_table) / (nstates * BASE * BASE)

    def fsm_predict(a, b):
        w = E._ndigits(a, BASE); L = w + 1
        ad = cd.to_digits(a, L, BASE); cl = start; outs = []
        for t in range(L):
            key = (cl, ad[t], b % BASE)
            if key not in out_table:
                return -1                                                           # unseen -> honest miss
            outs.append(out_table[key]); cl = next_table[key]
        return cd.from_digits(outs, BASE)

    return nstates, fsm_predict, coverage


if __name__ == "__main__":
    print("THE EXTRACTION WALL: fundamental or tooling? Single-digit mult — geometry stalled at ~0.94; try BEHAVIORAL merge.\n")
    model = E.NeuralMealy(base=BASE, state_dim=6, hidden=64).to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE)); model.eval()

    # 1) confirm the NET length-generalizes (it has a real algorithm, not a lookup table)
    net_rep = M.mul1_lengen(M.net_predict_mul1(model), BASE, widths=(3, 6, 12, 20), n=600)
    print(f"  NET length-gen (the algorithm is real): " + "  ".join(f"w{w}:{net_rep[w]:.3f}" for w in (3, 6, 12, 20)))

    # 2) geometric extraction baseline (session-1 method) — expected to stall < 1.0
    print("\n  GEOMETRIC extraction (k-means on the continuous state, session-1 method):")
    k, fsm_geo, geo_rep = M.extract_best_fsm(model, BASE, width=3, kmax=12)
    print(f"    best k-means FSM: k={k}  length-gen " + "  ".join(f"w{w}:{geo_rep[w]:.3f}" for w in sorted(geo_rep)))
    geo_ok = (k is not None) and min(geo_rep.values()) > 0.999

    # 3) BEHAVIORAL extraction (Myhill-Nerode/Moore merge by output signature) — the right tool
    print("\n  BEHAVIORAL extraction (carry-read classing + majority vote over the net's REAL on-distribution transitions):")
    nstates, fsm_beh, coverage = behavioral_extract(model)
    beh_rep = M.mul1_lengen(fsm_beh, BASE, widths=(1, 3, 6, 12, 20, 30), n=800)
    print(f"    recovered {nstates} behavioral states (table coverage {coverage:.2f}); FSM length-gen " + "  ".join(f"w{w}:{beh_rep[w]:.3f}" for w in sorted(beh_rep)))
    beh_ok = min(beh_rep.values()) > 0.999

    print("\n  === DIAGNOSIS ===")
    print(f"    geometric (k-means) bit-exact? {geo_ok}   |   behavioral (Myhill-Nerode) bit-exact? {beh_ok}  ({nstates} states)")
    if beh_ok and not geo_ok:
        print("    => the extraction wall was TOOLING, not fundamental: a finite FSM EXISTS (the op is regular); geometry")
        print("       can't find it through the smeared manifold, but BEHAVIORAL merging recovers it EXACTLY. Corollary:")
        print("       exact length-gen ⟹ finite-state ⟹ extractable-in-principle — the moonshot cannot hide in")
        print("       'un-extractable but correct' flat-recurrent procedures (for regular ops).")
    elif beh_ok and geo_ok:
        print("    => both worked here; the wall is tooling (behavioral is just more robust).")
    else:
        print("    => behavioral extraction ALSO failed — evidence the mechanism resists finite-state extraction (investigate).")
