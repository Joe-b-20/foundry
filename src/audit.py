"""
audit.py — verification of TRACKER claims. LOADS saved checkpoints and MEASURES.
No training, no new experiments. Prints measurements only; interpretation deferred
to the human / audit-5.

Saved checkpoints present: add d1/d2/d3, sub d1/d2, mul1 d6 (hidden64, noise0).
NO division checkpoint exists (it didn't meet the save threshold), so the trained
division model cannot be audited without retraining (not done, per instructions).
"""
from __future__ import annotations
import random, math
import numpy as np
import torch

import core_data as cd
import expA_mealy as E
import expA_mul1 as M1

DEVICE = E.DEVICE
SEED = 0


# ---------------------------------------------------------------- loaders
def load(state_dim, ckpt, hidden=16):
    m = E.NeuralMealy(base=10, state_dim=state_dim, hidden=hidden)
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    m.to(DEVICE).eval()
    return m

def add_model():  return load(1, "runs/expA_mealy_d1.pt")
def sub_model():  return load(1, "runs/expA_mealy_sub_d1.pt")
def mul_model():  return load(6, "runs/expA_mul1_d6_h64_n0.0.pt", hidden=64)


# ---------------------------------------------------------------- predict fns
def add_predict(m): return E.net_predict_fn(m, 10)        # predict(a,b)->a+b
def sub_predict(m): return E.net_predict_fn(m, 10)        # predict(a,b)->a-b (model saw a>=b)
def mul_predict(m): return M1.net_predict_mul1(m, 10)     # predict(a,d)->a*d (d single digit)


# ================================================================ AUDIT 1
def sample_band(lo, hi, rng):
    return rng.randint(lo, hi)

def audit1():
    print("\n" + "=" * 70)
    print("AUDIT 1 — out-of-distribution generalization")
    print("=" * 70)
    print("NOTE on ranges: training operands were in [0,999] (<=3 digits). I read")
    print("'2x' and '5x training range' as operand-MAGNITUDE bands. All non-training")
    print("bands have >=4 digits, i.e. they are ALSO beyond the trained sequence")
    print("length (the axis this digit-serial architecture is sensitive to).")
    bands = [
        ("train  [0,999]",        0,            999),
        ("2x     [1000,1999]",    1000,         1999),
        ("5x     [4000,4999]",    4000,         4999),
        ("never  [1e11,1e12]",    10**11,       10**12),
    ]
    specs = [
        ("ADD  a+b",  add_model(),  add_predict, "add"),
        ("SUB  a-b",  sub_model(),  sub_predict, "sub"),
        ("MUL  a*d (d in 0..9)", mul_model(), mul_predict, "mul1"),
    ]
    for name, model, mkpred, op in specs:
        pred = mkpred(model)
        print(f"\n----- {name} -----")
        for bname, lo, hi in bands:
            rng = random.Random(SEED + hash(bname) % 1000)
            N = 2000; correct = 0
            for _ in range(N):
                if op == "mul1":
                    a = sample_band(lo, hi, rng); b = rng.randint(0, 9); exp = a * b
                elif op == "sub":
                    a = sample_band(lo, hi, rng); b = sample_band(lo, hi, rng)
                    if a < b: a, b = b, a
                    exp = a - b
                else:
                    a = sample_band(lo, hi, rng); b = sample_band(lo, hi, rng); exp = a + b
                if pred(a, b) == exp:
                    correct += 1
            print(f"  {bname:22s}  exact-correct = {correct}/{N} = {correct/N:.3f}")
        # 10 examples for the never-seen band
        print(f"  10 random examples in the 'never [1e11,1e12]' band:")
        rng = random.Random(12345)
        for _ in range(10):
            if op == "mul1":
                a = rng.randint(10**11, 10**12); b = rng.randint(0, 9); exp = a * b
            elif op == "sub":
                a = rng.randint(10**11, 10**12); b = rng.randint(10**11, 10**12)
                if a < b: a, b = b, a
                exp = a - b
            else:
                a = rng.randint(10**11, 10**12); b = rng.randint(10**11, 10**12); exp = a + b
            got = pred(a, b)
            print(f"    a={a} b={b}  expected={exp}  got={got}  {'OK' if got==exp else 'WRONG'}")


# ================================================================ AUDIT 2
@torch.no_grad()
def state_trajectory(model, a, b, base=10):
    """Return (output_digits, list_of_state_vectors) for one (a,b)."""
    width = max(E._ndigits(a, base), E._ndigits(b, base)); L = width + 1
    a_oh = E.onehot_seq([a], L, base).to(DEVICE)
    b_oh = E.onehot_seq([b], L, base).to(DEVICE)
    s = model.s0.unsqueeze(0).expand(1, -1).to(DEVICE)
    states = [s[0].cpu().numpy().copy()]
    outs = []
    for t in range(L):
        x = torch.cat([a_oh[:, t], b_oh[:, t]], dim=-1)
        logits, s = model.step(s, x)
        outs.append(int(logits.argmax(-1).item()))
        states.append(s[0].cpu().numpy().copy())
    return outs, states

def audit2():
    print("\n" + "=" * 70)
    print("AUDIT 2 — output variation across 50 addition problems (best op = addition)")
    print("=" * 70)
    m = add_model()
    rng = random.Random(7)
    probs = [(rng.randint(0, 999), rng.randint(0, 999)) for _ in range(50)]
    out_seqs = []
    all_states = []
    for a, b in probs:
        outs, states = state_trajectory(m, a, b)
        out_seqs.append(tuple(outs))
        all_states.extend(states)
    distinct_outputs = len(set(out_seqs))
    print(f"  output = the model's emitted digit sequence (the answer), per problem.")
    print(f"  distinct output digit-sequences across 50 problems: {distinct_outputs}")
    print(f"  (the saved 50 answers all equal a+b: "
          f"{sum(cd.from_digits(list(o),10)==a+b for o,(a,b) in zip(out_seqs,probs))}/50 correct)")
    print(f"  -> outputs vary with input (they are the sums). Showing first 6:")
    for (a, b), o in list(zip(probs, out_seqs))[:6]:
        print(f"     a={a:3d} b={b:3d} -> output digits(LSB-first)={list(o)} = {cd.from_digits(list(o),10)}")
    # internal representation: the recurrent STATE (d=1 -> scalar)
    arr = np.array(all_states).reshape(-1)   # all scalar states across all steps/problems
    print(f"\n  internal STATE values collected across all steps of all 50 problems: "
          f"n={len(arr)}")
    print(f"    min={arr.min():.4f} max={arr.max():.4f}")
    # cluster the scalar states by sign / value
    pos = arr[arr > 0]; neg = arr[arr <= 0]
    print(f"    values <=0: n={len(neg)} mean={neg.mean():.4f} (sd {neg.std():.4f}) "
          f"| values >0: n={len(pos)} mean={pos.mean():.4f} (sd {pos.std():.4f})")
    # distinct states rounded to 1 decimal
    rounded = sorted(set(np.round(arr, 1).tolist()))
    print(f"    distinct state values rounded to 1 decimal: {rounded}")
    print(f"    -> the model REUSES the same small set of internal state values across")
    print(f"       all 50 different problems (the per-problem OUTPUTS differ; the state ALPHABET does not).")
    print(f"  NOTE: the program-search experiments (Exp B/C) instead emit ONE program")
    print(f"  that is applied to ALL inputs; that program does not vary per input by design.")


# ================================================================ AUDIT 3
def pca_2d(X):
    Xc = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    proj = Xc @ Vt[:2].T
    ev = (S**2) / (S**2).sum()
    return proj, ev[:2]

@torch.no_grad()
def collect_states(model, n=4000, single_digit_b=False, base=10, width=3):
    rng = random.Random(3)
    s_all = []
    for _ in range(n):
        a = rng.randint(0, base**width - 1)
        b = rng.randint(0, base-1) if single_digit_b else rng.randint(0, base**width - 1)
        _, states = state_trajectory(model, a, b)
        s_all.extend(states)
    return np.array(s_all)

def audit3():
    print("\n" + "=" * 70)
    print("AUDIT 3 — internal representation inspection")
    print("=" * 70)
    print("ARCHITECTURE NOTE: there are NO 'operation vectors' in this project. Each")
    print("operation is a SEPARATE trained model; there is no shared network conditioned")
    print("on an operation embedding. The closest learned continuous representation is")
    print("the recurrent STATE vector of each model. I dump those below.")

    # addition d=1: scalar state
    print("\n  [ADD d=1] state is 1-DIMENSIONAL (a single scalar). A 2D projection is")
    print("  not meaningful for a 1-D quantity; reporting the value distribution instead.")
    S = collect_states(add_model(), n=2000)
    v = S.reshape(-1)
    print(f"    n={len(v)} values, min={v.min():.4f} max={v.max():.4f}")
    print(f"    histogram (10 bins over [min,max]):")
    hist, edges = np.histogram(v, bins=10)
    for i in range(10):
        print(f"      [{edges[i]:+.3f},{edges[i+1]:+.3f}): {hist[i]}")
    rounded = sorted(set(np.round(v, 1).tolist()))
    print(f"    distinct values rounded to 1 decimal: {rounded}")

    # mul1 d=6: 6-D state
    print("\n  [MUL x1 d=6] state is 6-DIMENSIONAL. PCA to 2D.")
    Sm = collect_states(mul_model(), n=3000, single_digit_b=True)
    print(f"    raw: first 8 state vectors (6-D each):")
    for row in Sm[:8]:
        print("      [" + ", ".join(f"{x:+.3f}" for x in row) + "]")
    proj, ev = pca_2d(Sm)
    print(f"    PCA explained-variance ratio of comp1,comp2 = {ev[0]:.3f}, {ev[1]:.3f}")
    print(f"    2D-projected coords, first 12 points:")
    for row in proj[:12]:
        print(f"      ({row[0]:+.3f}, {row[1]:+.3f})")
    # crude clustering counts
    sign_patterns = set(tuple((Sm[i] > 0).astype(int).tolist()) for i in range(len(Sm)))
    rounded_states = set(tuple(np.round(Sm[i], 1).tolist()) for i in range(len(Sm)))
    print(f"    distinct sign-patterns visited (of 2^6=64 possible): {len(sign_patterns)}")
    print(f"    distinct states rounded to 1 decimal: {len(rounded_states)}")
    # kmeans inertia elbow (numpy kmeans from fsm_extract)
    import fsm_extract as FX
    print(f"    k-means inertia vs k (on the 6-D states):")
    for k in (2, 4, 6, 8, 9, 10, 12, 16):
        C = FX.kmeans(Sm, k, seed=0)
        D = ((Sm[:, None, :] - C[None, :, :])**2).sum(-1).min(1)
        print(f"      k={k:2d}  inertia={D.sum():.1f}")
    # save scatter PNGs for the human
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        plt.figure(); plt.hist(v, bins=40); plt.title("ADD d=1 scalar state values")
        plt.savefig("runs/audit_add_state_hist.png", dpi=90); plt.close()
        plt.figure(); plt.scatter(proj[:, 0], proj[:, 1], s=3, alpha=0.3)
        plt.title("MUL x1 d=6 state PCA-2D"); plt.savefig("runs/audit_mul_state_pca.png", dpi=90); plt.close()
        print("    saved runs/audit_add_state_hist.png and runs/audit_mul_state_pca.png")
    except Exception as e:
        print(f"    (plot save skipped: {e})")


# ================================================================ AUDIT 4
def audit4():
    print("\n" + "=" * 70)
    print("AUDIT 4 — training data audit (best result = addition, expA_mealy.py)")
    print("=" * 70)
    print("  Code path: expA_mealy.make_batch -> train. One TRAINING EXAMPLE is:")
    print("    input  = two operands a,b as one-hot digit sequences (LSB-first),")
    print("             length train_width+1 (a,b sampled uniformly in [0,10^width)).")
    print("    target = the digit sequence of (a+b), one target digit per position.")
    print("  Decoded sample of an actual training batch (E.make_batch(.., op='add', width=3)):")
    a_oh, b_oh, tgt = E.make_batch(6, 3, 10, op="add", seed=1)
    for i in range(6):
        ad = a_oh[i].argmax(-1).tolist(); bd = b_oh[i].argmax(-1).tolist()
        a = cd.from_digits(ad, 10); b = cd.from_digits(bd, 10)
        td = tgt[i].tolist()
        print(f"    a={a:3d} (digits {ad})  b={b:3d} (digits {bd})  -> target digits {td} = {cd.from_digits(td,10)}")
    print("  Were correct ALGORITHMS ever shown? NO. The supervision is ONLY (input, answer):")
    print("    the target is the final sum's digit at each output position. Carries,")
    print("    intermediate state, or any procedure are NEVER provided as targets.")
    print("  Exact loss: CrossEntropyLoss over each output-digit position vs the true")
    print("    digit of (a+b) (expA_mealy.train: lossf(logits.reshape(-1,base), tgt.reshape(-1))).")
    print("    No RL/reward; plain per-position supervised classification of the answer digits.")
    print("  Affects interpretation: (1) the answer IS provided (teacher-forced per position),")
    print("    so 'discovery' refers to the INTERNAL mechanism it must form to predict those")
    print("    digits, not to discovering the answer unsupervised. (2) Main addition result")
    print("    trained at a SINGLE width (3); generalization tested at larger widths.")
    print("  Exp B (program search) used NO supervised answer digits as targets either; its")
    print("    fitness was exact/per-digit correctness of (a,b)->a+b. Still only (input,answer),")
    print("    never the carry procedure.")


# ================================================================ AUDIT 5
def audit5():
    print("\n" + "=" * 70)
    print("AUDIT 5 — sanity check on one 'discovered' claim")
    print("=" * 70)
    print("  CLAIM (TRACKER, Exp A addition, d=1): 'a single continuous scalar, trained")
    print("  from scratch with no hint of carry, discovers the carry bit; the extracted")
    print("  2-state FSM length-generalizes exactly to width 20.'")
    # Exhaustively verify the extracted FSM vs the TRUE carry transducer.
    m = add_model()
    tables, fsm_predict, info = E.extract_fsm(m, base=10, probe_width=3)
    out_table, next_table = tables
    states = info["states"]
    print(f"  Extracted FSM has {len(states)} states: {states}, start={info['start']}")
    # infer each extracted state's carry value k = out on (0,0)
    carry_of = {st: out_table[(st, 0, 0)] for st in states}
    print(f"  inferred carry per state (out on a=0,b=0): {carry_of}")
    # exhaustive check over all (state, a_digit, b_digit)
    total = 0; out_ok = 0; next_ok = 0; bad = []
    # build reverse map carry-value -> state (assume one state per carry 0/1)
    state_for_carry = {}
    for st in states:
        state_for_carry.setdefault(carry_of[st], st)
    for st in states:
        k = carry_of[st]
        for a in range(10):
            for b in range(10):
                total += 1
                true_out = (a + b + k) % 10
                true_next_carry = 1 if (a + b + k) >= 10 else 0
                if out_table[(st, a, b)] == true_out:
                    out_ok += 1
                else:
                    bad.append((st, a, b, "out", out_table[(st, a, b)], true_out))
                if true_next_carry in state_for_carry and next_table[(st, a, b)] == state_for_carry[true_next_carry]:
                    next_ok += 1
    print(f"  exhaustive check over all {total} (state,a_digit,b_digit) transitions:")
    print(f"    output digit matches (a+b+carry)%10 :  {out_ok}/{total}")
    print(f"    next-state matches carry-out rule    :  {next_ok}/{total}")
    if bad[:5]:
        print(f"    first mismatches: {bad[:5]}")
    # also report sampled length-gen of the extracted FSM (already in tracker)
    rep = cd.length_gen_report(fsm_predict, "add", base=10,
                               widths=(3, 6, 12, 20, 40), n_per_width=2000)
    print(f"  sampled exact len-gen of extracted FSM: " +
          " ".join(f"w{w}:{rep[w]:.3f}" for w in rep))
    print("  WEAK interpretation (minimum the evidence supports): on the inputs tested,")
    print("    the model's argmax outputs equal a+b, and a 2-state discretization of its")
    print("    scalar reproduces that; length-gen is verified on SAMPLES (not exhaustively")
    print("    at large widths).")
    print("  STRONG interpretation: the model literally implements the carry transducer")
    print("    (the minimal FSM for base-10 addition) for ALL inputs.")
    print("  Which the evidence supports: the exhaustive 200-transition check above tests")
    print("    the strong claim AT THE FSM LEVEL (all digit/carry combinations). The per-")
    print("    width sampled accuracy tests the trained net. Read the two numbers above and")
    print("    judge: if both 200/200, the EXTRACTED FSM is exactly the carry transducer")
    print("    (provable, since it is finite and total); the claim that the underlying NET")
    print("    equals it everywhere remains supported by sampling, not proof.")


if __name__ == "__main__":
    audit1()
    audit2()
    audit3()
    audit4()
    audit5()
    print("\n[audit.py done]")
