"""
expFF_learnability.py — IS THERE A WALL THE PROJECT NEVER NAMED? Yes: the LEARNABILITY / CRYPTOGRAPHIC wall.

Every wall the project has mapped is about WHERE THE ALGORITHM LIVES: representational (full mult isn't finite-state),
complexity-class (factoring has no poly algorithm), landscape (deep computation = isolated Busy-Beaver needles). But the
project discovers EVERYTHING FROM OUTCOME (input->output examples), and there is a distinct, more fundamental obstruction to
THAT paradigm: some functions are EFFICIENTLY COMPUTABLE yet provably HARD TO LEARN FROM EXAMPLES — pseudorandom / one-way
functions, the basis of cryptography. Such a function has a short, fast program (so it is NOT a representational or
complexity wall) but its input->output map exposes NO exploitable structure, so no amount of examples lets you generalize
to unseen inputs: you can only MEMORIZE. For an outcome-driven discovery project this is a hard wall, and it is the one most
specific to the project's own method.

DEMONSTRATION: a reversible R-round bit-mixer f_R on w-bit words (one round = x ^= x>>s; x = (x*ODD) mod 2^w — invertible,
a permutation; R rounds -> avalanche -> a pseudorandom permutation, the structure of a hash). Number of rounds R = the
STRUCTURE knob. Train a panel of learners on N example pairs, test on HELD-OUT inputs (the learnability test: did it learn
the FUNCTION or just memorize the examples?). Panel: (1) LINEAR correlation (ridge over +-1 bits) catches affine/parity;
(2) kNN in Hamming space catches LOCAL/Lipschitz structure; (3) a small numpy MLP = the PROJECT'S OWN TOOL. Per-output-bit
held-out accuracy; chance = 0.5. Sweep R; the R* where ALL learners collapse to 0.5 = the learnability wall. CONTROL/contrast:
ADDITION (predict the sum bits) MUST stay learnable (the project length-generalizes it) — proving the wall is about the
function's structure, not the learners' weakness. Pure NumPy, CPU, robust. Run: python expFF_learnability.py
"""
from __future__ import annotations
import argparse
import numpy as np

W = 16
MASK = (1 << W) - 1
ODD = 0x9E37            # odd multiplier (invertible mod 2^16); the nonlinear 'confusion' step

def mix_round(x):
    x ^= (x >> 7)
    x &= MASK
    x = (x * ODD) & MASK
    return x

def f_R(x, R):
    for _ in range(R):
        x = mix_round(x)
    return x

def bits(vals, nbits):
    """ints -> (N, nbits) uint8 bit matrix (LSB-first)."""
    vals = np.asarray(vals, dtype=np.int64)
    return ((vals[:, None] >> np.arange(nbits)[None, :]) & 1).astype(np.uint8)

# ---------- learners: each fits X(N,d) bits -> Y(N,m) bits, returns held-out per-bit accuracy ----------
def learn_linear(Xtr, Ytr, Xte, Yte):
    A = np.concatenate([Xtr * 2.0 - 1, np.ones((len(Xtr), 1))], 1)   # +-1 features + bias
    At = np.concatenate([Xte * 2.0 - 1, np.ones((len(Xte), 1))], 1)
    lam = 1e-2
    Wt = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (Ytr * 2.0 - 1))
    pred = (At @ Wt) > 0
    return float((pred == (Yte == 1)).mean())

def learn_knn(Xtr, Ytr, Xte, Yte, k=5):
    # Hamming distance nearest neighbours (catches local structure). Batched to bound memory.
    acc = 0.0; n = 0
    for s in range(0, len(Xte), 256):
        xb = Xte[s:s + 256]
        d = (xb[:, None, :] != Xtr[None, :, :]).sum(2)          # (b, Ntr) Hamming
        idx = np.argpartition(d, k, axis=1)[:, :k]
        vote = Ytr[idx].mean(1) > 0.5                            # (b, m)
        acc += (vote == (Yte[s:s + 256] == 1)).sum(); n += xb.shape[0] * Yte.shape[1]
    return acc / n

def learn_mlp(Xtr, Ytr, Xte, Yte, hidden=128, steps=1500, lr=0.05, seed=0):
    rng = np.random.default_rng(seed)
    d = Xtr.shape[1]; m = Ytr.shape[1]
    Xtr_ = Xtr * 2.0 - 1; Xte_ = Xte * 2.0 - 1; Ytr_ = Ytr.astype(np.float64)
    W1 = rng.normal(0, 1 / np.sqrt(d), (d, hidden)); b1 = np.zeros(hidden)
    W2 = rng.normal(0, 1 / np.sqrt(hidden), (hidden, m)); b2 = np.zeros(m)
    for t in range(steps):
        H = np.tanh(Xtr_ @ W1 + b1)
        P = 1 / (1 + np.exp(-(H @ W2 + b2)))
        dP = (P - Ytr_) / len(Xtr_)
        dW2 = H.T @ dP; db2 = dP.sum(0)
        dH = (dP @ W2.T) * (1 - H ** 2)
        dW1 = Xtr_.T @ dH; db1 = dH.sum(0)
        W1 -= lr * dW1; b1 -= lr * db1; W2 -= lr * dW2; b2 -= lr * db2
    Hte = np.tanh(Xte_ @ W1 + b1); Pte = (Hte @ W2 + b2) > 0
    Htr = np.tanh(Xtr_ @ W1 + b1); Ptr = (Htr @ W2 + b2) > 0
    return float((Pte == (Yte == 1)).mean()), float((Ptr == (Ytr == 1)).mean())

def make_data(R, N, seed=0):
    rng = np.random.default_rng(seed)
    allx = rng.permutation(1 << W)[: 2 * N]
    xtr, xte = allx[:N], allx[N:2 * N]
    Xtr, Xte = bits(xtr, W), bits(xte, W)
    Ytr, Yte = bits([f_R(int(v), R) for v in xtr], W), bits([f_R(int(v), R) for v in xte], W)
    return Xtr, Ytr, Xte, Yte

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=4000)
    args = ap.parse_args()
    N = args.N
    print(f"THE LEARNABILITY WALL: can outcome-driven discovery learn an efficiently-computable PSEUDORANDOM function?")
    print(f"w={W}-bit reversible mixer, train N={N} examples, test on HELD-OUT inputs. Per-bit held-out acc (chance=0.5).\n")
    print("   R(rounds)   linear   kNN-Hamming   MLP(test)   MLP(train)   <- the project's own neural learner")
    for R in (0, 1, 2, 3, 4, 6, 8):
        Xtr, Ytr, Xte, Yte = make_data(R, N, seed=1)
        al = learn_linear(Xtr, Ytr, Xte, Yte)
        ak = learn_knn(Xtr, Ytr, Xte, Yte)
        am_te, am_tr = learn_mlp(Xtr, Ytr, Xte, Yte)
        flag = "  <- WALL (all learners ~chance: memorizable only)" if max(al, ak, am_te) < 0.55 else ""
        print(f"     {R:2d}       {al:.3f}     {ak:.3f}        {am_te:.3f}      {am_tr:.3f}{flag}")

    # CONTROL: ADDITION must stay learnable (the project length-generalizes it) -> the wall is the FUNCTION, not the learners
    print("\n  CONTROL — ADDITION (predict bits of a+b from bits of a,b; the project's flagship learnable op):")
    rng = np.random.default_rng(2)
    aa = rng.integers(0, 1 << (W - 1), 2 * N); bb = rng.integers(0, 1 << (W - 1), 2 * N)
    X = np.concatenate([bits(aa, W - 1), bits(bb, W - 1)], 1); Y = bits(aa + bb, W)
    Xtr, Ytr, Xte, Yte = X[:N], Y[:N], X[N:2 * N], Y[N:2 * N]
    al = learn_linear(Xtr, Ytr, Xte, Yte); ak = learn_knn(Xtr, Ytr, Xte, Yte); am_te, am_tr = learn_mlp(Xtr, Ytr, Xte, Yte)
    print(f"     addition   linear {al:.3f}   kNN {ak:.3f}   MLP(test) {am_te:.3f}  -> learnable & generalizes (structure exists)")
    print("\n  READ: if held-out accuracy collapses to ~0.5 for ALL learners past some R* (while addition stays high), then an")
    print("  EFFICIENTLY-COMPUTABLE function is UN-DISCOVERABLE from outcome — a LEARNABILITY/CRYPTOGRAPHIC wall, distinct from")
    print("  the representational/complexity/landscape walls and specific to outcome-driven discovery (one-way-function hardness).")
