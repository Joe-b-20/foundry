"""
expZ_invariants.py — a SECOND, independent INTRINSIC-MEANING bridge signal: INVARIANT DISCOVERY.

The project's central moonshot result is the TWO WALLS: target-driven search CONVERGES to the known
optimum (rediscovery engine), pure open-endedness DIVERGES to a noise zoo. The escape is the BRIDGE
between them — an INTRINSIC selection signal that surfaces MEANINGFUL structure with NO human target.
So far only ONE family of bridge signal has been validated: COMPRESSION/sophistication (expX on binary
sequences, expY on cellular automata, where it ranks the class-4 rules against ground truth). The
diagnosis named other untried families; this tests SELF-CONSISTENCY / INVARIANTS.

The signal: a dynamical system (a map f over the finite phase space (Z/p)^2) is INTERESTING iff it
admits a LOW-COMPLEXITY CONSERVED QUANTITY — a low-degree polynomial phi with phi(f(s)) == phi(s).
This is genuinely in the bridge band: it is TARGET-FREE (we never say WHICH invariant to find; the
system's own structure dictates it), and it is the project's exact-verification ethos pushed to a PROOF
— we enumerate the ENTIRE phase space, so "phi is invariant" holds completely over GF(p), not sampled.
Finding phi reduces to the NULL SPACE of the linear system [phi(f(s)) - phi(s) = 0 for all s] over
GF(p), via exact modular Gaussian elimination. The intrinsic score = dimension of the NON-constant
invariant space (0 = generic/chaotic; >0 = integrable/structured).

Validation (à la expY's Wolfram ground truth): known-integrable maps must light up; generic/chaotic
maps must be REJECTED (the noise baseline). Guards: (1) complete enumeration of (Z/p)^2; (2) a
CROSS-PRIME check — a real integer-liftable invariant survives two primes, a field coincidence should
not. GIVEN: the map family, the monomial basis, the degree cap, the field. DISCOVERED: whether an
invariant exists and what it is. Run: python expZ_invariants.py
"""
from __future__ import annotations
import argparse
import numpy as np


# ----------------------------------------------------------------------------- GF(p) linear algebra
def monomials(D):
    """All monomials x^i y^j of total degree i+j <= D, constant (0,0) first."""
    return [(i, d - i) for d in range(D + 1) for i in range(d + 1)]


def eval_monos(monos, X, Y, p):
    """Return (nmon, N) array: row k = X^i * Y^j mod p for monomial (i,j)."""
    out = np.empty((len(monos), X.size), dtype=np.int64)
    for k, (i, j) in enumerate(monos):
        v = np.ones_like(X)
        for _ in range(i):
            v = (v * X) % p
        for _ in range(j):
            v = (v * Y) % p
        out[k] = v
    return out


def gfp_nullspace(M, p):
    """Right null space of M (R x C, entries mod p) over GF(p). Returns (basis_list, rank)."""
    A = (M % p).astype(np.int64).copy()
    R, C = A.shape
    pivot_for_col = {}
    r = 0
    for c in range(C):
        nz = np.nonzero(A[r:, c])[0]
        if nz.size == 0:
            continue
        pr = r + int(nz[0])
        A[[r, pr]] = A[[pr, r]]
        A[r] = (A[r] * pow(int(A[r, c]), p - 2, p)) % p     # normalize pivot to 1
        col = A[:, c].copy(); col[r] = 0                     # eliminate this column from all other rows
        hit = np.nonzero(col)[0]
        if hit.size:
            A[hit] = (A[hit] - np.outer(col[hit], A[r])) % p
        pivot_for_col[c] = r
        r += 1
        if r == R:
            break
    basis = []
    for fc in (c for c in range(C) if c not in pivot_for_col):
        v = np.zeros(C, dtype=np.int64)
        v[fc] = 1
        for c, pr in pivot_for_col.items():
            v[c] = (-A[pr, fc]) % p
        basis.append(v)
    return basis, r


def invariants_of(map_fn, p, D):
    """Discover all degree<=D polynomial invariants of map_fn over GF(p), by complete enumeration of
    (Z/p)^2. Returns (basis, monos): basis = null space of [phi(f(s)) - phi(s)] (includes the constant)."""
    g = np.arange(p, dtype=np.int64)
    X, Y = np.meshgrid(g, g)
    X = X.ravel(); Y = Y.ravel()
    Xn, Yn, mask = map_fn(X, Y, p)
    X, Y, Xn, Yn = X[mask], Y[mask], Xn[mask], Yn[mask]
    monos = monomials(D)
    diff = ((eval_monos(monos, Xn, Yn, p) - eval_monos(monos, X, Y, p)) % p).T   # (N, nmon)
    basis, _ = gfp_nullspace(diff, p)
    return basis, monos


def nontrivial_count(basis):
    """# of independent NON-constant invariants. (Column 0 = the constant monomial is always a null
    vector since phi=1 gives 1-1=0; it occupies exactly one null-space dimension.)"""
    return max(0, len(basis) - 1)


def show_invariant(v, monos, p):
    terms = []
    for c, (i, j) in zip(v, monos):
        c = int(c) % p
        if c == 0:
            continue
        cc = c if c <= p // 2 else c - p                     # symmetric residue, easier to read
        mon = "1" if (i, j) == (0, 0) else ("x" * i + "y" * j)
        terms.append(f"{cc:+d}*{mon}" if mon != "1" else f"{cc:+d}")
    return " ".join(terms) if terms else "0"


def is_invariant(map_fn, coef_vec, monos, p):
    """Exact check: does the given polynomial (coef over monos) satisfy phi(f(s))=phi(s) for ALL
    defined s in (Z/p)^2?  Used for the cross-prime guard and recovery verification."""
    g = np.arange(p, dtype=np.int64)
    X, Y = np.meshgrid(g, g); X = X.ravel(); Y = Y.ravel()
    Xn, Yn, mask = map_fn(X, Y, p)
    X, Y, Xn, Yn = X[mask], Y[mask], Xn[mask], Yn[mask]
    cv = np.array([int(c) % p for c in coef_vec], dtype=np.int64)
    phi_s = (cv[:, None] * eval_monos(monos, X, Y, p)).sum(0) % p
    phi_fs = (cv[:, None] * eval_monos(monos, Xn, Yn, p)).sum(0) % p
    return bool(np.all(phi_s == phi_fs))


# ----------------------------------------------------------------------------- the map zoo
def m_rot90(X, Y, p):       return (-Y) % p, X % p, np.ones(X.size, bool)            # inv: x^2+y^2 (deg 2)
def m_shear(X, Y, p):       return (X + Y) % p, Y % p, np.ones(X.size, bool)          # inv: y      (deg 1)
def m_swap(X, Y, p):        return Y % p, X % p, np.ones(X.size, bool)                # inv: x+y, xy (deg 1,2)
def m_identity(X, Y, p):    return X % p, Y % p, np.ones(X.size, bool)                # everything invariant (control)


def make_henon(c):
    """Area-preserving Henon map (x,y)->(y, -x + y^2 + c): the canonical NON-integrable quadratic map."""
    def f(X, Y, p):
        return Y % p, (-X + Y * Y + c) % p, np.ones(X.size, bool)
    return f


def make_random_poly(seed, p, deg=2):
    """A generic quadratic polynomial map with random GF(p) coefficients (the noise baseline)."""
    rng = np.random.default_rng(seed)
    monos = monomials(deg)
    ca = rng.integers(0, p, len(monos)); cb = rng.integers(0, p, len(monos))
    def f(X, Y, p):
        E = eval_monos(monos, X % p, Y % p, p)
        return (ca[:, None] * E).sum(0) % p, (cb[:, None] * E).sum(0) % p, np.ones(X.size, bool)
    return f


def make_qrt(H):
    """QRT map from a biquadratic H[i][j] = coeff of x^i y^j (i,j in 0..2). Built as the composition of
    the two H-preserving involutions (vertical then horizontal root-switch). PROVABLY preserves H
    (each involution swaps the two roots of H at fixed x or y, which share the same H value)."""
    H = np.array(H, dtype=np.int64)                          # shape (3,3)
    def polyval(coefs, t, p):                                # coefs low->high in t
        v = np.zeros_like(t)
        for k in range(len(coefs) - 1, -1, -1):
            v = (v * t + coefs[k]) % p
        return v
    def f(X, Y, p):
        inv = np.zeros(p, dtype=np.int64)
        for a in range(1, p):
            inv[a] = pow(a, p - 2, p)
        # iota_y: switch the two y-roots at fixed x.  A_y(x)=coeff of y^2, B_y(x)=coeff of y^1.
        Ay = polyval(H[:, 2], X, p); By = polyval(H[:, 1], X, p)
        defy = Ay != 0
        Y2 = (-By * inv[Ay] - Y) % p
        # iota_x: switch the two x-roots at fixed y2.  A_x(y)=coeff of x^2, B_x(y)=coeff of x^1.
        Ax = polyval(H[2, :], Y2, p); Bx = polyval(H[1, :], Y2, p)
        defx = Ax != 0
        X2 = (-Bx * inv[np.where(defx, Ax, 1)] - X) % p
        return X2 % p, Y2 % p, defy & defx
    return f, H


def qrt_hvec(H, monos):
    """Coefficient vector of the biquadratic H over the monomial basis (for recovery verification)."""
    return np.array([int(H[i][j]) if (i <= 2 and j <= 2) else 0 for (i, j) in monos], dtype=np.int64)


# ----------------------------------------------------------------------------- experiment
def report(name, map_fn, p, D, extra=""):
    basis, monos = invariants_of(map_fn, p, D)
    n = nontrivial_count(basis)
    inv_strs = [show_invariant(v, monos, p) for v in basis if any(int(x) % p for x in v[1:])]
    tag = "STRUCTURED (invariant!)" if n > 0 else "generic (no low-deg invariant)"
    print(f"  {name:<22} p={p:<4} deg<= {D}:  #non-const invariants = {n:<2}  [{tag}]  {extra}")
    for s in inv_strs[:4]:
        print(f"        phi = {s}")
    return n, basis, monos


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=23)
    ap.add_argument("--p2", type=int, default=101, help="cross-prime guard")
    ap.add_argument("--D", type=int, default=4, help="max total degree of the searched invariant")
    ap.add_argument("--scan", type=int, default=1, help="run the simple-rule scan (Part 3)")
    args = ap.parse_args()
    p, p2, D = args.p, args.p2, args.D
    print("INVARIANT-DISCOVERY BRIDGE SIGNAL — target-free; does 'admits a conserved quantity' separate\n"
          "the structured (integrable) maps from the generic (chaotic) ones?  [exact over all of (Z/p)^2]\n")

    # --- Part 0: sanity — rotation must yield x^2+y^2 and NOTHING else; identity yields everything ----
    print("  === Part 0: sanity ===")
    report("rotation (x,y)->(-y,x)", m_rot90, p, 2, "<- expect exactly x^2+y^2")
    report("identity (control)", m_identity, p, 2, "<- expect MANY (all monomials invariant)")
    print()

    # --- Part 1: ground-truth POSITIVES (known integrable) vs NEGATIVES (chaotic / random noise) ------
    print("  === Part 1: ground truth — POSITIVES (integrable) should light up; NEGATIVES rejected ===")
    report("rotation", m_rot90, p, D)
    report("shear", m_shear, p, D)
    report("swap", m_swap, p, D)
    qf, H = make_qrt([[3, 1, 2], [1, 4, 1], [2, 1, 1]])     # a random-ish biquadratic H
    nq, bq, mq = report("QRT (nonlinear)", qf, p, D, "<- integrable; should recover its biquadratic H")
    hv = qrt_hvec(H, mq)
    print(f"        [recovery check] is the built-in H = {show_invariant(hv, mq, p)}")
    print(f"        invariant on ALL defined points of (Z/{p})^2 ? {is_invariant(qf, hv, mq, p)}  "
          f"(and over Z/{p2} ? {is_invariant(qf, hv, mq, p2)})")
    print("  --- NEGATIVES (the noise baseline — these MUST be rejected) ---")
    report("Henon c=1 (chaotic)", make_henon(1), p, D, "<- non-integrable; expect 0")
    report("Henon c=3 (chaotic)", make_henon(3), p, D, "<- non-integrable; expect 0")
    for sd in (0, 1, 2):
        report(f"random quad map #{sd}", make_random_poly(sd, p), p, D, "<- noise; expect 0")
    print()

    # --- Part 2: CROSS-PRIME guard — a real invariant survives two primes; a coincidence should not ---
    print(f"  === Part 2: cross-prime guard (p={p} AND p={p2}) — separates real invariants from GF(p) flukes ===")
    for name, fn in [("rotation", m_rot90), ("QRT", qf), ("Henon c=1", make_henon(1)),
                     ("random #0", make_random_poly(0, p)), ("random #1", make_random_poly(1, p))]:
        n1 = nontrivial_count(invariants_of(fn, p, D)[0])
        n2 = nontrivial_count(invariants_of(fn, p2, D)[0])
        verdict = "REAL (both primes)" if (n1 > 0 and n2 > 0) else ("none" if (n1 == 0 and n2 == 0) else "FLUKE (one prime only)")
        print(f"  {name:<14} invariants: GF({p})={n1}  GF({p2})={n2}  -> {verdict}")
    print()

    # --- Part 3: target-free SCAN of a simple-rule family — surface the structured members -----------
    if args.scan:
        print("  === Part 3: target-free scan of the symmetric family f=(y, -x + a*y^2 + b*y + c), "
              "a,b,c in [-2,2] ===")
        hits = []; total = 0
        for a in range(-2, 3):
            for b in range(-2, 3):
                for c in range(-2, 3):
                    def fmap(X, Y, p, a=a, b=b, c=c):
                        return Y % p, (-X + a * Y * Y + b * Y + c) % p, np.ones(X.size, bool)
                    total += 1
                    n1 = nontrivial_count(invariants_of(fmap, p, D)[0])
                    if n1 > 0:
                        n2 = nontrivial_count(invariants_of(fmap, p2, D)[0])
                        if n2 > 0:                            # cross-prime confirmed
                            hits.append((a, b, c, n1))
        print(f"  scanned {total} maps; {len(hits)} admit a cross-prime-confirmed degree<={D} invariant (the "
              f"structured/integrable members, surfaced target-free):")
        for a, b, c, n in hits:
            kind = "linear (a=0)" if a == 0 else "nonlinear"
            print(f"     a={a:+d} b={b:+d} c={c:+d}  #inv={n}  [{kind}]")
        print("  (HONEST: a=0 is the linear/integrable sub-family; nonlinear hits are the interesting ones. "
              "These are KNOWN structures — rediscovery, not a novel object. The point is the signal RANKS them.)")
