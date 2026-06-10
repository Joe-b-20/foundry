"""
fn_telescope.py — THE FUNCTION TELESCOPE (phase 4: Frontier-2 recognition, made concrete).

The oe/loop experiments left the project holding STRUCTURED UNNAMED functions — top
organisms with edge-of-chaos-maximal I/O maps that match nothing in an 85-entry named
suite (nearest-named bit-similarity ~0.55-0.77 ≈ chance). "Unnamed" so far just means
"not in a finite list." This is the instrument that turns that into an actual
characterization: given a 2-input 16-bit function (as a stack-VM program), run a battery
of structure analyses and produce a classification report.

THE BATTERY (each test = a different lens on what kind of object f is):
  1. ENCODING-EQUIVALENCE: is f equal to T_out(g(T_a(a), T_b(b))) for a named g and
     lightweight encodings T in {id, ~, <<1, >>1, bit-reverse} (+ arg swap)? Catches
     "named op in disguise" — the carry-save lesson (same function, re-encoded).
  2. INTEGER-POLYNOMIAL FIT mod 2^16: solve f(a,b) = sum c_ij a^i b^j (mod 2^16),
     deg<=3, by exact Gaussian elimination over Z/2^16 on probe equations; verify on
     held-out probes. Catches arithmetic-polynomial functions (a+b, 2a+3ab, ...).
  3. ANF / ALGEBRAIC DEGREE over GF(2): Mobius transform on random 16-bit subcubes
     (8 bits of a x 8 bits of b) -> per-output-bit algebraic degree + monomial density.
     deg 1 = linear/affine (xor-like); deg 2-3 sparse = structured arithmetic (carry
     chains); high deg + dense ~ random-like mixing.
  4. BLR LINEARITY DISTANCE: Pr[f(x xor y) = f(x) xor f(y) xor f(0)] over random pairs
     (1.0 = GF(2)-linear).
  5. EQUIVARIANCE: shift (f(a<<1,b<<1) == f(a,b)<<1), xor-translation
     (f(a^c,b^c) == f(a,b)^c), arg-symmetry (f(a,b)==f(b,a)).
  6. BIT-DEPENDENCY MATRIX: sensitivity of each output bit to each input bit (sampled
     single-bit flips) -> triangular = carry-like propagation; diagonal-banded = local;
     dense = global mixing. Plus per-output-bit balance/bias.
CONTROLS: hand-written VM programs for a+b and a^b must classify exactly (gate: if the
controls misclassify, the report is void).

Run:  python fn_telescope.py --selftest
      python fn_telescope.py --targets oe_loop   (the banked unnamed attractors)
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import torch

import gpu_avida_oe as oe
import gpu_avida_loop as loop

DEV = oe.DEV
MASK = 0xFFFF
S_STACK = 12


def parse_prog(prog_str, opn):
    """Decode 'a b add dup ...' back to an opcode array."""
    idx = {nm: i for i, nm in enumerate(opn)}
    return np.array([idx[t] for t in prog_str.split()], dtype=np.int64)


def make_fn(prog_ops):
    """Wrap a straight-line opcode array as f(a_batch, b_batch) -> out_batch (torch)."""
    p = torch.tensor(prog_ops[None], device=DEV)

    def f(a, b):
        n = len(a)
        pe = p.expand(n, -1)
        st = oe.run_vm(pe, a.to(DEV), b.to(DEV), S_STACK)
        return st[:, S_STACK - 1].cpu()      # CPU out: callers compare/index on CPU
    return f


# ---------------------------------------------------------------------------
# 1. encoding-equivalence vs the named suite
# ---------------------------------------------------------------------------
def _bitrev16(x):
    r = torch.zeros_like(x)
    for i in range(16):
        r |= ((x >> i) & 1) << (15 - i)
    return r

ENCODINGS = {
    "id": lambda x: x, "not": lambda x: (~x) & MASK,
    "shl1": lambda x: (x << 1) & MASK, "shr1": lambda x: x >> 1,
    "brev": _bitrev16,
}


def encoding_equiv(f, aa, bb):
    """Test f == T_out(g(T_a(a), T_b(b))) over named g and encoding triples (+ swap).
    All tensors CPU (f returns CPU; the named suite is built on CPU inputs here)."""
    out = f(aa, bb)
    hits = []
    for swap in (False, True):
        a_in, b_in = (bb, aa) if swap else (aa, bb)
        for tna, Ta in ENCODINGS.items():
            for tnb, Tb in ENCODINGS.items():
                ga, gb = Ta(a_in), Tb(b_in)
                named = oe._named_suite(ga, gb)
                for gname, gv in named.items():
                    for tno, To in ENCODINGS.items():
                        if bool((To(gv) == out).all()):
                            hits.append(f"{tno}({gname}[{tna}(a'),{tnb}(b')]{' swap' if swap else ''})")
    return hits


# ---------------------------------------------------------------------------
# 2. integer polynomial fit mod 2^16 (deg<=3), exact solve over Z/2^16
# ---------------------------------------------------------------------------
def _solve_mod2k(Amat, y, k=16):
    """Solve A x = y (mod 2^k) by Gaussian elimination preferring odd pivots
    (odd = invertible mod 2^k). Returns x or None."""
    m = 1 << k
    A = [row[:] + [yy] for row, yy in zip(Amat.tolist(), y.tolist())]
    n_rows, n_cols = len(A), len(A[0]) - 1
    piv_rows = []
    r = 0
    for c in range(n_cols):
        sel = None
        for i in range(r, n_rows):
            if A[i][c] % 2 == 1:
                sel = i; break
        if sel is None:
            continue
        A[r], A[sel] = A[sel], A[r]
        inv = pow(A[r][c], -1, m)
        A[r] = [(v * inv) % m for v in A[r]]
        for i in range(n_rows):
            if i != r and A[i][c] % m:
                fac = A[i][c] % m
                A[i] = [(vi - fac * vr) % m for vi, vr in zip(A[i], A[r])]
        piv_rows.append((r, c)); r += 1
        if r == n_rows:
            break
    x = [0] * n_cols
    for rr, cc in piv_rows:
        x[cc] = A[rr][-1] % m
    # check consistency on remaining rows
    for i in range(n_rows):
        s = sum(A[i][c] * x[c] for c in range(n_cols)) % m
        if s != A[i][-1] % m:
            return None
    return x


MONOS = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2), (3, 0), (2, 1), (1, 2), (0, 3)]


def poly_fit(f, seed=5):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, MASK + 1, 64); b = rng.integers(0, MASK + 1, 64)
    out = f(torch.tensor(a), torch.tensor(b)).cpu().numpy().astype(object)
    Amat = np.array([[(int(ai) ** i * int(bi) ** j) % (1 << 16) for (i, j) in MONOS]
                     for ai, bi in zip(a, b)], dtype=object)
    x = _solve_mod2k(Amat, out.astype(object))
    if x is None:
        return None
    # verify on held-out
    a2 = rng.integers(0, MASK + 1, 128); b2 = rng.integers(0, MASK + 1, 128)
    out2 = f(torch.tensor(a2), torch.tensor(b2)).cpu().numpy()
    pred = np.array([sum(x[k] * (int(ai) ** i * int(bi) ** j) for k, (i, j) in enumerate(MONOS)) % (1 << 16)
                     for ai, bi in zip(a2, b2)])
    if (pred == out2).all():
        terms = [f"{x[k]}*a^{i}b^{j}" for k, (i, j) in enumerate(MONOS) if x[k]]
        return " + ".join(terms) if terms else "0"
    return None


# ---------------------------------------------------------------------------
# 3. ANF degree profile on a random 16-bit subcube (8 bits of a x 8 of b)
# ---------------------------------------------------------------------------
def anf_profile(f, seed=7):
    rng = np.random.default_rng(seed)
    abits = rng.choice(16, 8, replace=False)
    bbits = rng.choice(16, 8, replace=False)
    base_a = int(rng.integers(0, MASK + 1)) & ~int(sum(1 << int(i) for i in abits)) & MASK
    base_b = int(rng.integers(0, MASK + 1)) & ~int(sum(1 << int(i) for i in bbits)) & MASK
    n_var = 16
    idx = np.arange(1 << n_var, dtype=np.int64)
    aa = np.full(len(idx), base_a, dtype=np.int64)
    bb = np.full(len(idx), base_b, dtype=np.int64)
    for k in range(8):
        aa |= ((idx >> k) & 1) << int(abits[k])
    for k in range(8):
        bb |= ((idx >> (8 + k)) & 1) << int(bbits[k])
    out = f(torch.tensor(aa), torch.tensor(bb)).cpu().numpy()
    degs, dens = [], []
    popc = np.zeros(1 << n_var, dtype=np.int8)
    for k in range(n_var):
        popc[(idx >> k) & 1 == 1] += 1
    for bit in range(16):
        tt = ((out >> bit) & 1).astype(np.uint8)
        anf = tt.copy()
        for k in range(n_var):           # fast Mobius transform
            step = 1 << k
            mask_hi = (idx & step) == step
            anf[mask_hi] ^= anf[idx[mask_hi] ^ step]
        nz = anf != 0
        degs.append(int(popc[nz].max()) if nz.any() else -1)
        dens.append(float(nz.mean()))
    return dict(deg_per_bit=degs, max_deg=max(degs), mean_density=float(np.mean(dens)))


# ---------------------------------------------------------------------------
# 4-6. BLR, equivariance, bit-dependency
# ---------------------------------------------------------------------------
def blr_linearity(f, n=4000, seed=11):
    rng = np.random.default_rng(seed)
    ax, bx = rng.integers(0, MASK + 1, (2, n)); ay, by = rng.integers(0, MASK + 1, (2, n))
    f0 = f(torch.tensor([0]), torch.tensor([0]))[0]
    fx = f(torch.tensor(ax), torch.tensor(bx)); fy = f(torch.tensor(ay), torch.tensor(by))
    fxy = f(torch.tensor(ax ^ ay), torch.tensor(bx ^ by))
    ok = (fxy == (fx ^ fy ^ f0)).float().mean()
    return float(ok)


def equivariance(f, n=2000, seed=13):
    rng = np.random.default_rng(seed)
    a = torch.tensor(rng.integers(0, MASK + 1, n)); b = torch.tensor(rng.integers(0, MASK + 1, n))
    c = torch.tensor(rng.integers(0, MASK + 1, n))
    base = f(a, b)
    shift = float((f((a << 1) & MASK, (b << 1) & MASK) == ((base << 1) & MASK)).float().mean())
    xort = float((f(a ^ c, b ^ c) == (base ^ c)).float().mean())
    sym = float((f(b, a) == base).float().mean())
    return dict(shift_equiv=shift, xor_translate=xort, arg_symmetric=sym)


def bit_dependency(f, n=400, seed=17):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, MASK + 1, n); b = rng.integers(0, MASK + 1, n)
    base = f(torch.tensor(a), torch.tensor(b)).cpu().numpy()
    dep = np.zeros((32, 16))
    for i in range(16):
        fa = f(torch.tensor(a ^ (1 << i)), torch.tensor(b)).cpu().numpy()
        fb = f(torch.tensor(a), torch.tensor(b ^ (1 << i))).cpu().numpy()
        for o in range(16):
            dep[i, o] = ((fa ^ base) >> o & 1).mean()
            dep[16 + i, o] = ((fb ^ base) >> o & 1).mean()
    up = sum(dep[i % 16, o] > 0.01 for i in range(32) for o in range(16) if o >= i % 16)
    down = sum(dep[i % 16, o] > 0.01 for i in range(32) for o in range(16) if o < i % 16)
    return dict(n_deps=int((dep > 0.01).sum()), upward=int(up), downward=int(down),
                triangularity=round(float(up / max(1, up + down)), 3))


def telescope(name, prog_ops, full=True):
    f = make_fn(prog_ops)
    rng = np.random.default_rng(3)
    aa = torch.tensor(np.concatenate([[3, 100, 255, 1000, 40000, 12345, 7, 65535],
                                      rng.integers(0, MASK + 1, 56)]))
    bb = torch.tensor(np.concatenate([[5, 7, 200, 999, 25000, 11111, 3, 1],
                                      rng.integers(0, MASK + 1, 56)]))
    rep = dict(name=name)
    rep["encoding_equiv"] = encoding_equiv(f, aa, bb)[:6]
    rep["poly_mod_2_16"] = poly_fit(f)
    rep["anf"] = anf_profile(f)
    rep["blr_linear"] = blr_linearity(f)
    rep["equivariance"] = equivariance(f)
    rep["bit_dep"] = bit_dependency(f)
    return rep


def fmt(rep):
    a = rep["anf"]
    eq = rep["equivariance"]; bd = rep["bit_dep"]
    lines = [f"== {rep['name']}",
             f"  encoding-equiv : {rep['encoding_equiv'] or 'NONE (not a named op in light disguise)'}",
             f"  poly mod 2^16  : {rep['poly_mod_2_16'] or 'NONE (not an arithmetic polynomial, deg<=3)'}",
             f"  ANF            : max_deg={a['max_deg']} deg/bit={a['deg_per_bit']} density={a['mean_density']:.3f}",
             f"  BLR linearity  : {rep['blr_linear']:.3f}  (1.0 = GF(2)-linear)",
             f"  equivariance   : shift={eq['shift_equiv']:.2f} xor-translate={eq['xor_translate']:.2f} symmetric={eq['arg_symmetric']:.2f}",
             f"  bit-dependency : {bd['n_deps']}/512 edges, triangularity={bd['triangularity']} (1.0 = pure carry-like upward flow)"]
    return "\n".join(lines)


def selftest():
    print("=== telescope controls (must classify exactly) ===")
    # the oe output convention reads stack slot S-1 (=11): evolved organisms fill the
    # stack; controls must too — pad with dup so the result lands at slot 11.
    pad = " dup" * 11
    add_prog = parse_prog("a b add" + pad, oe.OPN)
    xor_prog = parse_prog("a b xor" + pad, oe.OPN)
    r1 = telescope("CONTROL a+b", add_prog)
    r2 = telescope("CONTROL a^b", xor_prog)
    print(fmt(r1)); print(fmt(r2))
    ok1 = (r1["poly_mod_2_16"] is not None and "1*a^1b^0" in r1["poly_mod_2_16"]
           and "1*a^0b^1" in r1["poly_mod_2_16"])
    ok2 = (r2["blr_linear"] == 1.0 and r2["anf"]["max_deg"] == 1)
    ok3 = any("a+b" in h for h in r1["encoding_equiv"]) and any("a^b" in h for h in r2["encoding_equiv"])
    print(f"  controls: a+b poly={ok1}, xor linear/deg1={ok2}, both named={ok3}")
    return ok1 and ok2 and ok3


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--targets", type=str, default="")
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(0 if selftest() else 1)
    if args.targets == "oe_loop":
        if not selftest():
            raise SystemExit("controls failed — telescope void")
        print()
        targets = []
        for nm, path, opn, maxit in [
            ("oe_fix_s7 top", "runs_pod/closures/oe_fix_s7/oe_log.json", oe.OPN, 0),
            ("oe_fix_s1 top", "runs_pod/closures/oe_fix_s1/oe_log.json", oe.OPN, 0),
            ("oe_fix_s3 top", "runs_pod/closures/oe_fix_s3/oe_log.json", oe.OPN, 0),
            ("loop_m1_s1 top", "runs_pod/phase2/loop_m1_s1/loop_log.json", loop.OPN_L, 1),
            ("loop_m16_s1 top", "runs_pod/phase2/loop_m16_s1/loop_log.json", loop.OPN_L, 16),
        ]:
            log = json.load(open(path))
            prog_str = log[-1]["top"][0]["prog"]
            ops = parse_prog(prog_str, opn)
            if maxit:
                exp, _ = loop.expand_program(ops[None, :], maxit)
                ops = exp[0].cpu().numpy()
            targets.append((nm, ops))
        reports = []
        for nm, ops in targets:
            r = telescope(nm, ops)
            reports.append(r)
            print(fmt(r)); print()
        os.makedirs("runs/telescope", exist_ok=True)
        json.dump(reports, open("runs/telescope/reports.json", "w"), indent=1)
        print("wrote runs/telescope/reports.json")
