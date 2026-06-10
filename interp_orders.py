"""
interp_orders.py — deep, parametrized interpretability of unified 4-op models trained under
DIFFERENT curriculum orders. Runs the SAME battery on every checkpoint and aggregates, so we
can ask: does training ORDER change (a) accuracy, (b) which dims host the algorithm
(scratchpad), (c) the GEOMETRY (axis-aligned vs distributed), (d) the MECHANISM (causal
carry/borrow, mult b-entanglement, op-code selector)?

Pre-registered (see expH_orders.py / TRACKER):
  H1 order->geometry: whichever of {add,sub} is introduced FIRST gets the clean AXIS-ALIGNED
     code; the later one is DISTRIBUTED (rotated) in the shared scratchpad.
     => predict carry-axis for {addfirst, mulfirst}, borrow-axis for {subfirst, reverse}.
  H2 mechanism invariance: carry/borrow are causal; mult-carry is b-entangled; op=selector —
     regardless of order. Only geometry changes.
  H3 accuracy: all orders ~equal final acc and SAME walls (coprime div).

Methods reused from interp_unified*.py: trace, true_latent, collect (s0 excluded),
best_single_dim, linear_probe, causal injection on the DISCOVERED scratchpad.

Run: bash run.sh interp_orders.py            # all checkpoints
     bash run.sh interp_orders.py <ckpt> <tag>
"""
import sys
import random
import numpy as np
import torch
import core_data as cd
import expA_mealy as E
import expF_unified as F

base = 10
DEV = F.DEVICE
OPS = F.OPS
torch.manual_seed(0)


def load(ckpt):
    m = F.UnifiedMealy(base=base, state_dim=8, hidden=96)
    m.load_state_dict(torch.load(ckpt, map_location=DEV)); m.to(DEV); m.eval()
    return m


# ----------------------------------------------------------------------------- tracing
def op_input_seq(op, a, b):
    if op in ("add", "sub"):
        w = max(E._ndigits(a, base), E._ndigits(b, base)); L = w + 1
        ad = cd.to_digits(a, L, base); bd = cd.to_digits(b, L, base)
        return [(ad[t], bd[t]) for t in range(L)]
    elif op == "mul":
        w = E._ndigits(a, base); L = w + 1; ad = cd.to_digits(a, L, base)
        return [(ad[t], b % base) for t in range(L)]
    else:
        L = E._ndigits(a, base); am = F.digits_msb(a, L, base)
        return [(am[t], b % base) for t in range(L)]


@torch.no_grad()
def trace(m, op, a, b):
    seq = op_input_seq(op, a, b); op_idx = OPS.index(op); s = m.s0.unsqueeze(0).to(DEV)
    states = [s.squeeze(0).cpu().numpy().copy()]
    for (da, db) in seq:
        x = torch.zeros(1, 2 * base + F.NOP, device=DEV)
        x[0, da] = 1.0; x[0, base + db] = 1.0; x[0, 2 * base + op_idx] = 1.0
        _, s = m.step(s, x); states.append(s.squeeze(0).cpu().numpy().copy())
    return seq, states


def true_latent(op, seq):
    lat = []
    if op == "add":
        c = 0
        for (da, db) in seq: lat.append(c); c = (da + db + c) // base
    elif op == "sub":
        bw = 0
        for (da, db) in seq: lat.append(bw); bw = 1 if (da - db - bw) < 0 else 0
    elif op == "mul":
        c = 0
        for (da, db) in seq: lat.append(c); c = (da * db + c) // base
    else:
        r = 0
        for (da, db) in seq: lat.append(r); r = (r * base + da) % db
    return lat


def collect(m, op, n=900, widths=(2, 3, 4, 6), seed=0, div_d=None):
    rng = random.Random(seed); S = []; Lat = []
    for _ in range(n):
        w = rng.choice(widths); a = rng.randint(0, base ** w - 1)
        if op in ("add", "sub"):
            b = rng.randint(0, base ** w - 1)
            if op == "sub" and a < b: a, b = b, a
        elif op == "mul":
            b = rng.randint(0, base - 1)
        else:
            b = div_d if div_d else rng.choice([2, 5])
        seq, states = trace(m, op, a, b); lat = true_latent(op, seq)
        for t in range(1, len(seq)):   # skip s0
            S.append(states[t]); Lat.append(lat[t])
    return np.array(S), np.array(Lat)


# ----------------------------------------------------------------------------- probes
def best_single_dim(S, y):
    """Best binary-accuracy single dim with optimal threshold (both polarities)."""
    best = (-1, -1.0)
    for d in range(S.shape[1]):
        x = S[:, d]
        for thr in np.unique(np.quantile(x, np.linspace(0, 1, 41))):
            for sgn in (+1, -1):
                pred = (x > thr).astype(int) if sgn > 0 else (x < thr).astype(int)
                acc = (pred == y).mean()
                if acc > best[1]: best = (d, acc)
    return best  # (dim, acc)


def linear_probe(S, y, steps=500, lr=0.05):
    if S.shape[1] == 0:
        return 1.0 / (int(y.max()) + 1), np.zeros(0)
    Xt = torch.tensor(S, dtype=torch.float32); yt = torch.tensor(y, dtype=torch.long)
    K = int(y.max()) + 1
    W = torch.zeros(S.shape[1], K, requires_grad=True); b = torch.zeros(K, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr); lossf = torch.nn.CrossEntropyLoss()
    for _ in range(steps):
        loss = lossf(Xt @ W + b, yt); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        acc = (Xt @ W + b).argmax(1).eq(yt).float().mean().item()
        imp = W.detach().abs().sum(1).numpy()
    return acc, imp


def chance(y):
    return float(max(np.bincount(y) / len(y)))


# ----------------------------------------------------------------------------- analyses
def discover_scratchpad(m, k=3):
    """Rank the 8 dims by summed (normalized) linear-probe importance across ops;
    the top-k are the shared scratchpad. Returns (scratch, complement, imp_matrix)."""
    rows = {}
    for op, dd in [("add", None), ("sub", None), ("mul", None), ("div", 2), ("div", 5)]:
        S, Lat = collect(m, op, n=800, div_d=dd)
        _, imp = linear_probe(S, Lat)
        imp = imp / (imp.sum() + 1e-9)
        rows[f"{op}{dd or ''}"] = imp
    M = np.stack(list(rows.values()))         # (n_op, 8)
    total = M.sum(0)
    scratch = sorted(np.argsort(-total)[:k].tolist())
    comp = [d for d in range(8) if d not in scratch]
    return scratch, comp, rows, total


def geometry(m):
    """For carry(add) and borrow(sub): best-single-dim vs linear-probe. axis-aligned if
    single>0.95. Returns dict op-> (single, probe, label, best_dim)."""
    out = {}
    for op, name in [("add", "carry"), ("sub", "borrow")]:
        S, Lat = collect(m, op, n=900)
        bd, bs = best_single_dim(S, Lat)
        pr, _ = linear_probe(S, Lat)
        label = "AXIS-ALIGNED" if bs > 0.95 else "DISTRIBUTED"
        out[op] = (bs, pr, label, bd)
    return out


def carry_borrow_angle(m, scratch):
    """Direction overlap between carry and borrow. Δcarry = mean(state|carry=1)-mean(|carry=0),
    Δborrow likewise. |cos|≈0 => orthogonal (separate axes / spatial partition); |cos| large
    => same direction (the later op superimposed on the earlier op's axis)."""
    Sa, La = collect(m, "add", n=1500); Ss, Ls = collect(m, "sub", n=1500)
    dca = Sa[La == 1].mean(0) - Sa[La == 0].mean(0)
    dbo = Ss[Ls == 1].mean(0) - Ss[Ls == 0].mean(0)

    def cos(u, v):
        return float(abs(u @ v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))
    return cos(dca, dbo), cos(dca[scratch], dbo[scratch]), dca, dbo


def shared_vs_partition(m, scratch, comp):
    out = {}
    for op, dd in [("add", None), ("sub", None), ("mul", None), ("div", 2), ("div", 5)]:
        S, Lat = collect(m, op, n=900, div_d=dd)
        full, _ = linear_probe(S, Lat)
        scr, _ = linear_probe(S[:, scratch], Lat)
        cmp_, _ = linear_probe(S[:, comp], Lat)
        out[f"{op}{dd or ''}"] = (full, scr, cmp_, chance(Lat))
    return out


@torch.no_grad()
def _out_from_state(m, state_vec, da, db, op_idx):
    x = torch.zeros(1, 2 * base + F.NOP, device=DEV)
    x[0, da] = 1.0; x[0, base + db] = 1.0; x[0, 2 * base + op_idx] = 1.0
    s = torch.tensor(state_vec, dtype=torch.float32, device=DEV).unsqueeze(0)
    logits, _ = m.step(s, x); return int(logits.argmax(-1))


def _host_dims(c0, c1, k=3):
    """Top-k dims by |centroid difference| — the dims that actually move with the latent.
    Tracks the latent's OWN code regardless of where order placed it (robust for all models)."""
    return sorted(np.argsort(-np.abs(c1 - c0))[:k].tolist())


def causal_carry(m):
    """Inject carry=1 into the latent's OWN host dims. On no-carry pairs (da+db<10) the output
    must flip (a+b)->(a+b+1). Returns (baseline, FULL, host3, comp_of_host3, host3)."""
    S, Lat = collect(m, "add", n=1500)
    c0 = S[Lat == 0].mean(0); c1 = S[Lat == 1].mean(0)
    host = _host_dims(c0, c1); comp = [d for d in range(8) if d not in host]

    def test(dims):
        inj = c0.copy(); inj[dims] = c1[dims]; ok = tot = 0
        for da in range(base):
            for db in range(base):
                if da + db >= base: continue
                tot += 1; ok += (_out_from_state(m, inj, da, db, 0) == (da + db + 1) % base)
        return ok / tot
    base_ok = 0; bt = 0
    for da in range(base):
        for db in range(base):
            if da + db >= base: continue
            bt += 1; base_ok += (_out_from_state(m, c0, da, db, 0) == (da + db) % base)
    return base_ok / bt, test(list(range(8))), test(host), test(comp), host


def causal_borrow(m):
    """Inject borrow=1 into the latent's OWN host dims. On no-borrow pairs (da>=db) output -1."""
    S, Lat = collect(m, "sub", n=1500)
    if (Lat == 1).sum() < 10 or (Lat == 0).sum() < 10:
        return None
    b0 = S[Lat == 0].mean(0); b1 = S[Lat == 1].mean(0)
    host = _host_dims(b0, b1); comp = [d for d in range(8) if d not in host]

    def test(dims):
        inj = b0.copy(); inj[dims] = b1[dims]; ok = tot = 0
        for da in range(base):
            for db in range(base):
                if da < db: continue
                tot += 1; ok += (_out_from_state(m, inj, da, db, 1) == (da - db - 1) % base)
        return ok / tot
    base_ok = 0; bt = 0
    for da in range(base):
        for db in range(base):
            if da < db: continue
            bt += 1; base_ok += (_out_from_state(m, b0, da, db, 1) == (da - db) % base)
    return base_ok / bt, test(list(range(8))), test(host), test(comp), host


@torch.no_grad()
def _trace_mul_states(m, a, b):
    w = E._ndigits(a, base); L = w + 1; ad = cd.to_digits(a, L, base)
    s = m.s0.unsqueeze(0).to(DEV); states = []
    for t in range(L):
        states.append(s.squeeze(0).cpu().numpy().copy())
        x = torch.zeros(1, 2 * base + F.NOP, device=DEV)
        x[0, ad[t]] = 1.0; x[0, base + (b % base)] = 1.0; x[0, 2 * base + 2] = 1.0
        _, s = m.step(s, x)
    return ad, states


def mult_entangle(m):
    """Matched vs mismatched multiplier on REAL carry=k states (interp_unified6 method)."""
    samples = {k: [] for k in range(9)}; rng = random.Random(0)
    for _ in range(5000):
        w = rng.choice([2, 3, 4, 6]); a = rng.randint(0, base ** w - 1); b = rng.randint(0, base - 1)
        ad, states = _trace_mul_states(m, a, b); c = 0
        for t in range(len(ad)):
            if len(samples[c]) < 50: samples[c].append((states[t].copy(), b))
            c = (ad[t] * b + c) // base
    rng2 = random.Random(1); mm = []; mmis = []
    for k in range(9):
        if not samples[k]: continue
        for (sv, b0) in samples[k]:
            okm = sum(_out_from_state(m, sv, da, b0, 2) == (da * b0 + k) % base for da in range(base)) / base
            bmis = rng2.choice([x for x in range(base) if x != b0])
            okx = sum(_out_from_state(m, sv, da, bmis, 2) == (da * bmis + k) % base for da in range(base)) / base
            mm.append(okm); mmis.append(okx)
    return float(np.mean(mm)), float(np.mean(mmis))


def op_selector(m):
    """step-0 output equals op-selected function (div on trained divisors 2/5)."""
    exp_fns = {"add": lambda a, b: (a + b) % base, "sub": lambda a, b: (a - b) % base,
               "mul": lambda a, b: (a * b) % base, "div": lambda a, b: a // b}
    agree = {op: [0, 0] for op in OPS}
    pairs = [(7, 3), (4, 5), (9, 2), (6, 2), (8, 5), (3, 5), (1, 2), (9, 5), (5, 2), (7, 5)]
    for (a, b) in pairs:
        for op in OPS:
            bb = b if op != "div" else (2 if b % 2 == 0 else 5)
            got = _out_from_state(m, m.s0.detach().cpu().numpy(), a, bb, OPS.index(op))
            agree[op][0] += (got == exp_fns[op](a, bb)); agree[op][1] += 1
    return {op: agree[op][0] / agree[op][1] for op in OPS}


def accuracy(m, widths=(1, 4, 12, 20)):
    out = {}
    for op in OPS:
        if op == "div":
            out["div125"] = {w: F.acc_at_width(m, op, base, w, n=400, div_filter={1, 2, 5}) for w in widths}
            out["divcop"] = {w: F.acc_at_width(m, op, base, w, n=400, div_filter={3, 7, 9}) for w in widths}
        else:
            out[op] = {w: F.acc_at_width(m, op, base, w, n=400) for w in widths}
    return out


# ----------------------------------------------------------------------------- driver
def analyze(ckpt, tag):
    m = load(ckpt)
    print("\n" + "=" * 80)
    print(f"MODEL [{tag}]  ({ckpt})")
    print("=" * 80)

    acc = accuracy(m)
    print("  ACCURACY (exact, per width):")
    for k, r in acc.items():
        print(f"    {k:7s}: " + "  ".join(f"w{w}:{r[w]:.3f}" for w in r))

    scratch, comp, imp_rows, total = discover_scratchpad(m)
    print(f"\n  SCRATCHPAD (top-3 dims by summed probe importance) = {scratch}  complement={comp}")
    print("    per-op per-dim probe importance (normalized):")
    print("       dim:   " + "  ".join(f"d{i}" for i in range(8)))
    for name, imp in imp_rows.items():
        print(f"     {name:6s}: " + "  ".join(f"{imp[i]:.2f}" for i in range(8)))
    print(f"     TOTAL : " + "  ".join(f"{total[i]:.2f}" for i in range(8)))

    geo = geometry(m)
    print("\n  GEOMETRY (carry/borrow: single-dim vs linear-probe):")
    for op, (bs, pr, label, bd) in geo.items():
        nm = "carry" if op == "add" else "borrow"
        print(f"    {op} ({nm:6s}): best-single-dim d{bd}={bs:.3f}  linear-probe={pr:.3f}  -> {label}")

    ang_full, ang_scr, _, _ = carry_borrow_angle(m, scratch)
    print(f"\n  CARRY vs BORROW direction overlap |cos|: full {ang_full:.3f} | scratchpad {ang_scr:.3f}")
    print("     (low => orthogonal/spatial-partition; high => superimposed on a shared direction)")

    svp = shared_vs_partition(m, scratch, comp)
    print(f"\n  SHARED-SCRATCHPAD test (decode from scratch{scratch} vs complement{comp}):")
    for name, (full, scr, cmp_, ch) in svp.items():
        print(f"    {name:6s}: full {full:.3f} | scratch {scr:.3f} | complement {cmp_:.3f} | chance {ch:.3f}")

    cc = causal_carry(m)
    print("\n  CAUSAL CARRY injection (output should flip (a+b)->(a+b+1)):")
    print(f"    baseline {cc[0]:.3f} | inject FULL {cc[1]:.3f} | inject HOST{cc[4]} {cc[2]:.3f} | inject COMP {cc[3]:.3f}")
    cb = causal_borrow(m)
    if cb:
        print("  CAUSAL BORROW injection (output should flip (a-b)->(a-b-1)):")
        print(f"    baseline {cb[0]:.3f} | inject FULL {cb[1]:.3f} | inject HOST{cb[4]} {cb[2]:.3f} | inject COMP {cb[3]:.3f}")

    me = mult_entangle(m)
    print(f"\n  MULT-CARRY b-entanglement: matched-b {me[0]:.3f} | mismatched-b {me[1]:.3f}  "
          f"(matched>>mismatched => entangled)")

    sel = op_selector(m)
    print("  OP-CODE selector (step-0 output == selected function): " +
          "  ".join(f"{op}:{sel[op]:.2f}" for op in OPS))

    # one-line machine-readable summary
    summ = {
        "tag": tag,
        "carry_single": round(geo["add"][0], 3), "carry_probe": round(geo["add"][1], 3),
        "carry_label": geo["add"][2], "carry_dim": geo["add"][3],
        "borrow_single": round(geo["sub"][0], 3), "borrow_probe": round(geo["sub"][1], 3),
        "borrow_label": geo["sub"][2], "borrow_dim": geo["sub"][3],
        "scratch": scratch,
        "carry_causal_full": round(cc[1], 3), "carry_causal_host": round(cc[2], 3),
        "carry_causal_comp": round(cc[3], 3),
        "borrow_causal_full": round(cb[1], 3) if cb else None,
        "borrow_causal_host": round(cb[2], 3) if cb else None,
        "mult_matched": round(me[0], 3), "mult_mismatched": round(me[1], 3),
        "div125_w20": round(acc["div125"][20], 3), "divcop_w20": round(acc["divcop"][20], 3),
        "mul_w20": round(acc["mul"][20], 3),
        "cb_cos_full": round(ang_full, 3), "cb_cos_scratch": round(ang_scr, 3),
        "add_w20": round(acc["add"][20], 3), "sub_w20": round(acc["sub"][20], 3),
    }
    print("\nSUMMARY " + str(summ))
    return summ


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        analyze(sys.argv[1], sys.argv[2])
        raise SystemExit(0)
    CKPTS = [
        ("runs/expH_order_addfirst.pt", "addfirst"),
        ("runs/expH_order_subfirst.pt", "subfirst"),
        ("runs/expH_order_mulfirst.pt", "mulfirst"),
        ("runs/expH_order_reverse.pt", "reverse"),
        ("runs/expF_unified_dynamic.pt", "EXIST-dynamic(addfirst)"),
        ("runs/expF_unified_cotrain.pt", "EXIST-cotrain(noorder)"),
    ]
    import os
    summaries = []
    for ckpt, tag in CKPTS:
        if not os.path.exists(ckpt):
            print(f"  [skip {tag}: {ckpt} not found]"); continue
        summaries.append(analyze(ckpt, tag))

    print("\n" + "=" * 80)
    print("CROSS-MODEL SUMMARY — does ORDER change geometry/mechanism/accuracy?")
    print("=" * 80)
    print(f"  {'model':26s} {'first':5s} | carry sgl/prb label        | borrow sgl/prb label       | cb|cos|")
    firstop = {"addfirst": "add", "subfirst": "sub", "mulfirst": "mul", "reverse": "div"}
    for s in summaries:
        fo = firstop.get(s["tag"], "?")
        print(f"  {s['tag']:26s} {fo:5s} | "
              f"{s['carry_single']:.2f}/{s['carry_probe']:.2f} {s['carry_label']:12s} | "
              f"{s['borrow_single']:.2f}/{s['borrow_probe']:.2f} {s['borrow_label']:12s} | {s['cb_cos_full']:.2f}")
    print("\n  H1 (first of add/sub gets the axis): predict carry-axis for addfirst/mulfirst,")
    print("     borrow-axis for subfirst/reverse. Read the labels above.")
    print(f"\n  {'model':26s} | mult matched/mismatch | scratch | div125_w20 | divcop_w20")
    for s in summaries:
        print(f"  {s['tag']:26s} | {s['mult_matched']:.2f}/{s['mult_mismatched']:.2f}            | "
              f"{str(s['scratch']):11s} | {s['div125_w20']:.3f}      | {s['divcop_w20']:.3f}")
    print("\nINTERP_ORDERS DONE")
