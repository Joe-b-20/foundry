"""
fsm_extract.py — general finite-state-transducer extraction from a trained
NeuralMealy by CLUSTERING the continuous state (k-means), not just sign bits.
Needed once algorithms have >2 states (e.g. single-digit multiplication's
multiplicative carry has 9 states). Picks the smallest #clusters k whose extracted
FSM length-generalizes exactly.

An FSM here = (start_cluster, out_table[(cluster,a,b)], next_table[(cluster,a,b)]).
Built by: cluster visited states -> for each cluster centroid, re-run the cell on
every (a,b) digit pair to get a canonical output digit and next state, assign the
next continuous state to its nearest centroid. Deterministic and verifiable.
"""
from __future__ import annotations
import numpy as np
import torch


@torch.no_grad()
def collect_states(model, a_oh, b_oh, device):
    """Run model over probe sequences; return all visited state vectors (incl start)."""
    model.eval()
    N, L, B = a_oh.shape
    s = model.s0.unsqueeze(0).expand(N, -1).to(device)
    seen = [s.clone()]
    for t in range(L):
        x = torch.cat([a_oh[:, t], b_oh[:, t]], dim=-1)
        _, s = model.step(s, x)
        seen.append(s.clone())
    S = torch.cat(seen, dim=0).cpu().numpy()
    return S


@torch.no_grad()
def collect_transitions(model, a_oh, b_oh, device):
    """Run model; return per-step transitions actually taken:
    s_prev (M,d), da (M,), db (M,), out (M,), s_next (M,d), and start state s0 (d,)."""
    model.eval()
    N, L, B = a_oh.shape
    s = model.s0.unsqueeze(0).expand(N, -1).to(device)
    sp, das, dbs, outs, sn = [], [], [], [], []
    for t in range(L):
        xa, xb = a_oh[:, t], b_oh[:, t]
        x = torch.cat([xa, xb], dim=-1)
        logits, s2 = model.step(s, x)
        sp.append(s.cpu().numpy()); sn.append(s2.cpu().numpy())
        das.append(xa.argmax(-1).cpu().numpy()); dbs.append(xb.argmax(-1).cpu().numpy())
        outs.append(logits.argmax(-1).cpu().numpy())
        s = s2
    import numpy as np
    return (np.concatenate(sp), np.concatenate(das), np.concatenate(dbs),
            np.concatenate(outs), np.concatenate(sn),
            model.s0.detach().cpu().numpy())


def build_fsm_empirical(model, base, centroids, transitions, device, ndigits_fn):
    """Faithful extractor: label states by nearest centroid, then read out/next tables
    by MAJORITY VOTE over the net's actually-observed transitions. Falls back to
    centroid-replay for (state,a,b) triples never observed, so the FSM stays total."""
    from collections import Counter, defaultdict
    sp, da, db, out, sn, s0 = transitions
    C = centroids
    k = C.shape[0]

    def label(S):  # nearest-centroid labels for rows of S (N,d)
        D = ((S[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        return D.argmin(1)
    lp, ln = label(sp), label(sn)
    out_votes = defaultdict(Counter); next_votes = defaultdict(Counter)
    for i in range(len(sp)):
        out_votes[(int(lp[i]), int(da[i]), int(db[i]))][int(out[i])] += 1
        next_votes[(int(lp[i]), int(da[i]), int(db[i]))][int(ln[i])] += 1
    out_table, next_table = {}, {}
    for key, cnt in out_votes.items():
        out_table[key] = cnt.most_common(1)[0][0]
    for key, cnt in next_votes.items():
        next_table[key] = cnt.most_common(1)[0][0]
    # centroid-replay fallback for unseen (state,a,b)
    Ct = torch.tensor(C, dtype=torch.float32, device=device)
    s0_lab = int(((Ct - torch.tensor(s0, device=device)) ** 2).sum(-1).argmin().item())
    import core_data as cd

    @torch.no_grad()
    def replay(st, a_, b_):
        c = Ct[st:st + 1]
        x = torch.zeros(1, 2 * base, device=device); x[0, a_] = 1.0; x[0, base + b_] = 1.0
        logits, s_next = model.step(c, x)
        d2 = ((Ct - s_next) ** 2).sum(-1)
        return int(logits.argmax(-1).item()), int(d2.argmin().item())

    def fsm_predict(a, b, b_is_broadcast=False):
        wa, wb = ndigits_fn(a, base), ndigits_fn(b, base)
        L = wa + wb + 2
        ad = cd.to_digits(a, L, base)
        bd = [b % base] * L if b_is_broadcast else cd.to_digits(b, L, base)
        st = s0_lab; digits = []
        for t in range(L):
            key = (st, ad[t], bd[t])
            if key in out_table:
                o = out_table[key]; nx = next_table[key]
            else:
                o, nx = replay(*key)
            digits.append(o); st = nx
        return cd.from_digits(digits, base)

    return out_table, next_table, s0_lab, fsm_predict


def kmeans(X, k, iters=100, seed=0):
    rng = np.random.RandomState(seed)
    # k-means++ init
    idx = [rng.randint(len(X))]
    for _ in range(k - 1):
        d2 = np.min(((X[:, None, :] - X[np.array(idx)][None, :, :]) ** 2).sum(-1), axis=1)
        probs = d2 / (d2.sum() + 1e-12)
        idx.append(rng.choice(len(X), p=probs))
    C = X[np.array(idx)].copy()
    for _ in range(iters):
        D = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        lab = D.argmin(1)
        newC = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j]
                         for j in range(k)])
        if np.allclose(newC, C):
            break
        C = newC
    return C


@torch.no_grad()
def build_fsm(model, base, centroids, device, ndigits_fn):
    """centroids: (k, d) array. Build canonical tables by re-running the cell from
    each centroid on all (a,b) digit pairs. next-state := nearest centroid."""
    model.eval()
    C = torch.tensor(centroids, dtype=torch.float32, device=device)
    k = C.shape[0]
    out_table, next_table = {}, {}
    for st in range(k):
        c = C[st:st + 1]
        for da in range(base):
            for db in range(base):
                x = torch.zeros(1, 2 * base, device=device)
                x[0, da] = 1.0; x[0, base + db] = 1.0
                logits, s_next = model.step(c, x)
                out_table[(st, da, db)] = int(logits.argmax(-1).item())
                d2 = ((C - s_next) ** 2).sum(-1)
                next_table[(st, da, db)] = int(d2.argmin().item())
    # start cluster = nearest centroid to s0
    s0 = model.s0.detach().to(device)
    start = int(((C - s0) ** 2).sum(-1).argmin().item())

    def fsm_predict(a, b, b_is_broadcast=False):
        """b_is_broadcast=True means b is a single digit fed at every position
        (single-digit multiplier mode)."""
        import core_data as cd
        wa, wb = ndigits_fn(a, base), ndigits_fn(b, base)
        L = wa + wb + 2   # upper bound on result length for add/sub/single-digit mul
        ad = cd.to_digits(a, L, base)
        if b_is_broadcast:
            bd = [b % base] * L
        else:
            bd = cd.to_digits(b, L, base)
        st = start
        digits = []
        for t in range(L):
            key = (st, ad[t], bd[t])
            digits.append(out_table[key])
            st = next_table[key]
        return cd.from_digits(digits, base)

    return out_table, next_table, start, fsm_predict
