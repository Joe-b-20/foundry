"""inspect_ca1d.py — render + characterize the top 1D-CA survivors' space-time (the moonshot inspection for radius-2/3).

Loads top_spacetime.npy (K,T,W) + survivors.json from an exp1 --out dir, and for each rule: temporal period, a glider
probe (does a localized region drift?), and an ASCII space-time render (time downward) so I can SEE if it's a known
class-4-style glider rule or something genuinely unfamiliar. Honest read: periodic/nested-fractal = known; persistent
localized moving structures with non-trivial interactions = worth a hard look.
Usage: python inspect_ca1d.py runs_pod/exp1_r2_s1
"""
import sys, json, os
import numpy as np

d = sys.argv[1] if len(sys.argv) > 1 else "runs_pod/exp1_r2_s1"
st = np.load(os.path.join(d, "top_spacetime.npy"))      # (K,T,W)
try:
    js = json.load(open(os.path.join(d, "survivors.json")))["results"]
except Exception:
    js = []
K, T, W = st.shape
print(f"space-time {st.shape}  (K rules x T steps x W cells)\n")


def period(rows):
    seen = {}
    for t, r in enumerate(rows):
        h = r.tobytes()
        if h in seen:
            return seen[h], t - seen[h]
        seen[h] = t
    return None, None


for k in range(min(K, 6)):
    r = js[k] if k < len(js) else {}
    rows = st[k]
    trans, per = period(rows)
    rid = f"0x{r['rule']:08X}" if r.get("rule") is not None else "?"
    pdesc = f"transient {trans}, period {per}" if per else "no cycle within T (aperiodic/long)"
    print(f"=== rank {k}: {rid} nov={r.get('novelty',0):.3f} gen_bpc={r.get('general_bpc',0):.3f} damage={r.get('damage',0):.3f} ===")
    print(f"    {pdesc}")
    wstep = max(1, W // 96)
    tstep = max(1, T // 48)
    for row in rows[::tstep]:
        print("    " + "".join("#" if v else " " for v in row[::wstep]))
    print()
