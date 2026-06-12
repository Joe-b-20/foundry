"""inspect_ca2d.py — characterize the top 2D-CA survivors' space-time (the moonshot inspection: familiar or not?).

Loads a (K,T,H,W) space-time .npy + the census JSON, and for each rule reports: temporal period (oscillator vs
aperiodic), activity level, and an ASCII render of the final frame so I can actually SEE the dynamics. Honest read:
short period / static = boring; aperiodic + localized moving structure = worth a closer look.
Usage: python inspect_ca2d.py runs_pod/exp1b_census
"""
import sys, json, os
import numpy as np

d = sys.argv[1] if len(sys.argv) > 1 else "runs_pod/exp1b_census"
st = np.load(os.path.join(d, "census_top_spacetime.npy"))   # (K,T,H,W)
js = json.load(open(os.path.join(d, "census_totalistic.json")))
K, T, H, W = st.shape
print(f"space-time {st.shape}  (K rules x T steps x {H}x{W})\n")


def period(frames):
    seen = {}
    for t, f in enumerate(frames):
        h = f.tobytes()
        if h in seen:
            return seen[h], t - seen[h]            # (transient, period)
        seen[h] = t
    return None, None


def bs(mask):
    return "".join(str(c) for c in range(9) if (mask >> c) & 1)


for k in range(K):
    r = js[k] if k < len(js) else {}
    frames = st[k]
    trans, per = period(frames)
    dens = frames[-1].mean()
    activity = float((frames[-1] != frames[-2]).mean())          # frame-to-frame change at the end
    tag = f"B{bs(r.get('birth',0))}/S{bs(r.get('survive',0))}" if r else "?"
    pdesc = f"transient {trans}, period {per}" if per else "no cycle within T (aperiodic/long)"
    print(f"=== rank {k}: {tag}  nov={r.get('novelty',0):.3f} gen_bpc={r.get('gen_bpc',0):.3f} ===")
    print(f"    final density {dens:.3f} | end activity {activity:.3f} | {pdesc}")
    # ASCII render of the final frame (coarse if large)
    step = max(1, H // 32)
    img = frames[-1][::step, ::step]
    for row in img:
        print("    " + "".join("#" if v else "." for v in row))
    print()
