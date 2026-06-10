"""
analyze_phase2.py — pull-side analysis + rendering for the phase-2 experiments
(render-before-verdict, RULES amendment). Run after results are pulled to runs_pod/.

  python analyze_phase2.py dstruct  runs_pod/phase2/dstruct_s1 [s2 ...]
  python analyze_phase2.py loop     runs_pod/phase2/loop_m1_s1 ...
  python analyze_phase2.py pcf      runs_pod/phase2/pcf_main
"""
from __future__ import annotations
import sys, os, json, glob
import numpy as np


def render_dstruct(dirs):
    from PIL import Image
    print("\n=== DEPTH-WHILE-STRUCTURED ===")
    print(f"{'dir/cond':38s} {'rt':>8s} {'track':>6s} {'tape':>6s} {'rerun_ok':>8s}")
    for d in dirs:
        res = json.load(open(os.path.join(d, "dstruct_results.json")))
        for cond, r in res.items():
            w = r["winners"][0]
            print(f"{os.path.basename(d)+'/'+cond:38s} {w['runtime']:8d} {w['track']:6.3f} "
                  f"{w['tape_s']:6.3f} {str(w.get('rerun_ok')):>8s}")
        # render the top machine of each condition: head position over time
        for pf in sorted(glob.glob(os.path.join(d, "*_positions.npy"))):
            pos = np.load(pf)
            if len(pos) < 4:
                continue
            T = len(pos); lo, hi = pos.min(), pos.max()
            W = max(2, hi - lo + 1)
            # downsample time to <=2000 rows
            step = max(1, T // 2000)
            rows = pos[::step]
            img = np.full((len(rows), W), 255, np.uint8)
            for t, p in enumerate(rows):
                img[t, p - lo] = 0
            Image.fromarray(img).save(pf.replace("_positions.npy", "_render.png"))
        print(f"  rendered {os.path.basename(d)} head-position traces -> *_render.png")
    print("  READ: compare track scores across conditions; LOOK at the renders — a structured")
    print("  trace (nested sweeps / triangular growth) vs a metronome vs a random walk.")


def analyze_loop(dirs):
    print("\n=== LOOP-SUBSTRATE NAMING-DENSITY vs REACHABLE DEPTH ===")
    print(f"{'dir':28s} {'maxit':>5s} {'final_edge':>10s} {'top_named_sim':>13s} "
          f"{'named_in_top5':>13s} {'n_exact_named_total':>19s}")
    rows = []
    for d in dirs:
        log = json.load(open(os.path.join(d, "loop_log.json")))
        last = log[-1]
        # maxit from dirname loop_m{X}_s{Y}
        base = os.path.basename(d)
        mit = int(base.split("_m")[1].split("_")[0]) if "_m" in base else -1
        fm = last.get("first_match", {})
        rows.append((mit, last["best_edge"], last["top_named_sim"], last["n_named_in_top"], len(fm)))
        print(f"{base:28s} {mit:5d} {last['best_edge']:10.3f} {last['top_named_sim']:13.3f} "
              f"{last['n_named_in_top']:13d} {len(fm):19d}")
    # aggregate by maxit
    print("\n  by reachable-depth (mean over seeds):")
    import collections
    agg = collections.defaultdict(list)
    for mit, edge, sim, nnt, nfm in rows:
        agg[mit].append((edge, sim, nnt, nfm))
    for mit in sorted(agg):
        vs = np.array(agg[mit])
        print(f"    maxit={mit:3d}: edge={vs[:,0].mean():.3f}  top_named_sim={vs[:,1].mean():.3f}  "
              f"named_in_top5={vs[:,2].mean():.2f}  distinct_named_ever={vs[:,3].mean():.1f}")
    print("  READ: top_named_sim DROPPING with maxit -> depth was the gate (named region shallow).")
    print("  FLAT/RISING -> naming-density extends deep; the ceiling is RECOGNITION not depth (Frontier 2).")


def analyze_pcf(dirs):
    print("\n=== IDENTITY HUNT (PCF) ===")
    for d in dirs:
        summ = os.path.join(d, "stage2_summary.json")
        if os.path.exists(summ):
            s = json.load(open(summ))
            print(f"  {d}: {s['n_hits']} hits | control_ok={s['control_ok']}")
            print(f"    by constant: {s['by_constant']}")
            th = s.get("tail_hits", [])
            print(f"    TAIL hits (catalan/zeta3/gamma): {len(th)}")
            for r in th[:40]:
                print(f"      {r['constant']:8s} A={r['A']} B={r['B']} v={r['value'][:26]} "
                      f"rel={r['relation']} height={r['height']} verified={r['verified']}")
        st1 = os.path.join(d, "stage1_survivors.npz")
        if os.path.exists(st1):
            z = np.load(st1)
            print(f"    stage1: {len(z['V']):,} distinct convergent non-trivial PCF values "
                  f"(crange={int(z['meta'][0])} terms={int(z['meta'][1])})")
    print("  READ: any verified TAIL hit (Catalan/zeta3/gamma) that is NOT a known PCF is a")
    print("  candidate — reported as 'true to N digits + not in references checked', never 'novel'.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__); raise SystemExit(1)
    kind, dirs = sys.argv[1], sys.argv[2:]
    {"dstruct": render_dstruct, "loop": analyze_loop, "pcf": analyze_pcf}[kind](dirs)
