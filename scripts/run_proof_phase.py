"""The gauntlet: re-run the ENTIRE proof-it-works phase on the current
engine, end to end, one command. This is also the standing regression
harness — run it after any engine change.

Stages (each must pass):
  1. module sanity checks (every __main__ assertion block)
  2. calibration A — sorting networks n=3..8 (hill-climb + recognizer)
  3. tiny-n optimality certificates by exhaustion (n=2..4)
  4. island ecology on the n=6/n=8 gaps (3 seeds each)
  5. PCF seeded-walk replication (3 seeds, controls every generation)
  6. PCF scoped sweep — 8/(7 zeta3) from outcome (3 seeds, null arms)
  7. Karatsuba rediscovery + naming (3 seeds)
  8. doctor exam C1/C2/C3 (9 universes)
  9. grokking probe (must report 0 would-be kills)
 10. audit probes (claims vs artifacts; no FAIL/ERROR)

Run from repo root:  python3 -m scripts.run_proof_phase
"""

import glob
import json
import subprocess
import sys
import time
from pathlib import Path

SANITY_MODULES = [
    "engine.core_lang", "engine.runner", "engine.molds", "engine.proposers",
    "engine.recorder", "engine.archive", "engine.recognizer",
    "engine.molds_pcf", "engine.numeric", "engine.molds_bilinear",
    "engine.molds_bits", "engine.doctor", "engine.molds_float",
    "engine.remez",
    "domains.sorting_networks", "domains.sorting_networks_shelf",
    "domains.pcf_shelf", "domains.pcf", "domains.bilinear",
    "domains.bilinear_shelf", "domains.bitmixer", "domains.rsqrt",
    "domains.rsqrt_shelf", "domains.tanh", "domains.tanh_shelf",
    "domains.sqrt_log2", "domains.exp2", "domains.sigmoid", "engine.ratfit",
]

STAGES = [
    ("calibration-A", ["scripts.run_calibration_a"]),
    ("certificates", ["scripts.certify_tiny_optimality"]),
    ("islands-n6-n8", ["scripts.run_islands_a"]),
    ("pcf-replication", ["scripts.run_pcf_replication", "0"],
     ["scripts.run_pcf_replication", "1"], ["scripts.run_pcf_replication", "2"]),
    ("pcf-sweep", ["scripts.run_pcf_sweep", "0"],
     ["scripts.run_pcf_sweep", "1"], ["scripts.run_pcf_sweep", "2"]),
    ("karatsuba", ["scripts.run_karatsuba"]),
    ("doctor-exam", ["scripts.run_doctor_exam"]),
    ("grokking-probe", ["scripts.run_grokking_probe"]),
    ("tanh-calibration", ["scripts.run_tanh_calibration"]),
    ("audit", ["scripts.run_audit_checks"]),
]


def run_mod(args):
    t = time.time()
    p = subprocess.run([sys.executable, "-m", *args],
                       capture_output=True, text=True)
    return {"cmd": " ".join(args), "rc": p.returncode,
            "seconds": round(time.time() - t, 2),
            "tail": p.stdout.strip().splitlines()[-3:]}


def main():
    t0 = time.time()
    rows = []
    ok_all = True

    sane = [run_mod([m]) for m in SANITY_MODULES]
    n_bad = sum(1 for r in sane if r["rc"] != 0)
    rows.append({"stage": "sanities", "ok": n_bad == 0,
                 "detail": f"{len(sane) - n_bad}/{len(sane)} modules pass",
                 "seconds": round(sum(r["seconds"] for r in sane), 2),
                 "runs": [r for r in sane if r["rc"] != 0]})
    ok_all &= n_bad == 0
    print(f"sanities: {len(sane) - n_bad}/{len(sane)}")

    for stage in STAGES:
        name, cmds = stage[0], stage[1:]
        results = [run_mod(c) for c in cmds]
        ok = all(r["rc"] == 0 for r in results)
        if name == "grokking-probe" and ok:
            rep = sorted(glob.glob("runs/grokking-probe-*/report.json"))[-1]
            killed = json.loads(Path(rep).read_text())["killed_before_find"]
            ok = killed == 0
            results.append({"cmd": "killed_before_find check",
                            "rc": 0 if ok else 1, "seconds": 0,
                            "tail": [f"killed={killed}"]})
        rows.append({"stage": name, "ok": ok,
                     "seconds": round(sum(r["seconds"] for r in results), 2),
                     "runs": results})
        ok_all &= ok
        print(f"{name}: {'PASS' if ok else 'FAIL'} "
              f"({rows[-1]['seconds']}s)")
        if not ok:
            for r in results:
                if r["rc"] != 0:
                    print(f"   FAILED: {r['cmd']}\n   " + "\n   ".join(r["tail"]))

    out_dir = Path("runs") / f"proof_phase-{int(t0)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"PASS": ok_all, "seconds": round(time.time() - t0, 2),
               "stages": rows}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nPROOF PHASE {'PASS' if ok_all else 'FAIL'} "
          f"in {summary['seconds']}s -> {out_dir}/summary.json")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
