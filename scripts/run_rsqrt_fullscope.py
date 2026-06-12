"""Settle the 0x5F375A87 flag bit-exactly on the FULL domain: exhaustive
max relative error over ALL positive normal float32 (exponents -126..127,
2,130,706,432 values) for the standard trick + one Newton step (3/2, 1/2).

Constants tested: Quake 0x5F3759DF; Lomont 0x5F375A86 and neighbors
0x5F375A85 / 0x5F375A87 / 0x5F375A88; our sample-stage 0x5F3759BB.

METRIC NOTE (claim discipline): our metric is max relative error vs the
FLOAT64 reference 1/sqrt(x). Lomont's 2003 paper compares against the
float32 value (float)(1.0/sqrt(x)) over "all floating point values" and
reports 0x5F375A86 best at 0.175124% — the two metrics differ at the ~6th
significant digit, so results are stated side by side; neither supersedes
the other. Whatever ranks best here is a fact UNDER OUR METRIC ONLY.

Run from repo root:  python3 -m scripts.run_rsqrt_fullscope
"""

import json
import sys
import time
from pathlib import Path

from domains.rsqrt import RsqrtPack
from domains.rsqrt_shelf import trick_newton
from engine.molds_float import FloatProgMold
from engine.recorder import Recorder

CONSTANTS = [
    ("quake", 0x5F3759DF),
    ("lomont-1", 0x5F375A85),
    ("lomont", 0x5F375A86),
    ("ours", 0x5F375A87),
    ("lomont+2", 0x5F375A88),
    ("ours-sample-stage", 0x5F3759BB),
]


def main():
    t0 = time.time()
    run_id = f"rsqrt-fullscope-{int(t0)}"
    rec = Recorder(Path("runs") / run_id, run_id)
    pack = RsqrtPack(lo_exp=-126, hi_exp=128)
    mold = FloatProgMold()
    rec.event("foreman", "runspec",
              payload={"constants": {n: hex(c) for n, c in CONSTANTS},
                       "scope": "all positive normal float32",
                       "values": pack.scope_size,
                       "metric": "max rel err vs float64 reference"},
              reason="predeclaration; metric differs from Lomont's "
                     "float32-reference at the ~6th digit — side-by-side "
                     "statement only, no supersession claims")
    rows = []
    for name, c0 in CONSTANTS:
        t1 = time.time()
        ok, det = pack.verify_trusted(mold, trick_newton(c0))
        assert ok, det
        rows.append({"name": name, "c0": hex(c0),
                     "E": det["max_rel_err"],
                     "worst_at": det["worst_at_bits"],
                     "seconds": round(time.time() - t1, 1)})
        rec.event("judge", "exhaustive", payload=rows[-1])
        print(f"{name} {hex(c0)}: E={det['max_rel_err']:.9e} "
              f"({rows[-1]['seconds']}s)")
    rows.sort(key=lambda r: r["E"])
    report = {"run_id": run_id, "seconds": round(time.time() - t0, 1),
              "ranking": rows, "winner": rows[0],
              "metric": "max rel err vs float64 ref, all positive normal "
                        "float32 (exhaustive)"}
    rec.event("foreman", "report", payload=report)
    (rec.run_dir / "report.json").write_text(json.dumps(report, indent=2))
    rec.close()
    print(f"\nwinner under OUR metric: {rows[0]['name']} {rows[0]['c0']} "
          f"E={rows[0]['E']:.9e}  ({report['seconds']}s total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
