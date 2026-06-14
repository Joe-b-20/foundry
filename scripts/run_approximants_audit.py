"""Approximants audit — reproduce every certified numerical-approximant
result under the CURRENT engine and reconcile each against the claim I
committed (number, scope, metric, provenance, certificate).

Method: re-run each DETERMINISTIC hunt as a subprocess (so the candidate
is rebuilt by the real code — no hand reconstruction to mistype), read the
fresh report.json, extract the certified number, and compare to the
hardcoded CLAIM ledger below. A mismatch beyond tolerance = stale claim or
engine drift = FAIL. This is the "claim still matches certificate/scope/
metric/provenance after the latest engine changes" check Joe asked for.

The CLAIM ledger is the single source of truth for what the docs assert;
the audit doc (docs/audit_2026-06-13.md) reconciles README/TRACKER/memory
prose against THIS ledger.

Run from repo root:  python3 -m scripts.run_approximants_audit
  (optional: pass "fast" to skip the 187s full-domain rsqrt re-verify)
"""

import glob
import json
import subprocess
import sys
import time
from pathlib import Path

# name -> claim. err = claimed exhaustive error; rtol = allowed relative
# drift; scope/metric/provenance/certificate are reconciled in the doc.
CLAIMS = {
    "rsqrt-A87-fulldomain": dict(
        err=1.751287782e-3, rtol=1e-4,
        scope="all positive normal float32 [2^-126,2^128)", metric="rel",
        provenance="structure + derived Newton (3/2,1/2) GIVEN; magic from outcome",
        certificate="L1-exhaustive", hunt=["scripts.run_rsqrt_fullscope"],
        report="rsqrt-fullscope", extract=("ranking", "ours")),
    "rsqrt-armB-moroz": dict(
        err=8.787168e-4, rtol=1e-3,
        scope="all float32 [2^-8,2^8)", metric="rel",
        provenance="magic + k1 + k2 ALL from outcome (coupled optimizer)",
        certificate="L1-exhaustive", hunt=["scripts.run_rsqrt_armB"],
        report="rsqrt-armB", extract=("key", "exhaustive_max_rel")),
    "log2-L3": dict(
        err=4.3043e-2, rtol=2e-3,
        scope="all float32 [2^-8,2^8)", metric="abs",
        provenance="trick structure (Blinn) given; slope+offset from outcome",
        certificate="L1-exhaustive", hunt=["scripts.run_explog_hunt"],
        report="explog-hunt", extract=("rows", "fn", "log2", "E")),
    "sqrt-via-rsqrt": dict(
        err=1.7513e-3, rtol=2e-3,
        scope="all float32 [2^-8,2^8)", metric="rel",
        provenance="composition x*rsqrt_A87(x) (uses the engine's own artifact)",
        certificate="L1-exhaustive", hunt=[], report="explog-hunt",
        extract=("rows", "arm", "via-rsqrt-A87", "E")),
    "exp2-schraudolph": dict(
        err=2.9827e-2, rtol=2e-3,
        scope="signed float32 |x| in [2^-8,2^3)", metric="rel",
        provenance="slope STRUCTURALLY FIXED 2^23; bias from outcome",
        certificate="L1-exhaustive", hunt=["scripts.run_exp2_hunt"],
        report="exp2-hunt", extract=("result", "exhaustive_max_rel")),
    "sigmoid-2_2": dict(
        err=3.0817e-2, rtol=2e-3, scope="signed |x| in [2^-4,2^3)", metric="abs",
        provenance="rational coeffs from outcome (linearized IRLS)",
        certificate="L1-exhaustive", hunt=["scripts.run_rational_hunt", "sigmoid"],
        report="sigmoid-rational", extract=("rows", "rational", "[2/2]",
                                            "exhaustive_max_err")),
    "sigmoid-3_3": dict(
        err=2.6522e-3, rtol=2e-3, scope="signed |x| in [2^-4,2^3)", metric="abs",
        provenance="rational coeffs from outcome (linearized IRLS)",
        certificate="L1-exhaustive", hunt=[], report="sigmoid-rational",
        extract=("rows", "rational", "[3/3]", "exhaustive_max_err")),
    "tanh-2_2": dict(
        err=1.6959e-2, rtol=2e-3, scope="[2^-2,2^3)", metric="rel",
        provenance="rational coeffs from outcome", certificate="L1-exhaustive",
        hunt=["scripts.run_rational_hunt", "tanh"], report="tanh-rational",
        extract=("rows", "rational", "[2/2]", "exhaustive_max_err")),
    "tanh-3_3": dict(
        err=1.0066e-3, rtol=2e-3, scope="[2^-2,2^3)", metric="rel",
        provenance="rational coeffs from outcome", certificate="L1-exhaustive",
        hunt=[], report="tanh-rational",
        extract=("rows", "rational", "[3/3]", "exhaustive_max_err")),
    "erf-2_2": dict(
        err=1.8437e-1, rtol=2e-3, scope="signed |x| in [2^-4,2^3)", metric="abs",
        provenance="rational coeffs from outcome", certificate="L1-exhaustive",
        hunt=["scripts.run_rational_hunt", "erf"], report="erf-rational",
        extract=("rows", "rational", "[2/2]", "exhaustive_max_err")),
    "erf-3_3": dict(
        err=3.3812e-2, rtol=2e-3, scope="signed |x| in [2^-4,2^3)", metric="abs",
        provenance="rational coeffs from outcome", certificate="L1-exhaustive",
        hunt=[], report="erf-rational",
        extract=("rows", "rational", "[3/3]", "exhaustive_max_err")),
    "gelu-x3_3": dict(
        err=6.0295e-2, rtol=2e-3, scope="signed |x| in [2^-4,2^3)", metric="abs",
        provenance="x * rational(Phi); rational coeffs from outcome (|x|-weighted)",
        certificate="L1-exhaustive", hunt=["scripts.run_gelu_hunt"],
        report="gelu-hunt", extract=("rows", "gelu_rational", "x*[3/3]",
                                     "exhaustive_max_abs")),
    "sin": dict(
        err=5.1240e-4, rtol=2e-3, scope="signed |x| in [2^-6,2^4)", metric="abs",
        provenance="argument reduction (math consts given); odd poly from outcome",
        certificate="L1-exhaustive", hunt=["scripts.run_trig_hunt"],
        report="trig-hunt", extract=("rows", "fn", "sin", "exhaustive_max_abs")),
    "cos": dict(
        err=5.1272e-4, rtol=2e-3, scope="signed |x| in [2^-6,2^4)", metric="abs",
        provenance="cos=sin(x+pi/2); same reduction+poly", certificate="L1-exhaustive",
        hunt=[], report="trig-hunt", extract=("rows", "fn", "cos",
                                              "exhaustive_max_abs")),
}


def newest_report(prefix):
    fs = sorted(glob.glob("runs/" + prefix + "*/report.json"),
                key=lambda p: int(p.split("-")[-1].split("/")[0])
                if p.split("-")[-1].split("/")[0].isdigit() else 0)
    return fs[-1] if fs else None


def extract(report, spec):
    if spec[0] == "key":
        return report[spec[1]]
    if spec[0] == "result":
        return report["result"][spec[1]]
    if spec[0] == "ranking":          # fullscope: ranking list of dicts
        for r in report["ranking"]:
            if r["name"] == spec[1]:
                return r["E"]
        return None
    if spec[0] == "rows":             # rows; match field spec[1]==spec[2]
        _, field, val, numkey = spec
        for r in report["rows"]:
            if r.get(field) == val:
                return r[numkey]
        return None
    raise ValueError(spec)


def main():
    fast = "fast" in sys.argv
    t0 = time.time()
    rows = []
    # run each hunt once (those with a hunt cmd); others read same report
    ran = set()
    for name, c in CLAIMS.items():
        if not c["hunt"]:
            continue
        if name == "rsqrt-A87-fulldomain" and fast:
            continue
        cmd = tuple(c["hunt"])
        if cmd in ran:
            continue
        ran.add(cmd)
        print(f"re-running {' '.join(cmd)} ...", flush=True)
        subprocess.run([sys.executable, "-m", *cmd], capture_output=True,
                       text=True)

    for name, c in CLAIMS.items():
        if name == "rsqrt-A87-fulldomain" and fast:
            rows.append({"name": name, "status": "SKIPPED (fast)"})
            continue
        rep_path = newest_report(c["report"])
        if not rep_path:
            rows.append({"name": name, "status": "NO REPORT", "ok": False})
            continue
        report = json.load(open(rep_path))
        actual = extract(report, c["extract"])
        ok = (actual is not None
              and abs(actual - c["err"]) <= c["rtol"] * c["err"])
        rows.append({"name": name, "claimed": c["err"], "reproduced": actual,
                     "rtol": c["rtol"], "ok": ok, "metric": c["metric"],
                     "scope": c["scope"], "provenance": c["provenance"],
                     "certificate": c["certificate"],
                     "report": rep_path})
        flag = "OK " if ok else "MISMATCH"
        av = f"{actual:.6e}" if isinstance(actual, float) else actual
        print(f"[{flag}] {name}: claimed {c['err']:.6e} reproduced {av} "
              f"({c['metric']}, {c['scope']})")

    bad = [r for r in rows if not r.get("ok") and "SKIP" not in r.get("status", "")]
    out_dir = Path("runs") / f"approximants-audit-{int(t0)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "reconciliation.json").write_text(json.dumps(
        {"seconds": round(time.time() - t0, 1), "rows": rows,
         "mismatches": len(bad)}, indent=2))
    print(f"\napproximants audit: {len(rows)} claims, {len(bad)} mismatch "
          f"-> {out_dir}/reconciliation.json ({round(time.time()-t0,1)}s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
