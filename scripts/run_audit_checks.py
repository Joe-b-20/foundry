"""Audit probes: re-check claims against artifacts, re-execute archived
winners, and surface gaps. Rerunnable; writes findings to runs/audit-<ts>/.

Checks:
A1  PCF shelf rejects — verify the claimed mechanism (j=0 -> b(1)=0 ->
    truncation -> rational under our indexing convention).
A2  Every run directory cited in TRACKER.md exists and has report.json.
A3  Re-execute every archived sorting-network winner (verify-on-write,
    re-checked post hoc — the parent's exp2 lesson).
A4  Re-verify the sweep's two findings through a fresh pack.
A5  Artifact completeness: do karatsuba / doctor-exam reports persist the
    found candidate machine-readably (not just pretty strings)?
A6  Source TODO/FIXME census.

Run from repo root:  python3 -m scripts.run_audit_checks
"""

import glob
import json
import re
import sys
import time
from pathlib import Path

FINDINGS = []


def finding(check, status, detail):
    FINDINGS.append({"check": check, "status": status, "detail": detail})
    print(f"[{status}] {check}: {detail}")


def a1_pcf_rejects():
    from domains.pcf_shelf import candidate_refs
    refs = {rid: cand for rid, fam, params, cand in candidate_refs()}
    data = json.loads(Path("domains/pcf_refs.json").read_text())
    rejects = data["rejects"]
    explained, unexplained = [], []
    for rj in rejects:
        cand = refs.get(rj["id"])
        if cand is None:
            unexplained.append((rj["id"], "not in candidate_refs"))
            continue
        b = cand[1]
        b1 = sum(c * (1 ** k) for k, c in enumerate(b))
        if rj["reason"] == "rational" and b1 == 0:
            explained.append(rj["id"])
        else:
            unexplained.append((rj["id"], f"reason={rj['reason']}, b(1)={b1}"))
    status = "OK" if not unexplained else "FLAG"
    finding("A1 pcf-rejects", status,
            f"{len(explained)}/{len(rejects)} rejects explained by b(1)=0 "
            f"truncation; unexplained: {unexplained}")


def _expand_braces(p):
    m = re.match(r"(.*?)\{([0-9,]+)\}(.*)", p)
    if not m:
        return [p]
    out = []
    for v in m.group(2).split(","):
        out.extend(_expand_braces(m.group(1) + v + m.group(3)))
    return out


def a2_tracker_runs():
    text = Path("TRACKER.md").read_text()
    pats = set(re.findall(r"runs/([A-Za-z0-9_\-{},.]+)", text))
    missing = []
    for p in pats:
        for q in _expand_braces(p):
            pat = f"runs/{q.rstrip('/.')}*"
            if not glob.glob(pat):
                missing.append(pat)
    status = "OK" if not missing else "FLAG"
    finding("A2 tracker-artifacts", status,
            f"{len(pats)} cited run patterns; missing: {missing or 'none'}")


def a3_rerun_archived_winners():
    from engine import registry
    total, failed = 0, []
    for arch in glob.glob("runs/sorting_networks-*islands*/archive.json"):
        n = int(re.search(r"-n(\d+)-", arch).group(1))
        pack, mold = registry.build("sorting_networks", {"n": n})
        for e in json.loads(Path(arch).read_text()):
            cand = tuple(tuple(p) for p in e["cand"])
            ok, det = pack.verify_trusted(mold, cand)
            total += 1
            if not ok:
                failed.append((arch, e["provenance"], det))
    status = "OK" if not failed else "FAIL"
    finding("A3 re-execute-archived", status,
            f"{total} archived winners independently re-executed; "
            f"failures: {failed or 'none'}")


def a4_reverify_sweep():
    from engine import registry
    pack, mold = registry.build("pcf", {})
    results = {}
    for name, a in (("apery", (5, 27, 51, 34)), ("rm-8-7zeta3", (1, 5, 9, 6))):
        rec = pack.verify((a, (0, 0, 0, 0, 0, 0, -1)))
        results[name] = (rec["status"], rec.get("constant"))
    ok = all(v[0] == "verified" and v[1] == "zeta3" for v in results.values())
    finding("A4 reverify-sweep-findings", "OK" if ok else "FAIL", str(results))


def a5_artifact_completeness():
    gaps = []
    for f in glob.glob("runs/bilinear-karatsuba-*/report.json"):
        r = json.loads(Path(f).read_text())
        if r.get("found_r3") and "found_cand" not in r and "cand" not in str(r.keys()):
            gaps.append(("karatsuba", f))
    for f in glob.glob("runs/bitmixer-planted-*/report.json") \
            + glob.glob("runs/bitmixer-deceptive-*/report.json"):
        r = json.loads(Path(f).read_text())
        if r.get("found") and "found_cand" not in r:
            gaps.append(("bitmixer", f))
    status = "OK" if not gaps else "FLAG"
    finding("A5 artifact-completeness", status,
            f"reports lacking machine-readable found candidate: "
            f"{len(gaps)} ({sorted(set(g[0] for g in gaps)) or 'none'})")


def a6_todos():
    hits = []
    for f in glob.glob("engine/*.py") + glob.glob("domains/*.py") \
            + glob.glob("scripts/*.py"):
        if f.endswith("run_audit_checks.py"):
            continue                       # this file names the markers
        for i, line in enumerate(Path(f).read_text().splitlines(), 1):
            if re.search(r"\b(TODO|FIXME|XXX)\b", line):
                hits.append(f"{f}:{i}")
    finding("A6 todo-census", "OK" if not hits else "INFO",
            f"{len(hits)} TODO/FIXME markers: {hits or 'none'}")


def main():
    t0 = time.time()
    for fn in (a1_pcf_rejects, a2_tracker_runs, a3_rerun_archived_winners,
               a4_reverify_sweep, a5_artifact_completeness, a6_todos):
        try:
            fn()
        except Exception as e:        # an audit must not die mid-audit
            finding(fn.__name__, "ERROR", repr(e))
    out_dir = Path("runs") / f"audit-{int(t0)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "findings.json").write_text(json.dumps(
        {"seconds": round(time.time() - t0, 2), "findings": FINDINGS},
        indent=2))
    bad = [f for f in FINDINGS if f["status"] in ("FAIL", "ERROR")]
    print(f"\naudit: {len(FINDINGS)} checks, "
          f"{len(bad)} FAIL/ERROR -> {out_dir}/findings.json")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
