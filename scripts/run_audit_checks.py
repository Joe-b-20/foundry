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
    """Two known mechanisms (docs/audit_2026-06-12.md F2): b(1)=0
    truncation, and telescoping members that genuinely converge to a
    rational — the latter verified here by evaluation + pslq."""
    from domains.pcf_shelf import candidate_refs
    from engine import numeric
    refs = {rid: cand for rid, fam, params, cand in candidate_refs()}
    data = json.loads(Path("domains/pcf_refs.json").read_text())
    rejects = data["rejects"]
    truncated, telescoping, unexplained = [], [], []
    for rj in rejects:
        cand = refs.get(rj["id"])
        if cand is None:
            unexplained.append((rj["id"], "not in candidate_refs"))
            continue
        b1 = sum(cand[1])                    # b evaluated at n=1
        if rj["reason"] == "rational" and b1 == 0:
            truncated.append(rj["id"])
        elif rj["reason"] == "rational":
            r = numeric.eval_pcf(cand[0], cand[1], terms=1400, dps=60)
            if not r["degenerate"] and numeric.is_rational(r["value"]):
                telescoping.append(rj["id"])
            else:
                unexplained.append((rj["id"], "rational reject but value "
                                              "not confirmed rational"))
        else:
            unexplained.append((rj["id"], f"reason={rj['reason']}"))
    status = "OK" if not unexplained else "FLAG"
    finding("A1 pcf-rejects", status,
            f"{len(truncated)} b(1)=0 truncations + {len(telescoping)} "
            f"verified telescoping rationals = {len(truncated) + len(telescoping)}"
            f"/{len(rejects)}; unexplained: {unexplained or 'none'}")


def _brace_groups(p):
    return re.findall(r"\{([0-9,.]+)\}", p)


def _group_values(g):
    m = re.match(r"^(\d+)\.\.(\d+)$", g)        # {3..8} ranges
    if m:
        return [str(v) for v in range(int(m.group(1)), int(m.group(2)) + 1)]
    return [v for v in g.split(",") if v]


def a2_tracker_runs():
    text = Path("TRACKER.md").read_text()
    # an erratum'd citation reads "runs/<bad-path> [ERRATUM ... real ...]":
    # drop the bad path together with its bracket, then any stray brackets
    text = re.sub(r"runs/\S+\s*\[ERRATUM.*?\]", "", text, flags=re.S)
    text = re.sub(r"\[ERRATUM.*?\]", "", text, flags=re.S)
    pats = set(re.findall(r"runs/([A-Za-z0-9_\-{},.]+)", text))
    missing = []
    for p in pats:
        p = p.rstrip(",./")
        groups = _brace_groups(p)
        if len(groups) <= 1:
            cands = [p] if not groups else [
                p.replace("{" + groups[0] + "}", v, 1)
                for v in _group_values(groups[0])]
            for q in cands:
                if not glob.glob(f"runs/{q}*"):
                    missing.append(f"runs/{q}*")
        else:
            # multi-group shorthand (e.g. s{0,1,2}-ts{a,b,c}) means the
            # DIAGONAL, not the cross product: check each value per axis
            # with the other axes wildcarded
            for gi, g in enumerate(groups):
                for v in _group_values(g):
                    q = p
                    for gj, g2 in enumerate(groups):
                        q = q.replace("{" + g2 + "}", v if gj == gi else "*", 1)
                    if not glob.glob(f"runs/{q}*"):
                        missing.append(f"runs/{q}*")
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
    """Pre-fix runs are immutable history; only the NEWEST runs per family
    must carry machine-readable found candidates."""
    gaps, checked = [], 0
    fams = (("karatsuba", "runs/bilinear-karatsuba-*/report.json",
             "found_r3", 3),
            ("bitmixer", "runs/bitmixer-planted-*/report.json", "found", 3),
            ("bitmixer", "runs/bitmixer-deceptive-*/report.json", "found", 3))
    def ts(path):                 # newest = largest trailing run timestamp
        m = re.search(r"-(\d+)/report\.json$", path)
        return int(m.group(1)) if m else 0

    for fam, pat, flag, newest_n in fams:
        for f in sorted(glob.glob(pat), key=ts)[-newest_n:]:
            r = json.loads(Path(f).read_text())
            if r.get(flag):
                checked += 1
                if "found_cand" not in r:
                    gaps.append((fam, f))
    status = "OK" if not gaps else "FLAG"
    finding("A5 artifact-completeness", status,
            f"{checked} newest found-reports checked; lacking found_cand: "
            f"{gaps or 'none'}")


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
