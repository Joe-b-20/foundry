"""Calibration B, part 2: scoped in-grid sweep — rediscover RM 8/(7 zeta3)
FROM OUTCOME inside a predeclared coefficient box, with Apery's zeta(3)
fraction as the in-grid positive control and a blind-null arm.

Parent result replicated at reduced scope: its n6-family hunt (commit
c104290) rediscovered 8/(7 zeta3) via a Mobius-proximity prefilter with an
in-grid Apery control, on a GPU pod. Here, CPU only:

  grid: a(n) = c0 + c1 n + c2 n^2 + c3 n^3, b(n) = -n^6,
        c0 in 1..8, c1 in 1..30, c2 in 0..60, c3 in 1..40  (585,600 cands)
        — contains BOTH Apery (5,27,51,34) and the target (1,5,9,6).
  stage 1: float64 numpy recurrence, 150 terms, renormalized each step;
        keep converged values that land within 1e-9 of the Mobius net
        {(p+qC)/(r+sC): |p,q,r,s|<=8, ps-qr != 0, C in zeta3/pi^2/catalan}.
        (det == 0 entries are rationals — excluded; the rational trap
        handles those downstream anyway.)
  stage 2: survivors get the full two-stage mpmath verification
        (60 dps screen -> 250 dps residual) + structural naming + reference
        subtraction. Discovery is outcome-only: the prefilter sees battery
        CONSTANTS, never shelf forms; naming happens after.
  null arm: 100k random vectors from the disjoint box c3 in 41..80;
        prediction: zero verified survivors. A verified null-arm survivor
        would be reported as a finding for review, not hidden.

PASS = Apery control survives stage 1 and verifies (else VOID)
       AND rm-8-7zeta3 is found and named BY FORM from outcome
       AND the null arm has zero verified survivors.

Run from repo root:  python3 -m scripts.run_pcf_sweep
"""

import dataclasses
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import mpmath as mp

from domains.pcf import PCFPack
from domains.pcf_shelf import battery, factored_by_id
from engine.molds_pcf import PCFMold
from engine.recorder import Recorder

APERY = (5, 27, 51, 34)
TARGET = (1, 5, 9, 6)          # rm-8-7zeta3 — the rediscovery question


@dataclass
class SweepSpec:
    c0: tuple = (1, 8)
    c1: tuple = (1, 30)
    c2: tuple = (0, 60)
    c3: tuple = (1, 40)
    terms_stage1: int = 150
    conv_tol: float = 1e-11
    prefilter_constants: tuple = ("zeta3", "pi^2", "catalan")
    mobius_coeff_bound: int = 8
    prefilter_tol: float = 1e-9
    null_arm_c3: tuple = (41, 80)
    null_arm_samples: int = 100_000
    survivor_verify_cap: int = 200    # loudly logged if ever hit
    seed: int = 0
    schema: str = "pcfsweep-v0"


def mobius_net(spec):
    consts = {k: float(v) for k, v in battery(30).items()
              if k in spec.prefilter_constants}
    K = spec.mobius_coeff_bound
    vals = []
    for C in consts.values():
        for p in range(-K, K + 1):
            for q in range(-K, K + 1):
                for r in range(-K, K + 1):
                    for s in range(-K, K + 1):
                        if (r == 0 and s == 0) or p * s - q * r == 0:
                            continue
                        den = r + s * C
                        if abs(den) < 1e-9:
                            continue
                        v = (p + q * C) / den
                        if abs(v) <= 1e3:
                            vals.append(v)
    return np.unique(np.asarray(vals))


def stage1(c0, c1, c2, c3, spec):
    """Vectorized float64 PCF evaluation. Returns (value, converged)."""
    n_c = c0.shape[0]
    p2, q2 = np.ones(n_c), np.zeros(n_c)
    p, q = c0.astype(np.float64), np.ones(n_c)
    v_prev = np.full(n_c, np.nan)
    with np.errstate(all="ignore"):
        for n in range(1, spec.terms_stage1 + 1):
            an = ((c3 * n + c2) * n + c1) * n + c0
            bn = -float(n) ** 6
            p, p2 = an * p + bn * p2, p
            q, q2 = an * q + bn * q2, q
            scale = np.abs(q)
            scale[(scale == 0) | ~np.isfinite(scale)] = 1.0
            p, p2, q, q2 = p / scale, p2 / scale, q / scale, q2 / scale
            if n == spec.terms_stage1 - 1:
                v_prev = np.where(q != 0, p / q, np.nan)
        v = np.where(q != 0, p / q, np.nan)
        conv = (np.isfinite(v) & np.isfinite(v_prev)
                & (np.abs(v - v_prev) < spec.conv_tol))
    return v, conv


def prefilter(v, conv, net, tol):
    idx = np.searchsorted(net, v)
    lo = np.clip(idx - 1, 0, len(net) - 1)
    hi = np.clip(idx, 0, len(net) - 1)
    near = np.minimum(np.abs(v - net[lo]), np.abs(v - net[hi])) < tol
    return conv & near


def main():
    t0 = time.time()
    spec = SweepSpec(seed=int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    run_id = f"pcf-sweep-s{spec.seed}-{int(t0)}"
    rec = Recorder(Path("runs") / run_id, run_id)
    rec.event("foreman", "runspec", payload=dataclasses.asdict(spec),
              reason="predeclaration: box, prefilter, tolerances, controls "
                     "and pass conditions fixed before the sweep")

    pack, mold = PCFPack(), PCFMold()
    refs, rejects = pack.load_refs()
    dense_names = {}
    for rid, f in factored_by_id().items():
        dense_names.setdefault(PCFMold.dense(mold.tidy(f)), []).append(rid)

    net = mobius_net(spec)
    rec.event("foreman", "mobius_net", payload={"size": int(len(net))})

    # --- main arm -------------------------------------------------------
    r0 = np.arange(spec.c0[0], spec.c0[1] + 1)
    r1 = np.arange(spec.c1[0], spec.c1[1] + 1)
    r2 = np.arange(spec.c2[0], spec.c2[1] + 1)
    r3 = np.arange(spec.c3[0], spec.c3[1] + 1)
    g = np.stack(np.meshgrid(r0, r1, r2, r3, indexing="ij"), -1).reshape(-1, 4)
    v, conv = stage1(g[:, 0].astype(float), g[:, 1].astype(float),
                     g[:, 2].astype(float), g[:, 3].astype(float), spec)
    surv_mask = prefilter(v, conv, net, spec.prefilter_tol)
    survivors = [tuple(int(x) for x in row) for row in g[surv_mask]]
    rec.event("foreman", "stage1", payload={
        "grid": int(len(g)), "converged": int(conv.sum()),
        "survivors": len(survivors)})
    print(f"stage 1: {len(g)} candidates, {int(conv.sum())} converged, "
          f"{len(survivors)} prefilter survivors")

    if len(survivors) > spec.survivor_verify_cap:
        rec.event("foreman", "CAP-HIT", payload={
            "survivors": len(survivors), "cap": spec.survivor_verify_cap},
            reason="verifying only the cap; coverage is INCOMPLETE")
        print(f"!! CAP: {len(survivors)} survivors > {spec.survivor_verify_cap}"
              " — verifying the first cap only, coverage incomplete")
    to_verify = survivors[: spec.survivor_verify_cap]

    apery_surv = APERY in survivors
    found = []
    for a in to_verify:
        cand = mold.tidy((a, (-1, 6, ())))
        lab = pack.classify(mold, cand)
        names = dense_names.get(PCFMold.dense(cand), [])
        found.append({"a": list(a), "label": lab.get("label"),
                      "constant": lab.get("constant"),
                      "named_by_form": names,
                      "drop_reason": lab.get("drop_reason"),
                      "delta": lab.get("delta")})
        rec.event("judge", "classify", payload=found[-1])
    verified = [f for f in found if f["label"] != "DROPPED"]

    # --- control gate -----------------------------------------------------
    apery_ver = any(f["a"] == list(APERY) and f["label"] != "DROPPED"
                    and "apery-zeta3" in f["named_by_form"] for f in found)
    void = not (apery_surv and apery_ver)
    rec.event("control", "apery_in_grid", payload={
        "survived_stage1": apery_surv, "verified_and_named": apery_ver},
        outcome="C+" if not void else "C- VOID")

    # --- the question -----------------------------------------------------
    target_hit = any(f["a"] == list(TARGET) and f["label"] != "DROPPED"
                     and "rm-8-7zeta3" in f["named_by_form"] for f in found)

    # --- null arm -----------------------------------------------------------
    rng = np.random.default_rng(spec.seed)
    null_g = np.column_stack([
        rng.integers(spec.c0[0], spec.c0[1] + 1, spec.null_arm_samples),
        rng.integers(spec.c1[0], spec.c1[1] + 1, spec.null_arm_samples),
        rng.integers(spec.c2[0], spec.c2[1] + 1, spec.null_arm_samples),
        rng.integers(spec.null_arm_c3[0], spec.null_arm_c3[1] + 1,
                     spec.null_arm_samples)])
    nv, nconv = stage1(null_g[:, 0].astype(float), null_g[:, 1].astype(float),
                       null_g[:, 2].astype(float), null_g[:, 3].astype(float),
                       spec)
    nmask = prefilter(nv, nconv, net, spec.prefilter_tol)
    null_surv = [tuple(int(x) for x in row) for row in null_g[nmask]]
    null_verified = []
    for a in dict.fromkeys(null_surv):
        lab = pack.classify(mold, mold.tidy((a, (-1, 6, ()))))
        if lab.get("label") != "DROPPED":
            null_verified.append({"a": list(a), "label": lab.get("label"),
                                  "constant": lab.get("constant")})
    rec.event("foreman", "null_arm", payload={
        "samples": spec.null_arm_samples, "stage1_survivors": len(null_surv),
        "verified": null_verified})

    report = {
        "run_id": run_id, "spec": dataclasses.asdict(spec),
        "seconds": round(time.time() - t0, 2),
        "void": void,
        "stage1": {"grid": int(len(g)), "converged": int(conv.sum()),
                   "survivors": len(survivors)},
        "verified_findings": verified,
        "apery_control": {"survived": apery_surv, "verified": apery_ver},
        "target_rediscovered_from_outcome": target_hit,
        "null_arm": {"stage1_survivors": len(null_surv),
                     "verified": null_verified},
        "PASS": (not void) and target_hit and not null_verified,
    }
    rec.event("foreman", "report", payload=report)
    (rec.run_dir / "report.json").write_text(json.dumps(report, indent=2))
    rec.close()
    print(json.dumps({k: report[k] for k in
                      ("stage1", "apery_control",
                       "target_rediscovered_from_outcome", "null_arm",
                       "PASS", "seconds")}, indent=2))
    for f in verified:
        print("verified:", f["a"], "->", f["constant"],
              "| named:", f["named_by_form"] or "(no published form)")
    return 0 if report["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
