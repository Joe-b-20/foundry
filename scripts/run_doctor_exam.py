"""The C1/C2 exam — the calibration roof. Same interface, opposite
correct answers, both failure directions tested:

- C1 (planted, 3 seeds): a length-4 program from the mold's own op set is
  hidden behind the I/O interface. PASS = the engine finds it (exact on
  corpus, then verified exhaustively on all 2^16 pairs) AND the operator
  never accepts an abandon recommendation first (wrong-quit direction).
- C2 (keyed, 3 seeds): a 6-round rotation+key mixer outside the op set.
  PASS = the doctor recommends abandoning and the operator accepts it
  within budget, with no exact found (wrong-grind direction).

Operator policy (predeclared, v1 doctor recommends-only): a diagnosis is
requested every 50 generations; TWO consecutive abandon recommendations
are accepted. Single recommendations on C1 are logged as false-alarm
warnings; an ACCEPTED abandon on C1 fails the exam.

EXAM PASS = all 6 universes.

Run from repo root:  python3 -m scripts.run_doctor_exam
"""

import dataclasses
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from engine import judge, registry
from engine.doctor import WallDoctor
from engine.proposers import EvolutionProposer
from engine.recorder import Recorder


@dataclass
class ExamSpec:
    w: int = 8
    planted_len: int = 6            # the difficulty dial (4 was trivial:
    corpus_size: int = 128          # random init solved it at gen 0)
    pop_size: int = 64
    generations: int = 4000
    init_length: int = 4
    doctor_min_gens: int = 800
    doctor_window: int = 500
    doctor_margin: float = 0.03
    diagnose_every: int = 50
    accept_after: int = 2           # consecutive abandon recs accepted
    schema: str = "doctorexam-v0"


def run_universe(target, seed, spec):
    t0 = time.time()
    run_id = f"bitmixer-{target}-s{seed}-{int(t0)}"
    rec = Recorder(Path("runs") / run_id, run_id)
    rec.event("foreman", "runspec",
              payload={**dataclasses.asdict(spec), "target": target,
                       "seed": seed},
              reason="predeclaration incl. operator accept-after policy")
    pack, mold = registry.build("bitmixer", {
        "target": target, "seed": seed, "w": spec.w,
        "planted_len": spec.planted_len, "corpus_size": spec.corpus_size})
    baselines = pack.baselines()
    total_bits = spec.w * len(pack.corpus)
    doctor = WallDoctor(spec.doctor_min_gens, spec.doctor_window,
                        chance_margin=spec.doctor_margin)
    prop = EvolutionProposer(pop_size=spec.pop_size)
    rng = random.Random(seed)
    ctx = {"rng": rng, "mold": mold, "batch": spec.pop_size,
           "init_length": spec.init_length}

    best_frac, best_len, best_cand = 0.0, None, None
    found = None
    false_alarms, abandon_streak = 0, 0
    accepted_abandon_gen = None
    heldout_best = None
    evals = 0
    for gen in range(spec.generations):
        results = []
        for cand in prop.propose(ctx):
            tidy, sc, _cost = judge.score(pack, mold, cand)
            results.append((tidy, sc))
            evals += 1
            frac = sc[1] / total_bits
            if frac > best_frac:
                best_frac, best_len, best_cand = frac, len(tidy), tidy
            if sc[0] == 1 and found is None:
                found = (gen, tidy)
        prop.feedback(results)
        doctor.observe(gen, best_frac, best_len)
        if found:
            break
        if gen and gen % spec.diagnose_every == 0:
            heldout_best = (pack.heldout_frac(best_cand)
                            if best_cand else None)
            v = doctor.diagnose(baselines, heldout_best=heldout_best)
            if v:
                rec.event("doctor", "verdict", payload=v)
                if v["recommendation"] == "abandon-target":
                    abandon_streak += 1
                    if target == "planted":
                        false_alarms += 1
                    if abandon_streak >= spec.accept_after:
                        accepted_abandon_gen = gen
                        rec.event("operator", "decision",
                                  payload={"gen": gen},
                                  reason=f"{spec.accept_after} consecutive "
                                         "abandon recommendations — "
                                         "predeclared policy accepts",
                                  outcome="ABANDON")
                        break
                else:
                    abandon_streak = 0
            else:
                abandon_streak = 0

    out = {"run_id": run_id, "target": target, "seed": seed,
           "evals": evals, "seconds": round(time.time() - t0, 2),
           "best_frac": round(best_frac, 4),
           "heldout_best": (round(heldout_best, 4)
                            if heldout_best is not None else None),
           "chance_plus": round(max(baselines.values()), 4),
           "found": bool(found), "found_gen": found[0] if found else None,
           "false_alarm_recs": false_alarms,
           "abandon_accepted_gen": accepted_abandon_gen}
    if found:
        ok, det = pack.verify_trusted(mold, found[1])
        out["verified_exhaustive"] = ok
        out["found_pretty"] = mold.pretty(mold.tidy(found[1]))
        out["planted_pretty"] = mold.pretty(pack.reveal()) if pack.reveal() else None
        rec.event("judge", "verify_on_write",
                  payload=det, outcome="pass" if ok else "FAIL")
    if target == "planted":
        out["PASS"] = bool(found) and out.get("verified_exhaustive", False) \
            and accepted_abandon_gen is None
    else:
        out["PASS"] = (accepted_abandon_gen is not None) and not found
    rec.event("foreman", "report", payload=out)
    (rec.run_dir / "report.json").write_text(json.dumps(out, indent=2))
    rec.close()
    return out


def main():
    spec = ExamSpec()
    rows = []
    for target in ("planted", "keyed"):
        for seed in (0, 1, 2):
            r = run_universe(target, seed, spec)
            rows.append(r)
            extra = (f"found@{r['found_gen']} verified={r.get('verified_exhaustive')}"
                     if r["found"] else
                     f"abandon@{r['abandon_accepted_gen']}")
            print(f"{target} s{seed}: {'PASS' if r['PASS'] else 'FAIL'} "
                  f"corpus={r['best_frac']} heldout={r['heldout_best']} "
                  f"chance+={r['chance_plus']} {extra} "
                  f"false_alarms={r['false_alarm_recs']} t={r['seconds']}s")
            if r.get("found_pretty"):
                print(f"   found:   {r['found_pretty']}")
                print(f"   planted: {r['planted_pretty']}")
    exam_pass = all(r["PASS"] for r in rows)
    path = f"runs/doctor_exam_summary-{int(time.time())}.json"
    with open(path, "w") as fh:
        json.dump({"rows": rows, "EXAM_PASS": exam_pass}, fh, indent=2)
    print(f"\nEXAM PASS: {exam_pass} (bar: all 6 universes) -> {path}")
    return 0 if exam_pass else 1


if __name__ == "__main__":
    sys.exit(main())
