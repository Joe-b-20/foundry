"""The C1/C2/C3 exam — the calibration roof. Same interface, different
correct answers, three directions tested:

- C1 (planted, 3 seeds): a length-6 program from the mold's own op set is
  hidden behind the I/O interface. PASS = the engine finds it (exact on
  corpus, then verified exhaustively on all 2^16 pairs) AND the operator
  never accepts an abandon recommendation first (wrong-quit direction).
- C2 (keyed, 3 seeds): an 8-round rotation+key mixer outside the op set.
  PASS = the doctor recommends abandoning (high confidence) and the
  operator accepts within budget, no exact found (wrong-grind direction).
- C3 (deceptive, 3 seeds): out=(x^y)*(x+y) — reachable, but partial
  assembly plateaus ABOVE chance for a long time before a sudden grok-style
  reorganization. PASS = the doctor NEVER issues a high-confidence abandon
  (it may only ever say switch/raise). This is Joe's grokking case made a
  permanent exam question (2026-06-12).

Operator policy (predeclared, v1 doctor recommends-only): diagnosis every
50 generations; an abandon is accepted only if HIGH confidence and TWO
consecutive. Single/low-confidence recs on C1 are logged as false alarms;
an accepted abandon on C1 or C3 fails the exam.

EXAM PASS = all 9 universes.

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


DECEPTIVE_PLANT = (("XOR", 2, 0, 1), ("ADD", 3, 0, 1), ("MUL", 0, 2, 3))


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
    doctor_abandon_min_gens: int = 1600   # high-confidence abandon horizon
    diagnose_every: int = 50
    accept_after: int = 2           # consecutive HIGH-conf abandons accepted
    schema: str = "doctorexam-v1"


def run_universe(target, seed, spec):
    t0 = time.time()
    run_id = f"bitmixer-{target}-s{seed}-{int(t0)}"
    rec = Recorder(Path("runs") / run_id, run_id)
    rec.event("foreman", "runspec",
              payload={**dataclasses.asdict(spec), "target": target,
                       "seed": seed},
              reason="predeclaration incl. operator accept-after policy")
    build_kwargs = {"target": "keyed" if target == "keyed" else "planted",
                    "seed": seed, "w": spec.w,
                    "planted_len": spec.planted_len,
                    "corpus_size": spec.corpus_size}
    if target == "deceptive":
        build_kwargs["plant"] = DECEPTIVE_PLANT
    pack, mold = registry.build("bitmixer", build_kwargs)
    baselines = pack.baselines()
    total_bits = spec.w * len(pack.corpus)
    doctor = WallDoctor(spec.doctor_min_gens, spec.doctor_window,
                        chance_margin=spec.doctor_margin,
                        abandon_min_gens=spec.doctor_abandon_min_gens)
    prop = EvolutionProposer(pop_size=spec.pop_size)
    rng = random.Random(seed)
    ctx = {"rng": rng, "mold": mold, "batch": spec.pop_size,
           "init_length": spec.init_length}

    best_frac, best_len, best_cand = 0.0, None, None
    found = None
    rejected = set()        # corpus-exact but failed exhaustive: keep going
    false_alarms, abandon_streak = 0, 0
    high_conf_abandon_seen = False
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
            if sc[0] == 1 and found is None and tidy not in rejected:
                ok_v, det_v = pack.verify_trusted(mold, tidy)
                if ok_v:
                    found = (gen, tidy, det_v)
                else:
                    rejected.add(tidy)
                    rec.event("judge", "corpus_exact_but_failed_exhaustive",
                              payload={"gen": gen,
                                       "cand": mold.pretty(tidy)},
                              reason="corpus fit is not function identity; "
                                     "search continues")
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
                high = (v["recommendation"] == "abandon-target"
                        and v.get("confidence") == "high")
                if high:
                    high_conf_abandon_seen = True
                    abandon_streak += 1
                    if target in ("planted", "deceptive"):
                        false_alarms += 1
                    if abandon_streak >= spec.accept_after:
                        accepted_abandon_gen = gen
                        rec.event("operator", "decision",
                                  payload={"gen": gen},
                                  reason=f"{spec.accept_after} consecutive "
                                         "HIGH-confidence abandon recs — "
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
        _gen_f, tidy_f, det = found       # verified at find time, in-loop
        out["verified_exhaustive"] = True
        out["found_cand"] = [list(t) for t in tidy_f]
        out["found_pretty"] = mold.pretty(tidy_f)
        out["planted_pretty"] = mold.pretty(pack.reveal()) if pack.reveal() else None
        out["corpus_exact_rejected_by_exhaustive"] = len(rejected)
        rec.event("judge", "verify_on_write",
                  payload=det, outcome="pass")
    if target == "planted":
        out["PASS"] = bool(found) and out.get("verified_exhaustive", False) \
            and accepted_abandon_gen is None
    elif target == "deceptive":
        # the grok must never be confidently abandoned; finding it is a
        # bonus, not required (it's stochastic). The property under test is
        # the doctor's restraint.
        out["high_conf_abandon_seen"] = high_conf_abandon_seen
        out["PASS"] = (accepted_abandon_gen is None
                       and not high_conf_abandon_seen)
    else:  # keyed
        out["PASS"] = (accepted_abandon_gen is not None) and not found
    rec.event("foreman", "report", payload=out)
    (rec.run_dir / "report.json").write_text(json.dumps(out, indent=2))
    rec.close()
    return out


def main():
    spec = ExamSpec()
    rows = []
    for target in ("planted", "keyed", "deceptive"):
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
                if r.get("planted_pretty"):
                    print(f"   planted: {r['planted_pretty']}")
    exam_pass = all(r["PASS"] for r in rows)
    path = f"runs/doctor_exam_summary-{int(time.time())}.json"
    with open(path, "w") as fh:
        json.dump({"rows": rows, "EXAM_PASS": exam_pass}, fh, indent=2)
    print(f"\nEXAM PASS: {exam_pass} (bar: all 9 universes) -> {path}")
    return 0 if exam_pass else 1


if __name__ == "__main__":
    sys.exit(main())
