# PLAYBOOK — executor-agnostic plan + verification gates (2026-06-10)

Written by the Fable 5 instance at Joe's request: the forward plan, specified so that
**any executor** (a different model, a fresh session after context loss, or Joe
pasting commands) can carry it out, and so that the work can be **verified by anyone**
afterward without trusting the executor's narration. This codifies how the project
survived its own audit: artifacts over claims, re-execution over trust.

**If you are an executor picking this up cold:** read this file, then
`consolidation/README.md`, then the last 4 entries of `TRACKER.md`
(`grep -n '^## ' TRACKER.md | tail -4`). Do NOT rely on chat history. Do NOT
re-derive strategy — execute the queue in §3, gate by gate.

---

## 1. Ground rules for ANY executor (non-negotiable, from .claude/RULES.md)

1. **Artifacts or it didn't happen.** Every number you report must exist in a file
   under `runs/` or `runs_pod/`. No number read off a screen and retyped.
2. **Never edit old TRACKER entries** — append; corrections are dated `[ERRATUM]`
   brackets.
3. **Comparative claims need ≥3 seeds** or an explicit `n=1` tag.
4. **Re-execute any archived/stored winner** before reporting it (the exp2 lesson).
5. **Render before verdict** on any dynamical object ("structured/clean/class-4").
6. **Stratify mixed-condition losses** before claiming a fit (the exp3 lesson).
7. **Commit after each completed unit** with a message naming the artifact paths.
8. **No novelty claims.** Anything from the identity hunt is "verified to N digits +
   not in references checked," never "new mathematics."
9. **Don't tune a stuck approach >3 times** — mark ABANDONED, move on.
10. When in doubt: smaller, exact, honest.

## 2. Current state (as of 2026-06-10 ~14:45 UTC)

- **Pod (RunPod 4090, root@213.181.111.2:42706, key ~/.ssh/id_ed25519):**
  closures DONE + pulled + analyzed (TRACKER entries "CLOSURE A/B", commit e1cb715).
  Phase-2 running detached (`runs/PHASE2.out`): pcf_stage1 DONE
  (4.82M PCFs → 4.32M survivors), pcf_stage2 IN PROGRESS (~56 min, 10 hits @ chunk
  60/240), then 12 loop-sweep runs (`loop_m{1,4,16}_s{1,2}` — ~1.5 h).
  Everything is nohup + `.DONE` markers: SSH drops and local sleep are harmless.
- **Local 4060:** `gpu_depthstruct.py` seed 1 running via
  `nohup run_dstruct_local.sh` (`runs/DSTRUCT_LOCAL.out`); seed 2 queued after.
- **Done this session:** audit (consolidation/09) → doc repairs in place → RULES
  amendments → git (5 commits) → expFF_search closure (wall #5 measured) → exp2fix
  closure (QD verdict overturned; depth heavy-tailed) → oe_fix closure (a+b
  disconfirmed; unnamed-structured attractor).

## 3. Execution queue (each task: command → expected artifact → acceptance gate)

### T1. Collect phase-2 pod results (when `runs/PHASE2.ALLDONE` exists on pod)
```
ssh -p 42706 -i ~/.ssh/id_ed25519 root@213.181.111.2 'ls /workspace/math_lab/runs/PHASE2.ALLDONE'
mkdir -p ~/math_lab/runs_pod/phase2 && cd ~/math_lab/runs_pod/phase2
scp -P 42706 -i ~/.ssh/id_ed25519 -r 'root@213.181.111.2:/workspace/math_lab/runs/pcf_main' \
    'root@213.181.111.2:/workspace/math_lab/runs/loop_m*' \
    'root@213.181.111.2:/workspace/math_lab/runs/*.log' \
    'root@213.181.111.2:/workspace/math_lab/runs/PHASE2.out' .
```
- **Artifact:** `runs_pod/phase2/pcf_main/stage2_summary.json`, 6 `loop_m*_s*/loop_log.json`.
- **Gate:** `stage2_summary.json` has `"control_ok": true` (the injected 4/π control
  was recovered). If false, the whole pcf null/hit set is UNINTERPRETABLE — report
  that, do not interpret hits.

### T2. Analyze pcf (identity hunt)
```
cd ~/math_lab && python3 analyze_phase2.py pcf runs_pod/phase2/pcf_main
```
- **Gate 1:** control_ok true (again).
- **Gate 2:** for every TAIL hit (catalan/zeta3/gamma) printed: `verified: true`
  (the 250-digit re-check). Only verified hits may be mentioned.
- **Gate 3 (honesty):** classical π/e-family hits are REDISCOVERIES — label them so.
  Tail hits get: PCF definition (A,B), value to 30 digits, relation, height, and the
  sentence "true to 250 digits; not checked against literature beyond the project's
  references; novelty NOT claimed."
- **TRACKER entry** per template §4. Status WORKS if control_ok, regardless of hits.

### T3. Analyze loop sweep (naming-density vs depth)
```
python3 analyze_phase2.py loop runs_pod/phase2/loop_m1_s1 runs_pod/phase2/loop_m1_s2 \
  runs_pod/phase2/loop_m4_s1 runs_pod/phase2/loop_m4_s2 runs_pod/phase2/loop_m16_s1 runs_pod/phase2/loop_m16_s2
```
- **Gate:** the by-maxit aggregate prints 3 rows (maxit 1/4/16), 2 seeds each.
- **Interpretation rule (pre-registered):** top_named_sim falling with maxit ⇒ depth
  gates naming; flat (≈0.55–0.8 everywhere) ⇒ recognition is the ceiling (consistent
  with oe_fix). Do NOT invent a third reading; if the two seeds disagree wildly,
  report "inconclusive, n=2."
- **TRACKER entry** per §4.

### T4. Finish + analyze local depthstruct
```
ls runs/DSTRUCT_LOCAL.ALLDONE   # wait for it (local; survives via nohup)
python3 analyze_phase2.py dstruct runs/dstruct_s1 runs/dstruct_s2
```
- **Gate 1:** every winner row prints `rerun_ok=True`. Any False ⇒ that condition's
  numbers are VOID (report as such).
- **Gate 2 (render-before-verdict):** open the generated `*_render.png` head-position
  traces and DESCRIBE what is seen (sweep/counter/irregular) before any "structured"
  claim. The track/tape scores alone are not a verdict.
- **TRACKER entry** per §4: the question is whether depth×track found machines that
  are deep AND mid-track (structured) vs depth-only's metronome counters.

### T5. Docs follow-through (after T2–T4)
- Update `consolidation/03_wall_taxonomy.md` #4: replace the audit-era "QD verdict
  void" with the measured closure result (ME competitive; rt_span ME 2139 > evo 675
  n=1; evolution heavy-tailed 596–1887; cap not binding at Tmax 30k n=1).
- Update `08_what_not_to_redo.md` QD entry the same way; un-redact "use QD with a
  depth-aligned descriptor" as a live option.
- Update `07_open_frontiers.md` Frontier-2 box with the oe_fix result (unnamed-
  structured attractor disconfirms the vocabulary ceiling at the artifact level) +
  whatever T3 adds.
- Mark every edit "↻ closure 2026-06-10".

### T6. Wrap-up
- `git add -A && git commit` (message lists artifact paths).
- Final TRACKER session entry (template §4): what ran, what it cost, what changed
  in the framings, what's queued next.
- Stop/release the pod ONLY after T1's scp is verified complete (`ls` the local
  copies). Joe decides pod shutdown.

## 4. TRACKER entry template (copy exactly)
```
## YYYY-MM-DD — <short name>
What I tried: <1 paragraph: the question + the setup; name the script + config>
What happened: <1 paragraph: CONCRETE numbers, each present in a named artifact file>
What I learned: <1-2 sentences; separate measured fact from interpretation>
Status: WORKS / FAILED / PARTIAL / ABANDONED (+ n=1 tags where applicable)
Files: <artifact paths>
```

## 5. Verification protocol (how the reviewer checks ANY executor's work)

Run these regardless of who executed:
1. `git log --oneline` + `git diff <prev>..HEAD --stat` — every changed file
   explainable by the queue above? Unexplained changes ⇒ investigate before merging
   narrative.
2. For every number in the new TRACKER entries: `grep` it in the named artifact.
   Missing ⇒ strike the claim.
3. Spot re-execution: 1 random winner per experiment (qd genome, loop top program,
   dstruct machine) re-run from its stored artifact; result must match the logged
   value exactly.
4. Check gates: control_ok (pcf), rerun_ok (dstruct), n-seeds tags, render PNGs
   exist and were described.
5. Diff the doc edits against the actual closure numbers (no narrative drift —
   the audit's core lesson).

## 6. Known traps for executors (learned this session — do not repeat)
- PowerShell→WSL quoting mangles inline quotes/pipes: put commands in `.sh` files.
- The laptop SLEEPS when unattended: anything load-bearing runs on the POD under
  nohup with `.DONE` markers; local runs are best-effort only.
- `wsl -e bash <path>` gets Git-Bash path-mangled: use `wsl -e bash -c "bash /path"`.
- Duplicate-index CUDA scatter is nondeterministic (the exp2 bug); never use for
  archives.
- A mixed-width loss tail is not convergence (the exp3 lesson).
- runs_pod top level vs runs_pod/runs: prefer the directory with `.DONE` markers.
- mpmath: set `MPMATH_NOGMPY=1` (gmpy2 segfaults on pathological PCFs).
```
