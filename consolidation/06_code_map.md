# Code Map & Reproduction

How the code is organized, what each file does, the dependencies, how to run, the
configs that matter, and where results live. Distilled from a full read of every
file. All paths are relative to the project root
(`/home/joebachir20/math_lab`, mounted on Windows at
`\\wsl.localhost\ubuntu\home\joebachir20\math_lab`).

---

## How to run

**Local (4060):** everything runs in the conda env `mathlab` (py3.11, torch 2.6
cu124). The wrapper:
```
bash run.sh <script.py> [args...]
```
`run.sh` sources conda, activates `mathlab`, `cd`s to the project, and runs
`python -u` (unbuffered, so background runs show live progress). **Gotcha (still
true):** inline WSL commands with quotes/pipes get mangled in the PowerShell→wsl
handoff — put non-trivial commands in script files.

**Env setup** (idempotent): `env/setup_env.sh` installs Miniforge + the `mathlab`
env + torch; `env/probe.sh` and `env/status.sh` are diagnostics; `env/setup.log` is
its log. Already done — future sessions don't redo it.

**GPU (RunPod):** see the orchestration section below. The launch plan is
`runpod_plan.md` (read it first if going back to GPU). Env there: torch 2.4.1 cu124.

**Smoke first:** most files take `--smoke` (tiny config exercising every code path).
Scale configs are the same code with bigger numbers. Session-10 lesson: smoke then
*measure pace* before a multi-seed campaign (the neural OOM, 2D signal-craft, and QD
launch-bound were all scale-only).

---

## Two reusable cores (the load-bearing machinery)

Most experiments either import or reimplement one of these.

### A. `expG_controller.py` — GRU controller + integer-register VM
The substrate for all sequential-program discovery (mul, div, isqrt, factor, gcd,
sorting, the expM primitive tower).
- **VM**: integer registers (`ACC, REM, VAL, CUR, Q, J`); instruction set
  `[HALT, GETDIGIT, MULDIGIT, SHL, ADD_ACC, INC_J, COMBINE, SUB_D, STOREQ]`.
- **obs = 4-dim `[op_mul, op_div, ge, done]`** — deliberately **constant across a
  cycle's distinct instructions**, so a memoryless policy can't solve it; the
  program phase *must* live in the GRU recurrence.
- **Controller**: `nn.GRU(4, hidden=64) → Linear(hidden, 9)`.
- Imported by `expJ_selfdiscover`, `expG_discover`, `expJ_eval`, `test_extract`,
  `interp_controller*`. The per-op experiments (expM/O/P/K/Q/R/S) **reimplement** this
  pattern with an op-specific VM + obs, not import it.

### B. `expN_matmul.py` — batched differentiable CP decomposition + integer-lattice anneal
The substrate for all multiplication-count (bilinear rank) discovery.
- `matmul_tensor(m,k,p)`, batched `fit_batched`, `search_rank` (sweep R, polish),
  `exact_mask`, `verify_on_matrices`. `float64`, CPU (tiny tensors; GPU no faster).
- Reused by `expN_ladder.py` (rectangular shapes) and `expT_karatsuba.py` (polynomial
  tensor). `expL_matmul.py` is the original 2×2 version; `expN_gf2.py`/`expN_phase2.py`
  are the GF(2) and 4×4 dead-ends.

### Foundation: `core_data.py`
Pure-Python exact arithmetic — codecs (`to_digits`/`from_digits`, LSB-first),
`exact_result`, sampling, and the headline `length_gen_report`. No torch/numpy;
runnable by any python3 (`python core_data.py` self-checks). Imported nearly
everywhere. `expA_mealy.py` defines the project-wide `DEVICE` (cuda if available)
re-exported as `E.DEVICE`.

---

## CPU experiment files (by group)

**Exp A (neural Mealy + extraction):** `expA_mealy.py` (carry/borrow + `extract_fsm`;
the base module imported as `E`), `expA_mul1.py` (single-digit ×, uses
`fsm_extract.py` k-means extractor), `expA_div1.py` (÷ MSB-first), `expA_robustness.py`
(capacity×length sweep), `expMul_full.py` (the not-finite-state wall demo).

**Exp B/C (symbolic GP):** `expB_progsearch.py` (carry via register-VM GP; `--signed`
matters for borrow), `expC_compose.py` (mul = loop composition; `exact|fsm` mode),
`expC_fsm_primitives.py` (grounds mul on the extracted carry FSM; loads
`runs/expA_mealy_d1.pt`).

**Exp D (discrete / division diagnostics):** `expD_discrete.py` (sign-STE exact
extraction; `--op add|sub|mul1|div1`), `expD_divfixed.py` (fixed-divisor diagnostic —
**reused by expI_basechoice**), `expD_modm.py` (non-base-modulus isolation),
`expD_div_compose.py` (÷ = repeated-sub composition).

**Exp E/F/G/H (unified / controller / orders):** `expE_shared.py` (shared +/−),
`expF_unified.py` (the 4-op model + all curricula + hybrid ÷; writes
`expF_unified_dynamic.pt` — **the checkpoint the interp scripts load**),
`expG_controller.py` (core A), `expG_discover.py` (the REINFORCE failure),
`expH_orders.py` (thin driver over expF; writes `expH_order_*.pt`).

**Exp I (representation):** `expI_repr.py` (carry-save; `--sweep`),
`expI_basechoice.py` (base search; reuses expD_divfixed).

**Exp J (the recipe):** `expJ_selfdiscover.py` (core; `--ops mul,div --iters --seed
--save`), `expJ_eval.py` (per-op + per-divisor eval of a checkpoint), `test_extract.py`
(extraction sanity), `run_seeds.sh` (3-seed driver).

**Exp K–U (the breadth):** `expK_gcd.py` (own VM; `gcd_seeds.sh`), `expL_matmul.py` /
`expN_*.py` (tensor rank — core B), `expM_muldigit.py` / `expM_add.py` / `expM_tower.py`
(+ `*_seeds.py`; the primitive tower), `expO_isqrt.py` / `expP_newton.py` (+
`*_seeds.py`; isqrt by primitives), `expQ_sort.py` / `expR_selection.py` (+ `*_seeds.py`;
sorting), `expS_factor.py` (+ `expS_seeds.py`; factorization), `expT_karatsuba.py`,
`expU_superopt.py` (numpy-only exhaustive search, no torch).

**Moonshot CPU (sessions 8–9):** `expV_cfhunt.py` (mpmath continued fractions —
`probe_mp.py` is its dependency check), `expW_openended.py`, `expX_interesting.py`,
`expY_ca.py`, `expZ_invariants.py`, `expAA_multisignal.py`, `expBB_factoradic.py`
(torch; `--config 0..3` required), `expCC_census.py` (abandoned) / `expCC_ladder.py`
(the result), `expDD_evolve.py`, `expEE_logicaldepth.py` / `expEE_evolve.py`,
`expFF_learnability.py` (numpy-only), `expGG_extraction.py` (loads
`runs/expA_mul1_d6_h64_n0.0.pt`).

**Interp/audit:** `interp_unified.py`–`interp_unified6.py`, `interp_controller.py`,
`interp_controller2.py`, `interp_orders.py`, `interp_probe.py`, `audit.py`.

> **Library facts worth knowing:** `expBB`, `expGG`, and all `gpu_*` use torch;
> `expW/X/Y/Z/AA/DD/EE/FF` and `expCC_census` are numpy-only; `expV`/`probe_mp` are
> mpmath; `expCC_ladder` is pure Python. The tensor-rank files force `float64` on CPU.

---

## GPU files (session 10) — and their import graph

Two shared cores; mind the import arrows.

- **`gpu_exp1_novelty.py`** — 1D CA complex-finder at scale; **the shared CA core**.
  Imported by → `gpu_exp1b_ca2d.py` (2D census + Moore hunt) and `gpu_weird_lprog.py`
  (learning-progress). Args: `--radius --W --T --batch --pop --gens --neural
  --neural-steps --seed --out`.
- **`gpu_weird_soup.py`** — BFF primordial soup; **the shared BFF interpreter**
  (`run_programs`). Imported by → `gpu_alife.py` (instrumented long runs +
  `--analyze`), `gpu_metabolism.py` (computation-coupling), `inspect_soup.py`.
- **`gpu_exp2_qd.py`** — GPU Turing-machine sim + MAP-Elites vs evolution vs sampling
  (standalone). Args: `--n --L --Tmax --batch --gens --grid --seed --out`.
  **⚠ KNOWN BUG (2026-06-09 audit §1.2):** `insert()` uses duplicate-index CUDA
  scatter — the archive's fitness↔genome pairing is non-deterministic and was
  observed corrupted (archived best 154; stored genome re-runs to runtime=8000,
  non-halting). Do **not** reuse the MAP-Elites arm without a dedup-safe insert
  (sort + segment-reduce, or `scatter_reduce` amax + matching gather), and
  **re-execute any archived winner before reporting it**. The evolution and
  sampling arms are unaffected.
- **`gpu_exp3_memory.py`** — NTM-lite memory net vs baseline on `--op rev|mul`
  (imports `core_data`). Writes per-arch/seed `.pt` + appends `runs/exp3_results.txt`.
- **`gpu_avida.py`** — NAND-stack composable substrate (standalone; ops
  `nop,a,b,0,1,nand,dup,drop`). `gpu_avida_oe.py` — richer cross-bit stack VM
  (`nand,and,or,xor,add,sub,shl,shr,…`) + target-free edge-of-chaos merit.
- **Inspection tools:** `inspect_ca1d.py`, `inspect_ca2d.py`, `inspect_soup.py` —
  render/characterize survivors after a scale run (you **must** look, not trust the
  scalar).

### The load-bearing signal formulas (quote these exactly)
- 1D scale hunt (`gpu_exp1_novelty`): `interest = 4c(1−c)·exp(−((dmg−0.18)/0.12)²)`;
  `novelty = interest·(1 + clamp((c − neural_bpc)/c, 0, 1))`.
- 2D (`gpu_exp1b_ca2d`): `4c(1−c)·exp(−((dmg−0.12)/0.09)²)` + activity/quiescent gates.
- Target-free Avida (`gpu_avida_oe`): `merit = (dep_a+dep_b)·(0.3 + 4c(1−c))·(1 +
  dep_a·dep_b)·(0.5 + 0.5·nov)`, `c` = zlib ratio of the output grid. (↻ audit:
  the novelty factor was missing from this doc's first version — code is the
  authority, `gpu_avida_oe.py:106,114`.)
- CPU ancestors: expY/expAA `4c(1−c)`; expDD multi
  `4c(1−c)·min(1,NL/4)·exp(−((d−0.22)/0.12)²)` (damage **bump at 0.22**, not chaos);
  expEE `log(runtime)·4c(1−c)`.
- The `0.18`/`0.12` (1D) and `0.12`/`0.09` (2D) damage centers are the one hand-tuned
  knob — calibrate against known class-4 / Conway's Life before a hunt.

---

## Orchestration scripts

**Local seed drivers:** `run_seeds.sh` (expJ mul+div, seeds 0–2), `gcd_seeds.sh`
(expK, seeds 1–3).

**RunPod (session 10):** the campaign was driven **detached** (nohup + `.DONE`/
`.ALLDONE` markers + self-chaining tiers) because RunPod drops SSH constantly and runs
outlast a session.
- `pod_setup.sh` — extract the transferred tarball, verify torch/CUDA, run all four
  `--smoke`s. Run interactively (~1 min).
- `run_tier1.sh` — cheap, high-confidence results banked first (2D census, 1D radius-2
  seed 1, reversal memory-vs-baseline). Ends with `runs/TIER1.ALLDONE`.
- `run_tier2.sh` / `run_tier2b.sh` / `run_tier2c.sh` — the longer runs (1D seeds 2/3,
  the mul stretch, the QD landscape run). `tier2b`/`tier2c` are **live corrections**:
  the QD runs were resized (they were launch-bound at high Tmax) and ultimately
  dropped from the tier because a separate fast QD run already had the answer; `tier2c`
  ends with `runs/CAMPAIGN.ALLDONE`. The launch lines (exact flags) are inside these
  scripts and in `runpod_plan.md` §3.

> The tier scripts pattern (`run(){ ... >runs/$n.log; touch runs/$n.DONE; }` +
> `while [ ! -f X.ALLDONE ]; do sleep; done`) is the reusable recipe for unattended
> RunPod campaigns. Reuse it.

---

## Where results live

| Location | What | Authority |
|----------|------|-----------|
| `runs/` | local 4060 checkpoints (`.pt`), logs, PNGs, and copies of pulled GPU `.npy`/`.json` | authoritative for local experiments |
| `runs_pod/` (top level) | **partial / orphaned** snapshots pulled mid-run from the pod | **not authoritative** — several logs are truncated |
| `runs_pod/runs/` | the **authoritative** complete copy: `.DONE`/`.ALLDONE` markers, finished logs, full outputs | **use this when the two disagree** |

**Trap (documented):** `runs_pod/exp1_r2_s3.log` is a 224-byte red herring (died right
after compressor training); the real complete run is `runs_pod/runs/exp1_r2_s3.log`
(10 KB). Always prefer `runs_pod/runs/`.

**Key checkpoints (consumed by other scripts):**
- `runs/expA_mealy_d1.pt`, `runs/expA_mealy_sub_d1.pt` → carry/borrow FSM grounding
  (expC/expD).
- `runs/expF_unified_dynamic.pt` → the interp scripts.
- `runs/expG_controller.pt` → interp_controller.
- `runs/expM_muldigit.pt`, `runs/expM_add.pt` → the expM tower.
- `runs/expA_mul1_d6_h64_n0.0.pt` → expGG extraction probe.
- `runs/expJ_both_s{0,1,2}.pt`, `runs/expO_isqrt.pt`, `runs/expP_newton.pt`,
  `runs/expQ_sort.pt`, `runs/expR_selection.pt`, `runs/expS_factor.pt`,
  `runs/expK_gcd.pt` → the per-op discovered models.

**Session-10 GPU outputs of note** (in `runs_pod/runs/` unless said):
`exp1_r2_s{1,2,3}/survivors.json` (+ `top_spacetime.npy`), `exp1b_census*/` (3
threshold variants), `exp2_fast/qd_result.json` (the QD result: sampling 49 /
evolution 6238 / ME 154), `exp3_results.txt` (rev + mul len-gen tables),
`avida_s1/avida_log.json` (XOR/EQU ladder), `oe_s{1,7}/oe_log.json` (thin —
target-free), `alife_lowmut/` & `alife_l32/` (soup trajectories),
`weird_lprog/lprog_survivors.json`, `em{2,3,4}/` (soup emergence runs),
`metab_logic/` & `metab_intrinsic/` (the plateau negative).

---

## Reproduce the headline results (commands)

```
# the recipe: discover mul+div from outcome, one model, 3 seeds
bash run.sh expJ_selfdiscover.py --ops mul,div --iters 160 --seed 0 --save runs/expJ_both_s0.pt
# GCD -> Euclid
bash run.sh expK_gcd.py --iters 120 --seed 0
# isqrt -> binary search (square+compare) vs Newton (division VM)
bash run.sh expO_isqrt.py --iters 200 ;  bash run.sh expP_newton.py --iters 150
# matmul -> Strassen / Laderman ; Karatsuba
bash run.sh expN_matmul.py --m 3 --k 3 --p 3 --Rmax 27 --Rmin 22 --restarts 256
bash run.sh expT_karatsuba.py
# the two walls + the bridge (CPU)
bash run.sh expW_openended.py ;  bash run.sh expY_ca.py ;  bash run.sh expZ_invariants.py
# the wall demos
bash run.sh expFF_learnability.py ;  bash run.sh expCC_ladder.py
# GPU (on a pod): bash pod_setup.sh ; nohup bash run_tier1.sh >runs/TIER1.out 2>&1 &
```
