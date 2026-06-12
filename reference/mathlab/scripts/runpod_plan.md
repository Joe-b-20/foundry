# RunPod GPU Session Plan — mathlab

**Purpose of this file:** so the next session does NOT re-read 300KB of TRACKER.md. Everything needed to launch is here.
Built and tested locally on the 4060 (session 10, 2026-06-09). All code runs clean in `--smoke`; scale configs are
just-bigger-numbers of the same tested paths. **Read §0 → §2 → §5, then launch.**

---

## 0. Read-me-first (60-second orientation)

The project's moonshot = "a small system discovers a math/computational procedure humans don't know, and we can read it."
Across 9 sessions it never fell, and the reason is now **over-determined and mapped** (TRACKER "WALL TAXONOMY"):
regular ops are pinned to their unique minimal machine *by theorem* (rediscovery is forced); hard ops are pinned by
complexity; the escape — an **intrinsic "interestingness" signal** (the *bridge*) that surfaces structure with no target —
is real but **(a) only ever flags KNOWN structure-classes and (b) is scale-bound and landscape-gated.**

The session-9 synthesis named *exactly* what's missing, verbatim:
> "to find a human-unknown object you need an intrinsic signal pointed at a structure-class NOT already named, at a scale
> large enough that the tail is reachable... **None of those three is available on a 4060 in one session.**"

**That sentence is the GPU thesis.** Two walls are explicitly compute-movable: **SCALE (#10)** and **LANDSCAPE (#4, the
only hard wall marked "FIX: unknown").** This session funds swings at both, plus a clean representational-wall crossing.

**Honest prior:** the moonshot probably still won't fall. The *reliable* value is wall-understanding (Exp-2, Exp-3 will
produce clean results regardless). The moonshot swing (Exp-1) is high-variance and gets a thorough, multi-substrate try.

---

## 1. The thesis: what $75 of GPU actually buys

| Wall (taxonomy) | Movable by… | This session's swing |
|---|---|---|
| #10 SCALE | more compute | **Exp-1**: run the complex-finder over UNCATALOGUED spaces (radius-3 1D = 2^128, non-totalistic 2D Moore = 2^512) far past 4060 reach, + a NEW neural higher-order-structure axis, then INSPECT survivors. |
| #4 LANDSCAPE ("FIX: unknown") | a search built for rugged needle-landscapes | **Exp-2**: GPU-scale **MAP-Elites quality-diversity** over the Busy-Beaver program space, vs the plain-evolution baseline that stalled at 43 steps. |
| #1 REPRESENTATIONAL ("FIX: richer substrate") | give the net external memory | **Exp-3**: memory-augmented (stack/tape) net length-generalizes a NON-regular op where a flat recurrent net can't. |

**Key local finding (don't forget):** 1D radius-2 novelty search runs in **~5 seconds** on the 4060 — it does NOT need a
rented GPU. The genuine GPU consumers are **2D CA simulation** (Exp-1b) and **deep-TM simulation** (Exp-2). Budget flows
there. The moonshot swing is *cheap*, so it gets run many ways (seeds, radii, dims, neural on/off).

---

## 2. GPU pick + image + setup

**GPU: RTX 4090 (24 GB), Community Cloud, ~$0.69/hr** (RunPod pricing checked 2026-06-09).
- Why: every workload here is integer/bitwise-parallel CA/TM simulation + tiny neural nets — **CUDA-core/bandwidth-bound,
  modest VRAM, no tensor-core FLOPS or >24GB need.** The 4090 is the $/throughput sweet spot. 24GB is ample (largest
  alloc ~8GB at the recommended batch sizes).
- **Alternatives:** RTX **5090** (32GB, ~$0.99/hr) ≈ 1.3–1.8× faster wall-clock — pick it only if finishing-sooner beats
  $/throughput. **A40 / A6000** (48GB, ~$0.44–0.49/hr) — cheaper/hr but ~1.5–2× slower; only if you want max total
  compute and don't mind slower. **Do NOT** rent A100/H100 — pure waste for this integer-parallel work.
- At $0.69/hr, **$75 ≈ 108 hours.** Realistic spend below is **~15–50 GPU-hr (~$10–35)** — likely UNDER budget; headroom
  goes to deeper QD / more seeds / chasing deeper machines.

**Image:** official RunPod **PyTorch 2.4+ / CUDA 12.4+** template
(e.g. `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`). Only deps are **torch + numpy** (lzma/zlib are stdlib).
No extra installs needed; if numpy is somehow absent: `pip install numpy`.

**Files to transfer** (small — ~70KB total): `core_data.py`, `gpu_exp1_novelty.py`, `gpu_exp1b_ca2d.py`,
`gpu_exp2_qd.py`, `gpu_exp3_memory.py`. (Exp-1b imports Exp-1; Exp-2/Exp-3 are self-contained.)

```bash
# LOCAL (WSL), bundle the launch-ready code:
cd /home/joebachir20/math_lab
tar czf mathlab_gpu.tgz core_data.py gpu_exp1_novelty.py gpu_exp1b_ca2d.py gpu_exp2_qd.py gpu_exp3_memory.py runpod_plan.md
# transfer via runpodctl (one-time code) or scp to the pod, then on the POD:
mkdir -p ~/math_lab && cd ~/math_lab && tar xzf /path/to/mathlab_gpu.tgz && mkdir -p runs
# VERIFY env (must print cuda True + the 4090):
python -c "import torch,numpy; print('torch',torch.__version__,'cuda',torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# SMOKE every experiment (must all finish in seconds, no errors) BEFORE committing GPU-hours:
python gpu_exp1_novelty.py --smoke && python gpu_exp1b_ca2d.py --smoke && python gpu_exp2_qd.py --smoke && python gpu_exp3_memory.py --smoke
```

---

## 3. Experiments, in expected-value order

### Exp-1 — Residual/complex-finder at scale (THE MOONSHOT SWING) · ~2–8 GPU-hr
**Attacks:** SCALE wall (#10). **Goal first = moonshot.**
**What it does:** the project's *validated* edge-of-chaos complex-finder — `interest = 4c(1-c)·exp(-((damage-center)/w)²)`,
gated to not-named rules (dead/affine/short-period/chaotic/frozen) — run at GPU scale over spaces the 4060 can't sweep,
**plus a genuinely new axis**: a single neural next-row predictor SHARED across the whole population (shared weights can't
memorize per-rule local maps → only captures emergent regularity that generalizes across rules → a rule-agnostic
"higher-order structure" detector; `--neural`). Top survivors are saved as space-time `.npy` for **inspection**.

> NOTE on honesty: the raw "structured-but-not-named = compress_general − compress_named" residual was BUILT and then
> CORRECTED locally — pure subtraction rewards *maximal* compressibility (frozen/blinking rules), not complexity. The
> shipped score is the validated intermediate-compressibility finder + neural tilt. The moonshot lever is **scale +
> uncatalogued space + neural axis + human inspection**, NOT a brand-new selection mechanism. Don't overclaim.

```bash
# 1a. radius-2 (2^32), large, 3 seeds, neural axis ON — the cheap thorough sweep (~minutes each):
for s in 1 2 3; do python gpu_exp1_novelty.py --radius 2 --W 256 --T 256 --batch 32768 --pop 8192 \
   --gens 120 --neural --neural-steps 800 --seed $s --out runs/exp1_r2_s$s; done
# 1b. radius-3 (2^128, vast+SPARSE — needs the big seed batch; structured rules are needles):
python gpu_exp1_novelty.py --radius 3 --W 256 --T 256 --batch 16384 --pop 8192 --gens 200 --neural --out runs/exp1_r3
# 1c. 2D Life-like CENSUS (all 2^18 outer-totalistic rules — ROBUST, guaranteed structure, validates the signal in 2D):
python gpu_exp1b_ca2d.py --mode census --H 64 --W 64 --T 128 --batch 1024 --out runs/exp1b_census
# 1d. 2D non-totalistic Moore HUNT (2^512, chaos-dominated → MUST warm-start from Life-like stepping-stones):
python gpu_exp1b_ca2d.py --mode moore --warmstart --H 48 --W 48 --T 96 --batch 2048 --pop 512 --gens 60 --out runs/exp1b_moore
```
**Success / moonshot-hit:** a top survivor whose space-time, on inspection, shows a mechanism that looks **genuinely
unfamiliar** (novel particle/glider types, unusual scaling) = evidenceable-but-unprovable novelty → write it up carefully,
do NOT claim "proven novel." **Ceiling-holds (expected):** survivors are "just class-4 / Life-like" → still a clean result
(the complex-finder works at scale; the rediscovery ceiling holds in a new space).
**Kill:** if radius-3 / moore-hunt find zero in-band survivors after the big seed batch (all chaotic needles), stop that
sub-run — it's the landscape wall reappearing in CA space (itself a finding); fall back to radius-2 + 2D census.
**Inspect:** `runs/*/top_spacetime.npy` and `top2d_spacetime.npy` — render rows as an image / animate frames; compute
period, glider speeds, particle catalog. **Calibrate the 2D damage-center** against Conway's Life on the GPU first
(`--mode census` top list should include Life's neighborhood; nudge the `0.12` damage center in `novelty_scores` if not).

### Exp-2 — MAP-Elites vs the LANDSCAPE WALL (RELIABLE + maybe moonshot-adjacent) · ~8–30 GPU-hr (main consumer)
**Attacks:** LANDSCAPE wall (#4, the only hard wall marked "FIX: unknown").
**What it does:** GPU-batched n-state Turing-machine simulator (millions of TMs in parallel from a blank tape). Three
matched-budget conditions: **MAP-Elites** (archive over (tape-span, ones) niches, in-niche objective = runtime/depth) vs
**EVOLUTION** (single-objective depth hill-climb = the expEE baseline that stalled at 43) vs **SAMPLING** (floor). The
question: does quality-diversity's stepping-stone search **cross** the rugged Busy-Beaver needle-landscape where
hill-climbing stalls?

```bash
# primary: n=5 (where expEE stalled at 43; BB(5)=47,176,870), multiple seeds:
for s in 1 2 3; do python gpu_exp2_qd.py --n 5 --L 8192 --Tmax 50000 --batch 8192 --gens 3000 --grid 28 --seed $s --out runs/exp2_n5_s$s; done
# stretch: n=6 (uncharted; BB(6) is astronomically large):
python gpu_exp2_qd.py --n 6 --L 16384 --Tmax 100000 --batch 8192 --gens 4000 --grid 32 --out runs/exp2_n6
```
**Success (either is a real result):** MAP-Elites ≫ evolution (e.g. reaches 100s–1000s of steps where evolution plateaus)
= **QD CROSSES the landscape wall** (a movable wall, big finding). OR MAP-Elites ≈ evolution = **the landscape wall is
fundamental even for the method designed to beat it** (also a clean, publishable-grade wall result).
**Moonshot-adjacent:** if a *deep* halter also has *low-complexity structured* space-time (saved `deepest_tape.npy`),
inspect it. **Kill:** if even MAP-Elites coverage saturates and depth flatlines for >500 gens with no gain at n=5, stop and
record the stall (don't grind). **Perf gotcha (IMPORTANT):** the per-step loop is one Python iteration per TM step, so
**keep search-loop `--Tmax` ≤ ~50k** (high Tmax is launch-overhead-bound, not GPU-bound). For the deepest handful, do a
SEPARATE high-Tmax verify pass (re-run those genomes alone with `--Tmax` in the millions). Vary the descriptor if coverage
is poor (span/ones is a starting choice, not sacred).

### Exp-3 — Memory-augmented net vs the REPRESENTATIONAL WALL (RELIABLE) · ~3–12 GPU-hr
**Attacks:** REPRESENTATIONAL wall (#1, "FIX: richer substrate").
**What it does:** a tiny recurrent controller + external differentiable memory (stack/tape), vs a memory-LESS baseline,
on NON-regular ops (not finite-state). Train short, test exact length-generalization on long. **Local smoke already shows
the crossing on REVERSAL:** memory net w18 = **0.988** vs baseline w8 = 0.004. Multiplication is the hard stretch (both
fail at small scale → that's the GPU target: bigger memory + longer training + more seeds).

```bash
# clean demonstrator (reversal — memory length-generalizes, baseline collapses; PROVEN locally w18 0.988 vs 0.004):
python gpu_exp3_memory.py --arch both --op rev --steps 12000 --hidden 64 --seeds 3 \
  --train-width 1 2 3 4 5 --test-widths 3 5 8 12 16 20 --eval-n 1024
# the hard target (full multiplication — does richer memory + scale crack the non-finite-state op? OPEN):
python gpu_exp3_memory.py --arch both --op mul --steps 40000 --hidden 96 --mem_slots 40 --mem_width 12 --seeds 3 \
  --train-width 1 2 3 4 5 6 --test-widths 4 6 8 10 12 16 20 --eval-n 1024
```
*(`--train-width`/`--test-widths` are SPACE-separated lists. Results auto-append to `runs/exp3_results.txt`. mul is NOT
yet shown to length-generalize even with memory — at 4k steps it didn't fit in-distribution; the 40k-step run is the real
test. rev is the validated demonstration of the fix.)*
**Success:** memory net length-generalizes a non-regular op where baseline collapses = **richer substrate crosses wall #1**
(clean, expected for rev/Dyck; a genuine win if it cracks MUL). **Ceiling:** if even memory fails MUL length-gen, that
bounds the substrate fix (also informative). **Inspect:** if memory cracks an op, try to read the discrete stack/tape
program (the point of the project — an extractable procedure).

---

## 4. Budget ($75 ≈ 108 GPU-hr on the 4090; realistic spend far less)

| Exp | What | Est. GPU-hr | Priority |
|---|---|---|---|
| 1 | novelty/complex-finder 1D r2+r3, 2D census+moore, neural, 3 seeds | 2–8 | **moonshot — run first & thorough** |
| 2 | MAP-Elites QD n=5 (3 seeds) + n=6 stretch + deep-verify | 8–30 | **main consumer — reliable wall result** |
| 3 | memory-vs-baseline: rev/Dyck demos + mul stretch, 5 seeds | 3–12 | reliable wall result |
| — | inspection, re-runs, damage-center calibration, buffer | 5–15 | — |
| **Total** | | **~18–65 hr (~$12–45)** | **under $75 with headroom** |

Run order = **1c (2D census, fast, validates signal) → 1a (r2) → 2 (n5, the long one — launch and let it run) → 3
(rev+mul) → 1b/1d (r3, moore stretch) → 2 (n6 stretch) → inspection.** Launch Exp-2 n=5 early in the background since it's
the longest; work the others while it runs.

---

## 5. Launch checklist (the moment the GPU connects)

1. `nvidia-smi` → confirm RTX 4090, 24GB.
2. Transfer + extract files (§2), `mkdir -p runs`.
3. Run the env-verify + **all four `--smoke`** (§2). Any error → fix before spending hours. (All pass locally on the 4060.)
4. **Calibrate Exp-1b 2D damage-center**: run `gpu_exp1b_ca2d.py --mode census --H 48 --W 48 --T 96 --limit 20000` and
   check the top list is genuinely complex (intermediate gen_bpc ~0.4–0.6, damage in band). Nudge the `0.12` center if
   needed. (1D center `0.18` is already validated locally.)
5. **Launch Exp-2 n=5 (3 seeds) in the background** (`nohup ... &` or tmux) — it's the long pole.
6. While it runs: Exp-1 (1a/1c), then Exp-3 (rev then mul).
7. Pull `.npy`/`.json` outputs back locally for inspection; render/animate the top survivors and deepest TMs.
8. **Update TRACKER.md** per RULES — one honest entry per experiment, including the failures. Especially the failures.

---

## 6. How to judge results (and what to write in TRACKER)

- **Moonshot HIT** (rare): an Exp-1 survivor or Exp-2 deep machine shows a mechanism that, after honest inspection against
  known CA/TM phenomenology, looks genuinely unfamiliar. Write it as **evidenceable, NOT proven** novel (novelty is
  epistemically unprovable — wall #11). Save everything; characterize concretely (period, particles, scaling).
- **Wall result** (likely, still valuable): Exp-2 answers "can QD cross the landscape wall?" yes/no/partial — a clean
  result either way on a wall marked "FIX: unknown." Exp-3 demonstrates (or bounds) the richer-substrate fix to wall #1.
- **Ceiling holds** (expected for Exp-1): survivors are class-4/Life-like — the complex-finder works at scale but
  resurfaces known classes, consistent with the convergence-by-theorem story. Honest negative; no dressing-up.
- Discipline (RULES): exact verification; abandon stuck sub-runs (don't grind >3 tunings); separate
  discovered-from-outcome vs scaffolded; honesty over metrics.

---

## 7. Caveats / tuning knobs / gotchas (so you don't rediscover them on paid time)

- **VRAM scaling:** `run_ca` materializes (batch·T·W) bytes; (batch·H·W·T) in 2D. Recommended batches fit 24GB with
  margin (~≤8GB). If OOM: halve `--batch` or drop `--W/--T/--H`. `nonlinearity` is chunked (won't OOM at r3).
- **Exp-1 radius-3 & 2D-moore are CHAOS-DOMINATED:** random rules are ~all chaotic (damage ~0.5) → the gate rejects nearly
  everything. r3 needs a huge seed batch; moore NEEDS `--warmstart` (seeds from Life-like rules). This sparsity is itself
  the landscape wall showing up in CA space — note it if the hunts come up empty.
- **Exp-1 damage-center** (`0.18` 1D / `0.12` 2D in `novelty_scores`) is the one hand-tuned knob — calibrate vs known
  class-4/Life on the GPU; it sets which "complexity band" you hunt.
- **Exp-1 `--neural`** is the genuinely-new axis but the most fragile (a trained model). The lzma/zlib path is the solid
  default; treat the neural tilt as a bonus, not load-bearing.
- **Exp-2 Tmax** is launch-overhead-bound at high values — keep search Tmax ≤ ~50k, deep-verify the best separately.
- **Exp-2 descriptor** (span, ones) is a starting choice. If archive coverage is low or depth stalls, try alternatives
  (e.g. fraction-right-moves, distinct-transitions-used) — it's a knob, not sacred.
- **Compressor:** zlib in the search loop (fast), lzma for final survivor ranking (stronger) — already wired.
- The smoke configs exercise every code path; the scale configs are the SAME code with bigger numbers (no untested branch).
```
