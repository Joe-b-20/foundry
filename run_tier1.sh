#!/bin/bash
# TIER 1 — cheap, high-confidence results, banked first so we can INSPECT the signal at scale before spending on Tier 2.
# Launch DETACHED:  nohup bash /root/math_lab/run_tier1.sh >/root/math_lab/runs/TIER1.out 2>&1 &
# Survives SSH drops AND the chat session ending. Poll via runs/TIER1.log and the per-exp runs/<name>.DONE markers.
set -u
cd /root/math_lab
PY=$(command -v python || command -v python3)
mkdir -p runs
LOG=runs/TIER1.log
say(){ echo "$(date -u +%H:%M:%S) $*" | tee -a "$LOG"; }
run(){ local n="$1"; shift; say ">>> START $n"; "$@" >"runs/$n.log" 2>&1; say ">>> END $n rc=$?"; touch "runs/$n.DONE"; }

say "TIER1 START on $(hostname) — GPU $($PY -c 'import torch;print(torch.cuda.get_device_name(0))')"
# 1. moonshot in 2D: exhaustive Life-like census (validates the edge-of-chaos signal in 2D; saves warm-start material)
run exp1b_census $PY -u gpu_exp1b_ca2d.py --mode census --H 48 --W 48 --T 96 --batch 1024 --out runs/exp1b_census
# 2. moonshot in 1D: radius-2 complex-finder at scale, 1 seed, neural axis ON (the new higher-order-structure signal)
run exp1_r2_s1   $PY -u gpu_exp1_novelty.py --radius 2 --W 256 --T 256 --batch 32768 --pop 8192 --gens 100 --neural --neural-steps 800 --seed 1 --out runs/exp1_r2_s1
# 3. representational wall: reversal — the validated memory-vs-baseline length-gen crossing
run exp3_rev     $PY -u gpu_exp3_memory.py --arch both --op rev --steps 12000 --hidden 64 --seeds 3 --train-width 1 2 3 4 5 --test-widths 3 5 8 12 16 20 --eval-n 1024
say "TIER1 ALL-DONE"
touch runs/TIER1.ALLDONE
