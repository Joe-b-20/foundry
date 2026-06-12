#!/bin/bash
# TIER 2b — corrected continuation after resizing the QD runs (the original gens=2000/batch=8192 would be ~20 GPU-hr).
# Waits for the orphaned exp1_r2_s1 to finish (so we don't double-book the GPU), then runs the rest.
#   nohup bash /root/run_tier2b.sh >/root/math_lab/runs/TIER2b.out 2>&1 </dev/null &
set -u
cd /root/math_lab
PY=$(command -v python || command -v python3)
LOG=runs/TIER2.log
say(){ echo "$(date -u +%H:%M:%S) $*" | tee -a "$LOG"; }
run(){ local n="$1"; shift; say ">>> START $n"; "$@" >"runs/$n.log" 2>&1; say ">>> END $n rc=$?"; touch "runs/$n.DONE"; }

say "TIER2b waiting for orphaned exp1_r2_s1 to finish..."
while pgrep -f exp1_r2_s1 >/dev/null; do sleep 10; done
touch runs/exp1_r2_s1.DONE
say "TIER2b START on $(hostname)"
run exp1_r2_s2 $PY -u src/gpu_exp1_novelty.py --radius 2 --W 256 --T 256 --batch 32768 --pop 8192 --gens 100 --neural --neural-steps 800 --seed 2 --out runs/exp1_r2_s2
run exp1_r2_s3 $PY -u src/gpu_exp1_novelty.py --radius 2 --W 256 --T 256 --batch 32768 --pop 8192 --gens 100 --neural --neural-steps 800 --seed 3 --out runs/exp1_r2_s3
run exp3_mul   $PY -u src/gpu_exp3_memory.py --arch both --op mul --steps 40000 --hidden 96 --mem_slots 40 --mem_width 12 --seeds 3 --train-width 1 2 3 4 5 6 --test-widths 4 6 8 10 12 16 20 --eval-n 1024
run exp2_n5_s1 $PY -u src/gpu_exp2_qd.py --n 5 --L 8192 --Tmax 30000 --batch 4096 --gens 300 --grid 26 --seed 1 --out runs/exp2_n5_s1
run exp2_n5_s2 $PY -u src/gpu_exp2_qd.py --n 5 --L 8192 --Tmax 30000 --batch 4096 --gens 300 --grid 26 --seed 2 --out runs/exp2_n5_s2
say "TIER2 ALL-DONE"
touch runs/TIER2.ALLDONE
