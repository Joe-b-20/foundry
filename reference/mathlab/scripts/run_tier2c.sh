#!/bin/bash
# TIER 2c — final continuation. The QD result is already in hand (separate fast run: evo 6238 >> ME 73 >> sampling 49),
# so DROP the redundant launch-bound QD runs and finish only the valuable remaining work: 1D seed 3 + the multiplication
# stretch. Ends with CAMPAIGN.ALLDONE so the pod has a clean stop signal.
#   nohup bash /root/run_tier2c.sh >/root/math_lab/runs/TIER2c.out 2>&1 </dev/null &
set -u
cd /root/math_lab
PY=$(command -v python || command -v python3)
LOG=runs/TIER2.log
say(){ echo "$(date -u +%H:%M:%S) $*" | tee -a "$LOG"; }
run(){ local n="$1"; shift; say ">>> START $n"; "$@" >"runs/$n.log" 2>&1; say ">>> END $n rc=$?"; touch "runs/$n.DONE"; }

say "TIER2c waiting for orphaned exp1_r2_s2 to finish..."
while pgrep -f exp1_r2_s2 >/dev/null; do sleep 10; done
touch runs/exp1_r2_s2.DONE
say "TIER2c START on $(hostname)"
run exp1_r2_s3 $PY -u src/gpu_exp1_novelty.py --radius 2 --W 256 --T 256 --batch 32768 --pop 8192 --gens 100 --neural --neural-steps 800 --seed 3 --out runs/exp1_r2_s3
run exp3_mul   $PY -u src/gpu_exp3_memory.py --arch both --op mul --steps 40000 --hidden 96 --mem_slots 40 --mem_width 12 --seeds 3 --train-width 1 2 3 4 5 6 --test-widths 4 6 8 10 12 16 20 --eval-n 1024
say "CAMPAIGN ALL-DONE — safe to terminate the pod"
touch runs/CAMPAIGN.ALLDONE
