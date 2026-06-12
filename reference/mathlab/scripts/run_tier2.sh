#!/bin/bash
# TIER 2 — the longer runs (more moonshot seeds, the hard mul stretch, and the long-pole landscape-wall QD).
# Launch DETACHED only AFTER inspecting Tier 1:
#   nohup bash /root/run_tier2.sh >/root/math_lab/runs/TIER2.out 2>&1 </dev/null &
set -u
cd /root/math_lab
PY=$(command -v python || command -v python3)
mkdir -p runs
LOG=runs/TIER2.log
say(){ echo "$(date -u +%H:%M:%S) $*" | tee -a "$LOG"; }
run(){ local n="$1"; shift; say ">>> START $n"; "$@" >"runs/$n.log" 2>&1; say ">>> END $n rc=$?"; touch "runs/$n.DONE"; }

# Self-chaining: wait for Tier 1 to finish before grabbing the GPU (so this can be launched NOW and runs unattended).
say "TIER2 queued — waiting for TIER1.ALLDONE..."
while [ ! -f runs/TIER1.ALLDONE ]; do sleep 15; done
say "TIER2 START on $(hostname)"
# moonshot 1D radius-2, all 3 seeds (s1 re-run here after the Tier-1 neural-OOM fix) — seed-robustness of complex survivors
run exp1_r2_s1 $PY -u src/gpu_exp1_novelty.py --radius 2 --W 256 --T 256 --batch 32768 --pop 8192 --gens 100 --neural --neural-steps 800 --seed 1 --out runs/exp1_r2_s1
run exp1_r2_s2 $PY -u src/gpu_exp1_novelty.py --radius 2 --W 256 --T 256 --batch 32768 --pop 8192 --gens 100 --neural --neural-steps 800 --seed 2 --out runs/exp1_r2_s2
run exp1_r2_s3 $PY -u src/gpu_exp1_novelty.py --radius 2 --W 256 --T 256 --batch 32768 --pop 8192 --gens 100 --neural --neural-steps 800 --seed 3 --out runs/exp1_r2_s3
# the hard representational stretch: does richer memory + scale crack full multiplication length-gen?
run exp3_mul   $PY -u src/gpu_exp3_memory.py --arch both --op mul --steps 40000 --hidden 96 --mem_slots 40 --mem_width 12 --seeds 3 --train-width 1 2 3 4 5 6 --test-widths 4 6 8 10 12 16 20 --eval-n 1024
# the long pole: MAP-Elites QD vs the landscape wall. RESIZED from the live probe (sampling 309k machines = 150s, so the
# original gens=2000/batch=8192 = hours/seed). Feasible sizing: ~40 min/seed. (Probe seed 7 already gives a 3rd datapoint.)
run exp2_n5_s1 $PY -u src/gpu_exp2_qd.py --n 5 --L 8192 --Tmax 30000 --batch 4096 --gens 300 --grid 26 --seed 1 --out runs/exp2_n5_s1
run exp2_n5_s2 $PY -u src/gpu_exp2_qd.py --n 5 --L 8192 --Tmax 30000 --batch 4096 --gens 300 --grid 26 --seed 2 --out runs/exp2_n5_s2
say "TIER2 ALL-DONE"
touch runs/TIER2.ALLDONE
