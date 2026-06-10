#!/usr/bin/env bash
# Phase-2 autonomous experiments (post-audit directions). Detached on the pod:
#   nohup bash run_phase2.sh > runs/PHASE2.out 2>&1 &
# Ordered by value-if-interrupted: pcf (identity hunt) > depthstruct > loop sweep.
cd "$(dirname "$0")"
export MPMATH_NOGMPY=1
PY=python
mkdir -p runs
run () { n=$1; shift; echo "[phase2] $n start $(date -u +%H:%M:%S)"; "$@" > runs/$n.log 2>&1; touch runs/$n.DONE; echo "[phase2] $n done $(date -u +%H:%M:%S)"; }

# === 2c: IDENTITY HUNT (highest variance/payoff, self-contained) ===
# Stage 1 on GPU (large PCF grid), then Stage 2 PSLQ verify on 60 CPU cores.
run pcf_stage1 $PY -u gpu_pcf_hunt.py --stage1 --crange 6 --terms 90 --out runs/pcf_main
run pcf_stage2 $PY -u gpu_pcf_hunt.py --stage2 --procs 60 --dps 250 --limit 400000 --out runs/pcf_main

# === 2a: DEPTH-WHILE-STRUCTURED — RUN LOCALLY on the 4060 (frees pod budget); see
#         run_dstruct_local.sh. Not run here on the pod. ===

# === 2b: LOOP-SUBSTRATE NAMING-DENSITY SWEEP (depth vs nearest-named) ===
# matched N/gens; only maxit (reachable depth) varies. 2 seeds x 3 depths.
for seed in 1 2; do
  for mit in 1 4 16; do
    run loop_m${mit}_s${seed} $PY -u gpu_avida_loop.py --maxit $mit --N 6144 --P 24 --gens 600 --seed $seed --out runs/loop_m${mit}_s${seed}
  done
done

touch runs/PHASE2.ALLDONE
echo "[phase2] ALL DONE $(date -u +%H:%M:%S)"
