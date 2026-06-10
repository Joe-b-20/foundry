#!/usr/bin/env bash
# Phase-3 G1: GENERAL PCF sweep on the pod — GPU stage 1 ONLY (stage-2 PSLQ runs on
# the faster LOCAL CPUs per Joe's routing note). ~1e10 PCFs, dual in-grid controls,
# delta-scoring. Detached: nohup bash run_gen6.sh > runs/GEN6.out 2>&1 &
cd "$(dirname "$0")"
export MPMATH_NOGMPY=1
echo "[gen6] stage1 start $(date -u +%H:%M:%S)"
python -u gpu_pcf_hunt.py --stage1 --family gen6 --arange 9 --brange 2 --terms 90 \
    --near-thresh 1e-11 --out runs/pcf_gen6 > runs/gen6_stage1.log 2>&1
touch runs/gen6_stage1.DONE
echo "[gen6] stage1 done $(date -u +%H:%M:%S)"
touch runs/GEN6.ALLDONE
