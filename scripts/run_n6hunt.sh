#!/usr/bin/env bash
# Targeted zeta(3)-class identity hunt (deg-3 x -n^6 family, Apery control in-grid).
# Chains after PHASE2.ALLDONE. Detached: nohup bash run_n6hunt.sh > runs/N6HUNT.out 2>&1 &
cd "$(dirname "$0")/.."
export MPMATH_NOGMPY=1
PY=python
while [ ! -f runs/PHASE2.ALLDONE ]; do sleep 60; done
echo "[n6hunt] phase2 finished, starting $(date -u +%H:%M:%S)"
run () { n=$1; shift; echo "[n6hunt] $n start $(date -u +%H:%M:%S)"; "$@" > runs/$n.log 2>&1; touch runs/$n.DONE; echo "[n6hunt] $n done $(date -u +%H:%M:%S)"; }

# stage 1: 111^4 = 151.8M PCFs (|c|<=55 — Apery [5,27,51,34] IN-grid) + Mobius prefilter
run n6_stage1 $PY -u src/gpu_pcf_hunt.py --stage1 --family n6 --crange 55 --terms 90 --out runs/pcf_n6
# stage 2a: exact PSLQ verify of ALL NEAR candidates (priority arm)
run n6_stage2_near $PY -u src/gpu_pcf_hunt.py --stage2 --procs 60 --dps 250 --out runs/pcf_n6
# stage 2b: blind-null arm (uniform sample of non-near survivors)
mkdir -p runs/pcf_n6_blind && cp runs/pcf_n6/stage1_blindsample.npz runs/pcf_n6_blind/stage1_survivors.npz
run n6_stage2_blind $PY -u src/gpu_pcf_hunt.py --stage2 --procs 60 --dps 250 --out runs/pcf_n6_blind

touch runs/N6HUNT.ALLDONE
echo "[n6hunt] ALL DONE $(date -u +%H:%M:%S)"
