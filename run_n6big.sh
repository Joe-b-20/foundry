#!/usr/bin/env bash
# BIG zeta(3)-class sweep (STREAMING-prefilter version: constant host memory).
# p4 already done in the prior run; this is the n6 |c|<=120 push past the published
# Ramanujan-Machine coefficient region. Detached: nohup bash run_n6big.sh > runs/N6BIG.out 2>&1 &
cd "$(dirname "$0")"
export MPMATH_NOGMPY=1
PY=python
run () { n=$1; shift; echo "[n6big] $n start $(date -u +%H:%M:%S)"; "$@" > runs/$n.log 2>&1; touch runs/$n.DONE; echo "[n6big] $n done $(date -u +%H:%M:%S)"; }

# -n^6 family at |c|<=120 (241^4 = 3.37e9 PCFs, ~30 min; streaming Mobius prefilter)
run n6big_stage1 $PY -u gpu_pcf_hunt.py --stage1 --family n6 --crange 120 --terms 90 --out runs/pcf_n6big
run n6big_stage2 $PY -u gpu_pcf_hunt.py --stage2 --procs 60 --dps 250 --out runs/pcf_n6big
mkdir -p runs/pcf_n6big_blind && cp runs/pcf_n6big/stage1_blindsample.npz runs/pcf_n6big_blind/stage1_survivors.npz
run n6big_stage2_blind $PY -u gpu_pcf_hunt.py --stage2 --procs 60 --dps 250 --out runs/pcf_n6big_blind

touch runs/N6BIG.ALLDONE
echo "[n6big] ALL DONE $(date -u +%H:%M:%S)"
