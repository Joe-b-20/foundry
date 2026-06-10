#!/usr/bin/env bash
# BIG zeta(3)-class sweep: push past the Ramanujan-Machine published coefficient
# region (|c|<=120 -> 241^4 = 3.37e9 PCFs, ~35 min at measured 1.9M/s) + the -n^4
# family probe. Detached: nohup bash run_n6big.sh > runs/N6BIG.out 2>&1 &
cd "$(dirname "$0")"
export MPMATH_NOGMPY=1
PY=python
run () { n=$1; shift; echo "[n6big] $n start $(date -u +%H:%M:%S)"; "$@" > runs/$n.log 2>&1; touch runs/$n.DONE; echo "[n6big] $n done $(date -u +%H:%M:%S)"; }

# 1) +n^4 family at |c|<=55 (cheap probe; Apery zeta(2) control [3,11,11,0] in-grid)
run p4_stage1 $PY -u gpu_pcf_hunt.py --stage1 --family p4 --crange 55 --terms 90 --out runs/pcf_p4
run p4_stage2 $PY -u gpu_pcf_hunt.py --stage2 --procs 60 --dps 250 --out runs/pcf_p4

# 2) -n^6 family at |c|<=120 (the big push past the published region)
run n6big_stage1 $PY -u gpu_pcf_hunt.py --stage1 --family n6 --crange 120 --terms 90 --out runs/pcf_n6big
run n6big_stage2 $PY -u gpu_pcf_hunt.py --stage2 --procs 60 --dps 250 --out runs/pcf_n6big
mkdir -p runs/pcf_n6big_blind && cp runs/pcf_n6big/stage1_blindsample.npz runs/pcf_n6big_blind/stage1_survivors.npz
run n6big_stage2_blind $PY -u gpu_pcf_hunt.py --stage2 --procs 60 --dps 250 --out runs/pcf_n6big_blind

touch runs/N6BIG.ALLDONE
echo "[n6big] ALL DONE $(date -u +%H:%M:%S)"
