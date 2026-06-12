#!/usr/bin/env bash
# Full local re-mine (phase 3 tier-2): all pulled survivor sets, quadratic relations +
# extended battery. Local 20-core box. nohup bash run_quadmine_local.sh > runs/QUADMINE.out 2>&1 &
cd "$(dirname "$0")/.."
export MPMATH_NOGMPY=1
echo "[quadmine] start $(date +%H:%M:%S)"
bash run.sh src/pcf_quadmine.py \
  --inputs runs_pod/phase2/pcf_n6/stage1_survivors.npz \
           runs_pod/phase2/pcf_n6big/stage1_survivors.npz \
           runs_pod/phase2/pcf_n6/stage1_blindsample.npz \
           runs_pod/phase2/pcf_p4/stage1_survivors.npz \
           runs_pod/phase2/pcf_main/stage1_survivors.npz \
  --procs 18 --dps 220 --limit 120000 --out runs/quadmine \
  > runs/quadmine_full.log 2>&1
touch runs/QUADMINE.DONE
echo "[quadmine] done $(date +%H:%M:%S)"
