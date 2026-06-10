#!/usr/bin/env bash
# Depth-while-structured, run on the LOCAL 4060 in parallel with the pod (frees pod
# budget for pcf + loop). Both seeds, 3 conditions each.
cd "$(dirname "$0")/.."
for seed in 1 2; do
  echo "[dstruct-local] seed $seed start $(date +%H:%M:%S)"
  bash run.sh src/gpu_depthstruct.py --n 5 --L 8192 --Tmax 30000 --batch 4096 --gens 100 \
       --seed $seed --out runs/dstruct_s$seed > runs/dstruct_s$seed.log 2>&1
  touch runs/dstruct_s$seed.DONE
  echo "[dstruct-local] seed $seed done $(date +%H:%M:%S)"
done
touch runs/DSTRUCT_LOCAL.ALLDONE
echo "[dstruct-local] ALL DONE $(date +%H:%M:%S)"
