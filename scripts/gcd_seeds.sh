#!/usr/bin/env bash
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate mathlab
cd /home/joebachir20/math_lab
for s in 1 2 3; do
  echo "===== SEED $s ====="
  python -u src/expK_gcd.py --iters 90 --seed "$s" | grep -E "chose|discovered policy of the|discovered policy:|w1:1.000  w2:1.000  w4"
done
