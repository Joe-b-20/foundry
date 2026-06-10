#!/usr/bin/env bash
# Unified mul+div self-discovery across seeds (robustness check). Sequential to avoid CPU contention.
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate mathlab
cd /home/joebachir20/math_lab
for s in 0 1 2; do
  echo "===== SEED $s ====="
  python -u expJ_selfdiscover.py --ops mul,div --iters 160 --seed "$s" --save "runs/expJ_both_s$s.pt"
done
