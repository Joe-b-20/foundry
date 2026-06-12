#!/usr/bin/env bash
# Wait until the 4-order training run has saved all 4 checkpoints (or timeout), then report.
OUT="/mnt/c/Users/joeba/AppData/Local/Temp/claude/--wsl-localhost-ubuntu-home-joebachir20-math-lab/7d64a8b8-b7b7-4139-80bd-0b14fd201d72/tasks/bagr5jzv2.output"
for i in $(seq 1 20); do
  n=$(grep -c "saved runs" "$OUT" 2>/dev/null || echo 0)
  if [ "$n" -ge 4 ]; then echo "ALL 4 DONE"; break; fi
  sleep 30
done
echo "=== progress ==="
grep -E "# ORDER|after phase 4|mul  |saved runs" "$OUT" 2>/dev/null | tail -16
echo "=== ckpts ==="
ls -1t /home/joebachir20/math_lab/runs/expH_order_*.pt 2>/dev/null
