#!/usr/bin/env bash
# Post-audit closure runs (2026-06-09) — see consolidation/09_fable5_audit.md Part 5.
# Pod usage (detached, survives SSH drops — the session-10 pattern):
#   nohup bash run_closures.sh > runs/CLOSURES.out 2>&1 &
# Poll: ls runs/*.DONE ; done when runs/CLOSURES.ALLDONE exists.
cd "$(dirname "$0")"
PY=python
mkdir -p runs
run () { n=$1; shift; echo "[closures] $n start $(date)"; "$@" > runs/$n.log 2>&1; touch runs/$n.DONE; echo "[closures] $n done $(date)"; }

# --- Closure A: exp2 with the FIXED MAP-Elites archive (audit §1.2) ---
# A1: 3 seeds at the original config (Tmax 8000) — seed robustness with a valid archive
run exp2fix_s1 $PY -u gpu_exp2_qd.py --n 5 --L 8192 --Tmax 8000 --batch 4096 --gens 200 --grid 26 --seed 1 --desc span_ones --out runs/exp2fix_s1
run exp2fix_s2 $PY -u gpu_exp2_qd.py --n 5 --L 8192 --Tmax 8000 --batch 4096 --gens 200 --grid 26 --seed 2 --desc span_ones --out runs/exp2fix_s2
run exp2fix_s3 $PY -u gpu_exp2_qd.py --n 5 --L 8192 --Tmax 8000 --batch 4096 --gens 200 --grid 26 --seed 3 --desc span_ones --out runs/exp2fix_s3
# A2: descriptor-confound test — depth-aligned descriptor, 1 seed
run exp2fix_rtspan_s1 $PY -u gpu_exp2_qd.py --n 5 --L 8192 --Tmax 8000 --batch 4096 --gens 200 --grid 26 --seed 1 --desc rt_span --out runs/exp2fix_rtspan_s1
# A3: cap test — Tmax 30000 (does evolution keep climbing past the old 8000 cap?)
run exp2fix_T30k_s1 $PY -u gpu_exp2_qd.py --n 5 --L 8192 --Tmax 30000 --batch 4096 --gens 300 --grid 26 --seed 1 --desc span_ones --out runs/exp2fix_T30k_s1

# --- Closure B: avida_oe with waypoint evidence (audit §1.3) ---
# fine snapshots (every 10 gens), top-5 described, ~60-entry named suite on 64 probes,
# first_match table persisted — the evidence the original a+b/a^b claim lacked.
run oe_fix_s1 $PY -u gpu_avida_oe.py --N 12288 --P 28 --gens 1000 --snap 10 --seed 1 --out runs/oe_fix_s1
run oe_fix_s7 $PY -u gpu_avida_oe.py --N 12288 --P 28 --gens 1000 --snap 10 --seed 7 --out runs/oe_fix_s7
run oe_fix_s3 $PY -u gpu_avida_oe.py --N 12288 --P 28 --gens 1000 --snap 10 --seed 3 --out runs/oe_fix_s3

touch runs/CLOSURES.ALLDONE
echo "[closures] ALL DONE $(date)"
