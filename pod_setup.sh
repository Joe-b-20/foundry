#!/bin/bash
# Pod-side setup: extract project, verify env, run all 4 smokes. Run INTERACTIVELY (fast, ~1 min).
set -u
cd /root
PY=$(command -v python || command -v python3)
echo "PYTHON=$PY"
mkdir -p /root/math_lab
tar xzf /root/mathlab_gpu.tgz -C /root/math_lab
cd /root/math_lab
mkdir -p runs
echo "=== torch/env ==="
$PY -c "import torch,numpy; print('torch',torch.__version__,'cuda',torch.cuda.is_available(), (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOGPU'),'| numpy',numpy.__version__)"
echo "=== smokes (each must say SMOKE OK) ==="
for e in gpu_exp1_novelty.py gpu_exp1b_ca2d.py gpu_exp2_qd.py gpu_exp3_memory.py; do
  if $PY $e --smoke >runs/smoke_$e.log 2>&1; then echo "SMOKE OK   $e"; else echo "SMOKE FAIL $e"; tail -6 runs/smoke_$e.log; fi
done
echo "=== setup complete ==="
