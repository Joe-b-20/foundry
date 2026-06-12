#!/usr/bin/env bash
# Environment probe — no nested-quote traps.
set +e
echo "USER=$(whoami)  HOME=$HOME"
echo "--- python3 ---"
python3 --version
echo "--- conda/mamba/micromamba ---"
command -v conda mamba micromamba 2>/dev/null || echo "none on PATH"
ls -d "$HOME"/miniconda3 "$HOME"/miniforge3 "$HOME"/micromamba /opt/conda 2>/dev/null || echo "no conda dirs"
echo "--- pip ---"
python3 -m pip --version 2>/dev/null || echo "no pip"
echo "--- venv module ---"
python3 -c 'import venv; print("venv ok")' 2>/dev/null || echo "no venv"
echo "--- disk (home) ---"
df -h "$HOME" | tail -1
echo "--- cpu cores / mem ---"
nproc
free -h | sed -n '1,2p'
echo "--- nvidia-smi ---"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "no nvidia-smi"
echo "--- nvcc ---"
command -v nvcc >/dev/null && nvcc --version | tail -2 || echo "no nvcc (fine; pip wheels bundle CUDA)"
echo "--- existing torch? ---"
python3 -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())' 2>/dev/null || echo "no torch in system python"
