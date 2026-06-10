#!/usr/bin/env bash
# Idempotent environment setup for mathlab.
# Installs Miniforge (if absent), creates conda env `mathlab` (py3.11),
# installs torch (CUDA 12.4) + a few small utilities.
set -e
# Self-log so the job is pollable regardless of how it's launched.
LOGFILE="$(cd "$(dirname "$0")" && pwd)/setup.log"
exec > "$LOGFILE" 2>&1
LOG() { echo "[$(date +%H:%M:%S)] $*"; }

MF="$HOME/miniforge3"
CONDA="$MF/bin/conda"

if [ ! -x "$CONDA" ]; then
  LOG "Installing Miniforge to $MF ..."
  cd /tmp
  curl -fsSL -o miniforge.sh \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash miniforge.sh -b -p "$MF"
  LOG "Miniforge installed."
else
  LOG "Miniforge already present."
fi

# init conda for this shell
source "$MF/etc/profile.d/conda.sh"

if ! conda env list | grep -qE '^mathlab\s'; then
  LOG "Creating env mathlab (python=3.11) ..."
  conda create -n mathlab python=3.11 -y
else
  LOG "Env mathlab already exists."
fi

conda activate mathlab
LOG "Python: $(python --version)  at $(which python)"

LOG "Installing torch (cu124) ..."
pip install --upgrade pip >/dev/null
pip install torch --index-url https://download.pytorch.org/whl/cu124

LOG "Installing small utilities ..."
pip install numpy matplotlib tqdm

LOG "Verifying torch + CUDA ..."
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
    x = torch.randn(1024,1024, device="cuda")
    y = (x@x).sum().item()
    print("matmul ok, sum=", round(y,2))
PY
LOG "DONE."
