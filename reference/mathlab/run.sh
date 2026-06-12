#!/usr/bin/env bash
# Run a python file inside the mathlab conda env, from the project dir.
# Usage: bash run.sh <script.py> [args...]
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate mathlab
cd /home/joebachir20/math_lab
exec python -u "$@"   # -u: unbuffered, so background runs show live progress
