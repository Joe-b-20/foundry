#!/usr/bin/env bash
# Quick status check for env setup. No fragile inline quoting.
echo "=== setup.log (last 15 lines) ==="
tail -n 15 /home/joebachir20/math_lab/env/setup.log 2>/dev/null || echo "(no setup.log)"
echo "=== conda present? ==="
test -x "$HOME/miniforge3/bin/conda" && echo "yes: $HOME/miniforge3/bin/conda" || echo "no"
echo "=== setup processes ==="
pgrep -af 'setup_env|miniforge|conda|curl|pip' | grep -v pgrep || echo "none running"
