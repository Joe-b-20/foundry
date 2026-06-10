#!/usr/bin/env bash
# Poll the pod for closure/phase progress. Exits when the marker file appears,
# or after MAXMIN minutes — either way the harness re-invokes the agent.
# Usage: bash poll_pod.sh <marker-file-relative-to-/workspace/math_lab/> <MAXMIN>
MARKER=${1:-runs/CLOSURES.ALLDONE}
MAXMIN=${2:-20}
POD="root@213.181.111.2"
PORT=42706
KEY=~/.ssh/id_ed25519
SSHO="-o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new -o BatchMode=yes"
end=$(( $(date +%s) + MAXMIN*60 ))
while [ "$(date +%s)" -lt "$end" ]; do
    out=$(ssh $SSHO -p $PORT -i $KEY $POD "ls /workspace/math_lab/$MARKER 2>/dev/null; ls /workspace/math_lab/runs/*.DONE 2>/dev/null | wc -l" 2>/dev/null)
    echo "[poll $(date +%H:%M:%S)] $out" | tr '\n' ' '; echo
    if echo "$out" | grep -q "$MARKER"; then
        echo "MARKER_FOUND: $MARKER"
        exit 0
    fi
    sleep 120
done
echo "POLL_TIMEOUT after ${MAXMIN}m (marker $MARKER not yet present)"
