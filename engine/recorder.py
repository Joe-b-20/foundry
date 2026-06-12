"""Flight recorder v0: one JSONL file per run, append-only.

Every line: schema, ts, run id, seq, actor, action, then optional payload /
reason / outcome. Big objects belong in files next to the log, referenced by
path or hash. The first event of any run is the RunSpec — predeclaration:
budgets and cost rules are on the record BEFORE any search happens.

Design test (RULES.md): every decision must be reconstructible from this log
alone. These logs are the future operator-model's training data; failures get
logged at the same fidelity as wins.
"""

import json
import time
from pathlib import Path

SCHEMA = "fr-v0"


class Recorder:
    def __init__(self, run_dir, run_id):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.seq = 0
        self._fh = open(self.run_dir / "events.jsonl", "a", encoding="utf-8")

    def event(self, actor, action, payload=None, reason=None, outcome=None):
        line = {"schema": SCHEMA, "ts": round(time.time(), 3),
                "run": self.run_id, "seq": self.seq,
                "actor": actor, "action": action}
        if payload is not None:
            line["payload"] = payload
        if reason is not None:
            line["reason"] = reason
        if outcome is not None:
            line["outcome"] = outcome
        self._fh.write(json.dumps(line) + "\n")
        self._fh.flush()
        self.seq += 1

    def close(self):
        self._fh.close()


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        rec = Recorder(Path(d) / "r1", "r1")
        rec.event("operator", "decision", payload={"x": 1},
                  reason="because sanity", outcome="ok")
        rec.event("judge", "evaluate", payload={"score": [0, 1]})
        rec.close()
        lines = (Path(d) / "r1" / "events.jsonl").read_text().splitlines()
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first["seq"] == 0 and first["reason"] == "because sanity"
    print("recorder v0 ok")
