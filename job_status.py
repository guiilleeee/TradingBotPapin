"""Run history for the dashboard's "Cicles programats" panel (docs/job_status.json).

deploy/run_job.sh calls `record` after every VPS cycle -- including a failed one,
so a failure is visible on the page instead of the page silently going stale.
The dashboard matches each run to a scheduled slot by its start time.

    python job_status.py record --job cycle --started <iso> --rc 0 --stage cycle --log run.log
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from secrets_redaction import sanitize

DEFAULT_PATH = os.path.join("docs", "job_status.json")
MAX_RUNS = 60
MAX_ERROR_CHARS = 300


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def last_error_line(log_text: str) -> str:
    """The last non-empty line of a failed run's output: pytest's summary
    ("2 failed, 400 passed") or a traceback's final "ValueError: ...". When pytest
    listed failing tests ("FAILED tests/x.py::t - msg"), their ids follow the
    summary, so the dashboard says which test broke the gate, not just how many."""
    lines = (log_text or "").splitlines()
    summary = ""
    for line in reversed(lines):
        line = line.strip().strip("=").strip()
        if line:
            summary = line
            break
    failed = [ln.strip()[len("FAILED "):].split(" - ", 1)[0] for ln in lines
              if ln.strip().startswith("FAILED ")]
    if failed:
        summary = f"{summary} :: {', '.join(failed)}"
    return sanitize(summary)[:MAX_ERROR_CHARS]


def load_runs(path: str) -> List[Dict[str, Any]]:
    try:
        with open(path, encoding="utf-8") as handle:
            runs = json.load(handle).get("runs", [])
    except (OSError, ValueError, AttributeError):
        return []
    return [r for r in runs if isinstance(r, dict)] if isinstance(runs, list) else []


def record_run(
    path: str,
    job: str,
    started_at: str,
    rc: int,
    stage: str,
    log_text: str = "",
    finished_at: Optional[str] = None,
) -> Dict[str, Any]:
    run = {
        "job": job,
        "started_at": started_at,
        "finished_at": finished_at or utc_now_iso(),
        "status": "ok" if rc == 0 else "failed",
        "stage": stage,
        "error": "" if rc == 0 else last_error_line(log_text),
    }
    runs = (load_runs(path) + [run])[-MAX_RUNS:]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"generated_at": utc_now_iso(), "runs": runs}, handle, indent=2)
    return run


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("record")
    rec.add_argument("--job", required=True)
    rec.add_argument("--started", required=True)
    rec.add_argument("--rc", type=int, required=True)
    rec.add_argument("--stage", required=True)
    rec.add_argument("--log", default=None, help="file holding the run's output")
    rec.add_argument("--path", default=DEFAULT_PATH)
    args = parser.parse_args(argv)

    log_text = ""
    if args.log:
        try:
            with open(args.log, encoding="utf-8", errors="replace") as handle:
                log_text = handle.read()
        except OSError:
            pass
    run = record_run(args.path, args.job, args.started, args.rc, args.stage, log_text)
    print(f"[job_status] {run['job']}: {run['status']} ({run['stage']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
