#!/usr/bin/env bash
# One scheduled job, the same way the GitHub Actions workflows run it:
#   sync the repo -> run the Python entry point -> publish the dashboard files.
#
#   deploy/run_job.sh cycle       # trading_bot.yml      -> python main.py
#   deploy/run_job.sh watch       # volume_watch.yml     -> python volume_watch.py
#   deploy/run_job.sh refresh     # refresh_positions.yml -> python position_metrics.py
#   deploy/run_job.sh screening   # weekly_screening.yml -> python screening.py
#
# Environment (from $REPO_DIR/.env, see deploy/.env.example):
#   PUBLISH=1   commit + push results so the GitHub Pages dashboard updates
#               (needs push access from this server; PUBLISH=0 keeps everything local)
set -euo pipefail

JOB="${1:?usage: run_job.sh cycle|watch|refresh|screening}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi
PUBLISH="${PUBLISH:-0}"
PYTHON="$REPO_DIR/.venv/bin/python"

# ONE lock for every job, stricter than the workflows' per-workflow concurrency
# groups: a watch tick can start a full trading cycle in-process, so separate
# locks would let a wake-up cycle and a scheduled cycle trade side by side --
# and every job shares this one git working tree and trading_bot.db. Holding
# the lock covers everything.
#   cycle, screening : wait up to 1h for the lock (a scheduled run must happen)
#   watch, refresh   : skip this tick if busy (the next one is 15 min away)
# The lock is released automatically when this process exits, however it exits.
case "$JOB" in
  cycle|screening) FLOCK_ARGS="-w 3600" ;;
  watch|refresh)   FLOCK_ARGS="-n" ;;
  *) echo "unknown job: $JOB" >&2; exit 2 ;;
esac
LOCK_FILE="${TRADINGBOT_LOCK:-$REPO_DIR/.tradingbot.lock}"
exec 9>"$LOCK_FILE"
# shellcheck disable=SC2086
if ! flock $FLOCK_ARGS 9; then
  echo "[$JOB] another TradingBot job holds $LOCK_FILE; skipping this run."
  exit 0
fi

if [ "$PUBLISH" = "1" ]; then
  # Pick up config/code changes pushed since the last run, like a fresh checkout.
  git pull --rebase --quiet || { echo "[$JOB] git pull failed; running on the local copy."; }
fi

JOB_RC=0
case "$JOB" in
  cycle)
    # A failed cycle is still recorded and published (docs/job_status.json), so
    # the dashboard shows the failure instead of quietly going stale. The real
    # exit code is returned at the very end, so systemd still sees the failure.
    STARTED="$(date -u +%Y-%m-%dT%H:%M:%S+00:00)"
    RUN_LOG="$(mktemp)"
    STAGE=pytest
    set +e
    "$PYTHON" -m pytest -q 2>&1 | tee "$RUN_LOG"
    JOB_RC=${PIPESTATUS[0]}
    if [ "$JOB_RC" -eq 0 ]; then
      STAGE=cycle
      "$PYTHON" main.py --config config.yaml 2>&1 | tee "$RUN_LOG"
      JOB_RC=${PIPESTATUS[0]}
    fi
    set -e
    "$PYTHON" job_status.py record --job cycle --started "$STARTED" --rc "$JOB_RC" \
      --stage "$STAGE" --log "$RUN_LOG" || echo "[$JOB] could not record job status."
    rm -f "$RUN_LOG"
    FILES="docs/job_status.json docs/index.html docs/signals.csv docs/config.yaml signals.csv trading_bot.db"
    ;;
  watch)
    "$PYTHON" volume_watch.py --config config.yaml
    FILES="wake_state.json docs/index.html docs/signals.csv docs/config.yaml signals.csv trading_bot.db"
    ;;
  refresh)
    "$PYTHON" position_metrics.py --config config.yaml
    FILES=""
    ;;
  screening)
    "$PYTHON" -m pytest -q
    "$PYTHON" screening.py --output symbols.yaml
    FILES="symbols.yaml"
    ;;
esac

# ---------------------------------------------------------------- publish
mkdir -p docs
if [ "$JOB" = "cycle" ] || [ "$JOB" = "watch" ]; then
  cp dashboard.html docs/index.html
  [ -f signals.csv ] && cp signals.csv docs/signals.csv || true
  cp config.yaml docs/config.yaml
fi
for f in positions.json benchmark.json; do
  if [ -f "$f" ]; then
    cp "$f" "docs/$f"
    FILES="$FILES $f docs/$f"
  fi
done

[ "$PUBLISH" = "1" ] || { echo "[$JOB] done (PUBLISH=0, nothing pushed)."; exit "$JOB_RC"; }

# shellcheck disable=SC2086
git add -f $FILES 2>/dev/null || true
if git diff --staged --quiet; then
  echo "[$JOB] nothing changed."
  exit "$JOB_RC"
fi
git commit --quiet -m "chore: $JOB results (vps) [skip ci]"

for attempt in 1 2 3; do
  if git push --quiet; then
    echo "[$JOB] pushed on attempt $attempt."
    exit "$JOB_RC"
  fi
  echo "[$JOB] push rejected (attempt $attempt/3); rebasing."
  git pull --rebase --quiet || { git rebase --abort 2>/dev/null || true; exit 1; }
  sleep 5
done
echo "[$JOB] push failed after 3 attempts; results are committed locally only." >&2
exit 1
