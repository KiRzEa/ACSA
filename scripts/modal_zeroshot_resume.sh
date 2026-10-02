#!/usr/bin/env bash
# Resilient resume for the zeroshot_remaining_cheap_first phase: launches each remaining job as its OWN
# separate `modal run --detach --job "..."` invocation (fresh app/function per job), instead of one
# long-lived Python process looping over many .remote() calls -- the latter died unpredictably twice
# (Modal platform-side "function is stopped" error, cause unconfirmed) after only 2-12 jobs each time.
# Isolating each job to its own app means one job dying doesn't take the rest of the queue down with it.
#
# Tracks cumulative LOCAL wall-clock time against BUDGET_MIN and stops launching new jobs once exceeded
# (same cost-cap intent as modal_runner.py's --budget-hours, just at the shell level this time).
# PER_JOB_TIMEOUT_SEC bounds worst-case waste if a single job hangs, without being so short it kills a
# legitimately slow job (a real mistake made earlier in this session with a 280s timeout).
#
# Usage: BUDGET_MIN=95 bash scripts/modal_zeroshot_resume.sh
set -uo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/Library/Python/3.9/bin:$PATH"

BUDGET_MIN="${BUDGET_MIN:-95}"
PER_JOB_TIMEOUT_SEC="${PER_JOB_TIMEOUT_SEC:-1200}"

# macOS ships bash 3.2 (no `mapfile`, that's bash 4+) -- use a portable temp-file + while-read instead.
JOBLIST="$(mktemp)"
python3 -c "
import modal_runner as m
for k, s in m.ZEROSHOT_REMAINING_CHEAP_FIRST_PHASE:
    print(f'{k}:{s}')
" > "$JOBLIST"
n_jobs=$(wc -l < "$JOBLIST" | tr -d ' ')

echo "[$(date '+%H:%M:%S')] ${n_jobs} jobs queued, budget=${BUDGET_MIN}min, per-job timeout=${PER_JOB_TIMEOUT_SEC}s"
start=$(date +%s)
done_count=0
fail_count=0
while IFS= read -r job; do
  [ -z "$job" ] && continue
  elapsed_min=$(( ($(date +%s) - start) / 60 ))
  if [ "$elapsed_min" -ge "$BUDGET_MIN" ]; then
    echo "[$(date '+%H:%M:%S')] budget reached (${elapsed_min}min elapsed) -- stopping before ${job}"
    break
  fi
  echo "[$(date '+%H:%M:%S')] [${elapsed_min}min elapsed] running ${job} ..."
  # -k 15: if SIGTERM at PER_JOB_TIMEOUT_SEC doesn't kill the process (observed once: the modal CLI caught
  # SIGTERM to cancel the remote job cleanly, then hung ~4h retrying a dead gRPC connection instead of
  # exiting), force SIGKILL 15s later so the shell loop can never stall past one job's allotted time again.
  if timeout -k 15 "$PER_JOB_TIMEOUT_SEC" modal run --detach modal_runner.py --job "$job" > "/tmp/zs_$(echo "$job" | tr ':' '_').log" 2>&1; then
    echo "[$(date '+%H:%M:%S')]   OK: ${job}"
    done_count=$((done_count + 1))
  else
    echo "[$(date '+%H:%M:%S')]   FAILED: ${job} (see /tmp/zs_$(echo "$job" | tr ':' '_').log)"
    fail_count=$((fail_count + 1))
  fi
done < "$JOBLIST"
rm -f "$JOBLIST"
echo "[$(date '+%H:%M:%S')] finished: ${done_count} ok, ${fail_count} failed, $(( ($(date +%s) - start) / 60 ))min total"
