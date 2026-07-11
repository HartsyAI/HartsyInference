#!/usr/bin/env bash
# Minimal process supervisor for environments without systemd (containers without an init system, ad-hoc
# VMs, local testing of the restart behavior itself). Restarts the server whenever it exits, with a short
# backoff and a crash-loop breaker — same rationale as deploy/systemd/hartsyinference-server.service's
# Restart=always: some native/unsafe code paths in this engine can hit a corrupted-state exception (e.g.
# AccessViolationException) that .NET Core cannot catch in-process, so process-level restart is the actual
# mitigation for that class of failure, not a workaround for something fixable in code.
#
# Usage: deploy/run-with-restart.sh /path/to/HartsyInference.Server.dll [extra dotnet args...]
# Configure via env vars before invoking, e.g.:
#   ASPNETCORE_URLS=http://0.0.0.0:5099 HartsyInference__Backend=Cuda deploy/run-with-restart.sh ./HartsyInference.Server.dll

set -u

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/HartsyInference.Server.dll [extra dotnet args...]" >&2
  exit 1
fi

SERVER_DLL="$1"
shift
RESTART_DELAY_SECS="${HARTSY_RESTART_DELAY_SECS:-2}"
CRASH_WINDOW_SECS="${HARTSY_CRASH_WINDOW_SECS:-60}"
MAX_CRASHES_IN_WINDOW="${HARTSY_MAX_CRASHES_IN_WINDOW:-5}"

crash_times=()

while true; do
  start_ts=$(date +%s)
  echo "[run-with-restart] starting: dotnet '$SERVER_DLL' $*"
  dotnet "$SERVER_DLL" "$@"
  exit_code=$?
  end_ts=$(date +%s)

  if [[ $exit_code -eq 0 ]]; then
    echo "[run-with-restart] server exited cleanly (code 0) — not restarting."
    exit 0
  fi

  echo "[run-with-restart] server exited with code $exit_code after $((end_ts - start_ts))s — restarting in ${RESTART_DELAY_SECS}s."

  # Crash-loop breaker: mirrors systemd's StartLimitIntervalSec/StartLimitBurst so this script doesn't spin
  # forever on something that will never recover on its own (e.g. a bad model path in config).
  crash_times+=("$end_ts")
  cutoff=$((end_ts - CRASH_WINDOW_SECS))
  recent=()
  for t in "${crash_times[@]}"; do
    [[ $t -ge $cutoff ]] && recent+=("$t")
  done
  crash_times=("${recent[@]}")
  if [[ ${#crash_times[@]} -ge $MAX_CRASHES_IN_WINDOW ]]; then
    echo "[run-with-restart] ${#crash_times[@]} crashes within ${CRASH_WINDOW_SECS}s — giving up (likely a config/startup error, not a transient fault). Fix the underlying issue and restart manually." >&2
    exit 1
  fi

  sleep "$RESTART_DELAY_SECS"
done
