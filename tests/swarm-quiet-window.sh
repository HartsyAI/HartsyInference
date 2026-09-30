#!/usr/bin/env bash
# Swarm quiet-window gate — READ-ONLY against a running SwarmUI. Never stops, restarts or reconfigures
# SwarmUI or its backends; the only API calls are GetNewSession (creates a session record, nothing else),
# GetCurrentStatus, ListBackends and ListRecentLogMessages.
#
# Exits 0 only after --minutes consecutive minutes (default 10) in which EVERY poll (every --interval seconds,
# default 30) found the checks below clean. Any dirty poll restarts the window.
#
# Checked on every poll (all printed):
#   1. POST /API/GetCurrentStatus   -> live_gens, waiting_gens and loading_models must all be 0.
#   2. POST /API/ListBackends       -> ABORT (exit 2) if a backend whose title equals --forbid-backend-title
#                                      (default "HartsyInference GPU1 (3060)") reports enabled=true. That backend
#                                      is disabled today; enabled would mean Swarm can place gens on the 3060
#                                      underneath an in-process Phase 1 run.
#   3. nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory
#                                   -> every process on a target card (--gpu, matched by UUID or name substring)
#                                      must be SwarmUI itself: the swarmui.service MainPID or a process whose
#                                      name contains "SwarmUI". Anything else is FOREIGN and resets the window.
#   4. POST /API/ListRecentLogMessages from a cursor (skip with --no-log-check)
#                                   -> every Swarm log line since the previous poll must be benign (session
#                                      creation, the backend's idle memory housekeeping). Any other line — a T2I
#                                      "requested N image", an AudioLab ProcessTTS/ProcessSTT, an LLMAssistant
#                                      turn, a UI error — resets the window. A Swarm restart (new MainPID or the
#                                      log sequence going backwards) resets it too.
#
# WHY TWO KINDS OF CHECK: checks 1 and 3 are snapshots — a gen or a foreign process that starts and ends
# between two polls is invisible to them. The Swarm log is continuous, so check 4 sees every request in the
# window, including AudioLab and LLMAssistant requests served inside the SwarmUI process, which are neither
# T2I gens nor new processes. The compute-apps listing keyed by GPU UUID is what covers another agent's
# in-process CLI run, a `dotnet test` GPU suite or a stray `hartsy-bench` worker, which never reach Swarm at
# all (the "GPU turn-taking" and "GPU shared hard gate" directives) — this script gates on the CARD, not only on
# SwarmUI. Remaining blind spot: a non-Swarm GPU process that starts and exits between two polls.
#
# Usage:
#   tests/swarm-quiet-window.sh [--gpu <uuid|name-substring>]... [--minutes 10] [--interval 30]
#                               [--max-minutes N] [--state FILE] [--host H] [--port P] [--no-log-check]
#                               [--forbid-backend-title "HartsyInference GPU1 (3060)"]
#   --gpu           target card(s), repeatable; default = every card nvidia-smi lists.
#   --minutes 0     a spot check: exits 0 when the first poll is clean.
#   --max-minutes   give up (exit 3) after this much wall time in THIS invocation; default: wait forever.
#   --state FILE    persist the running window (and the log cursor) across invocations: a call within
#                   2*interval of the previous poll continues the same window instead of starting over, and its
#                   first poll checks the log lines emitted in between. Exists because some harnesses cap one
#                   foreground command below 10 minutes. A stale state file, or one from before a Swarm restart,
#                   is discarded.
#   --host/--port   default 192.168.10.188:7801 (or SWARM_HOST / SWARM_PORT).
#
# Exit codes: 0 quiet window achieved (state file removed)
#             2 aborted: forbidden backend enabled, Swarm API or nvidia-smi unusable after one retry, bad arguments
#             3 --max-minutes reached without a window (state kept when --state is set)
set -uo pipefail

HOST_DEFAULTED=1
[[ -n "${SWARM_HOST:-}" ]] && HOST_DEFAULTED=0
HOST="${SWARM_HOST:-192.168.10.188}"
PORT="${SWARM_PORT:-7801}"
WINDOW_MIN=10
INTERVAL_S=30
MAX_MIN=""
STATE=""
LOG_CHECK=1
FORBID_TITLE="HartsyInference GPU1 (3060)"
GPU_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) GPU_ARGS+=("$2"); shift 2 ;;
        --minutes) WINDOW_MIN="$2"; shift 2 ;;
        --interval) INTERVAL_S="$2"; shift 2 ;;
        --max-minutes) MAX_MIN="$2"; shift 2 ;;
        --state) STATE="$2"; shift 2 ;;
        --host) HOST="$2"; HOST_DEFAULTED=0; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --no-log-check) LOG_CHECK=0; shift ;;
        --forbid-backend-title) FORBID_TITLE="$2"; shift 2 ;;
        -h|--help) awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

BASE="http://${HOST}:${PORT}"
WINDOW_S=$(python3 -c "print(int(float('${WINDOW_MIN}') * 60))")
MAX_S=""
if [[ -n "${MAX_MIN}" ]]; then
    MAX_S=$(python3 -c "print(int(float('${MAX_MIN}') * 60))")
fi

log() { printf '%s %s\n' "$(date -u +%H:%M:%SZ)" "$*"; }
api() { curl -s --max-time 20 -X POST "${BASE}/API/$1" -H 'Content-Type: application/json' -d "$2"; }

new_session() {
    SESSION=$(api GetNewSession '{}' | python3 -c 'import sys, json; print(json.load(sys.stdin)["session_id"])' 2>/dev/null)
    [[ -n "${SESSION}" ]]
}

# swarm_call ENDPOINT [EXTRA_JSON_MEMBERS] — POSTs {"session_id": ..., EXTRA}. An unusable reply (transport
# failure, expired session, error object, non-JSON) gets one retry on a fresh session after 5 s.
swarm_call() {
    local ep="$1" extra="${2:-}" out attempt
    for attempt in 1 2; do
        out=$(api "${ep}" "{\"session_id\":\"${SESSION}\"${extra:+,${extra}}}")
        if printf '%s' "${out}" | python3 -c '
import sys, json
d = json.load(sys.stdin)
sys.exit(0 if isinstance(d, dict) and "error" not in d and "error_id" not in d else 1)' 2>/dev/null; then
            printf '%s' "${out}"
            return 0
        fi
        if [[ ${attempt} -eq 1 ]]; then
            sleep 5
            new_session || true
        fi
    done
    printf '%s' "${out}"
    return 1
}

smi_apps() {
    local out attempt
    for attempt in 1 2; do
        if out=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader 2>/dev/null); then
            printf '%s' "${out}"
            return 0
        fi
        sleep 3
    done
    return 1
}

swarm_pid() {
    local pid
    pid=$(systemctl --user show -p MainPID --value swarmui.service 2>/dev/null || echo 0)
    echo "${pid:-0}"
}

# The log cursor is the last sequence id per type, kept positional ("12,34,…") so the state file holds digits only.
LOG_TYPES='["Debug","Info","Init","Warning","Error"]'
LOG_PY='
import json, re, sys
types = ["Debug", "Info", "Init", "Warning", "Error"]
benign = [re.compile(p) for p in (r"^Creating new session .", r"^\[CudaBackend\] FreeAllDeviceMemory",
                                  r"^\[HostMemory\] free-memory sweep")]
mode = sys.argv[1]
old = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 and sys.argv[2] else [0] * len(types)
if mode == "request":
    print(json.dumps(dict(zip(types, old))) if any(old) else "{}")
    sys.exit(0)
try:
    d = json.load(sys.stdin)
    data = d.get("data") or {}
except Exception as e:
    print("ERR", e)
    sys.exit(0)
msgs = sorted((m.get("sequence_id", 0), t, m.get("time", ""), m.get("message", "")) for t, ms in data.items() for m in ms)
new = list(old)
for s, t, _, _ in msgs:
    if t in types:
        new[types.index(t)] = max(new[types.index(t)], s)
print("CURSOR " + ",".join(str(x) for x in new))
if mode == "init":
    sys.exit(0)
last = d.get("last_sequence_id")
if isinstance(last, int) and max(old) > 0 and last < max(old):
    print("RESTART")
for s, t, tm, msg in msgs:
    kind = "benign" if any(p.search(msg) for p in benign) else "FOREIGN"
    print((kind + " " + str(s) + " " + t + " " + tm + " " + msg[:180]).replace("\n", " "))
'

# log_fetch MODE CURSOR — MODE init prints only "CURSOR …"; MODE diff also prints one line per new message.
log_fetch() {
    local mode="$1" cursor="$2" seqs out
    seqs=$(python3 -c "${LOG_PY}" request "$( [[ "${mode}" == init ]] && echo "" || echo "${cursor}" )")
    out=$(swarm_call ListRecentLogMessages "\"types\":${LOG_TYPES},\"last_sequence_ids\":${seqs}") || return 1
    printf '%s' "${out}" | python3 -c "${LOG_PY}" "${mode}" "${cursor}"
}

# ---- target cards ----------------------------------------------------------------------------------------
GPU_TABLE=$(nvidia-smi --query-gpu=uuid,name,index --format=csv,noheader 2>/dev/null)
if [[ $? -ne 0 || -z "${GPU_TABLE}" ]]; then
    log "ABORT: nvidia-smi --query-gpu failed"
    exit 2
fi
TARGET_UUIDS=()
if [[ ${#GPU_ARGS[@]} -eq 0 ]]; then
    while IFS= read -r line; do
        TARGET_UUIDS+=("${line%%,*}")
    done <<<"${GPU_TABLE}"
else
    for want in "${GPU_ARGS[@]}"; do
        match=$(printf '%s\n' "${GPU_TABLE}" | python3 -c '
import sys
want = sys.argv[1].strip().lower()
for line in sys.stdin:
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 2:
        continue
    uuid, name = parts[0], parts[1]
    if want == uuid.lower() or want in name.lower():
        print(uuid)' "${want}")
        if [[ -z "${match}" ]]; then
            log "ABORT: no GPU matches '${want}'. nvidia-smi lists:"
            printf '  %s\n' "${GPU_TABLE}"
            exit 2
        fi
        while IFS= read -r u; do
            TARGET_UUIDS+=("${u}")
        done <<<"${match}"
    done
fi
TARGET_DESC=$(printf '%s\n' "${GPU_TABLE}" | python3 -c '
import sys
targets = set(sys.argv[1].split())
for line in sys.stdin:
    parts = [p.strip() for p in line.split(",")]
    if parts and parts[0] in targets:
        print(f"    {parts[0]}  {parts[1]}  (nvidia-smi index {parts[2]})")' "${TARGET_UUIDS[*]}")

# SwarmUI's own pid: the user unit's MainPID; a name match is the fallback for a Swarm started by hand.
SWARM_PID=$(swarm_pid)

if ! new_session; then
    log "ABORT: GetNewSession failed against ${BASE}"
    exit 2
fi

log "quiet-window gate: host=${BASE}$( [[ ${HOST_DEFAULTED} -eq 1 ]] && echo " (default; --host or SWARM_HOST to change)" ) session=${SESSION:0:8}... window=${WINDOW_MIN}min interval=${INTERVAL_S}s max=${MAX_MIN:-unbounded}min"
log "  forbidden backend title: '${FORBID_TITLE}' (must stay disabled)"
log "  swarmui.service MainPID: ${SWARM_PID} (allowed on the target cards, plus any process named *SwarmUI*)"
log "  Swarm request log check: $( [[ ${LOG_CHECK} -eq 1 ]] && echo "on (any non-benign line since the previous poll resets the window)" || echo "OFF (--no-log-check)" )"
log "  target card(s):"
printf '%s\n' "${TARGET_DESC}"

# ---- resume state ----------------------------------------------------------------------------------------
NOW=$(date +%s)
START=${NOW}
# --max-minutes caps THIS invocation; START (restored from --state) only reports the total wait.
INVOCATION_START=${NOW}
QUIET_SINCE=""
POLLS=0
LOG_CURSOR=""
if [[ -n "${STATE}" && -f "${STATE}" ]]; then
    STATE_QUIET_SINCE="" STATE_LAST_POLL="" STATE_START="" STATE_POLLS="" STATE_LOG_CURSOR="" STATE_SWARM_PID=""
    # Parsed, not sourced: only known keys, digits and commas only.
    while IFS='=' read -r key value; do
        case "${key}" in
            STATE_QUIET_SINCE|STATE_LAST_POLL|STATE_START|STATE_POLLS|STATE_LOG_CURSOR|STATE_SWARM_PID)
                [[ "${value}" =~ ^[0-9,]*$ ]] && printf -v "${key}" '%s' "${value}" ;;
        esac
    done < "${STATE}"
    if [[ -n "${STATE_LAST_POLL}" ]] && (( NOW - STATE_LAST_POLL <= 2 * INTERVAL_S )) && [[ "${STATE_SWARM_PID}" == "${SWARM_PID}" ]]; then
        QUIET_SINCE="${STATE_QUIET_SINCE}"
        START="${STATE_START:-${NOW}}"
        POLLS="${STATE_POLLS:-0}"
        LOG_CURSOR="${STATE_LOG_CURSOR}"
        log "  resumed state from ${STATE}: quiet_since=${QUIET_SINCE:-none} polls=${POLLS} (last poll $(( NOW - STATE_LAST_POLL ))s ago)"
    else
        log "  stale state in ${STATE} (last poll too old, or Swarm restarted since) — starting a fresh window"
        rm -f "${STATE}"
    fi
fi
if [[ ${LOG_CHECK} -eq 1 && -z "${LOG_CURSOR}" ]]; then
    INIT=$(log_fetch init "") || { log "ABORT: ListRecentLogMessages unusable (needs the ViewLogs permission; --no-log-check to skip)"; exit 2; }
    LOG_CURSOR=$(printf '%s\n' "${INIT}" | sed -n 's/^CURSOR //p' | head -1)
    if [[ -z "${LOG_CURSOR}" ]]; then
        log "ABORT: ListRecentLogMessages returned no cursor: ${INIT:0:200}"
        exit 2
    fi
fi

save_state() {
    if [[ -n "${STATE}" ]]; then
        printf 'STATE_QUIET_SINCE=%s\nSTATE_LAST_POLL=%s\nSTATE_START=%s\nSTATE_POLLS=%s\nSTATE_LOG_CURSOR=%s\nSTATE_SWARM_PID=%s\n' \
            "${QUIET_SINCE}" "$(date +%s)" "${START}" "${POLLS}" "${LOG_CURSOR}" "${SWARM_PID}" > "${STATE}"
    fi
}

# ---- poll loop -------------------------------------------------------------------------------------------
while true; do
    NOW=$(date +%s)
    POLLS=$((POLLS + 1))
    REASONS=()

    CURRENT_PID=$(swarm_pid)
    if [[ "${CURRENT_PID}" != "${SWARM_PID}" ]]; then
        REASONS+=("Swarm restarted (MainPID ${SWARM_PID} -> ${CURRENT_PID})")
        SWARM_PID="${CURRENT_PID}"
        new_session || true
        if [[ ${LOG_CHECK} -eq 1 ]]; then
            LOG_CURSOR=$(log_fetch init "" | sed -n 's/^CURSOR //p' | head -1)
        fi
    fi

    STATUS_JSON=$(swarm_call GetCurrentStatus)
    STATUS_LINE=$(printf '%s' "${STATUS_JSON}" | python3 -c '
import sys, json
try:
    s = json.load(sys.stdin)["status"]
    print(s.get("live_gens", -1), s.get("waiting_gens", -1), s.get("loading_models", -1), s.get("waiting_backends", -1))
except Exception as e:
    print("ERR", e)' 2>/dev/null)
    if [[ -z "${STATUS_LINE}" || "${STATUS_LINE}" == ERR* ]]; then
        log "ABORT: GetCurrentStatus unusable after a retry: ${STATUS_JSON:0:200}"
        save_state
        exit 2
    fi
    read -r LIVE WAITING LOADING WBACK <<<"${STATUS_LINE}"
    if [[ "${LIVE}" != "0" || "${WAITING}" != "0" || "${LOADING}" != "0" ]]; then
        REASONS+=("Swarm queue live=${LIVE} waiting=${WAITING} loading=${LOADING}")
    fi

    BACKENDS_JSON=$(swarm_call ListBackends)
    FORBID_STATE=$(printf '%s' "${BACKENDS_JSON}" | python3 -c '
import sys, json
title = sys.argv[1]
try:
    d = json.load(sys.stdin)
except Exception as e:
    print("ERR", e); sys.exit(0)
hits = [b for b in d.values() if isinstance(b, dict) and (b.get("title") or "") == title]
if not hits:
    print("absent")
elif any(b.get("enabled") is True for b in hits):
    print("ENABLED " + ",".join(str(b.get("status")) for b in hits))
else:
    print("disabled " + ",".join(str(b.get("status")) for b in hits))' "${FORBID_TITLE}")
    if [[ "${FORBID_STATE}" == ERR* ]]; then
        log "ABORT: ListBackends unusable after a retry: ${BACKENDS_JSON:0:200}"
        save_state
        exit 2
    fi
    if [[ "${FORBID_STATE}" == ENABLED* ]]; then
        log "ABORT: backend '${FORBID_TITLE}' is ENABLED (${FORBID_STATE#ENABLED }) — Swarm could place gens on the 3060."
        save_state
        exit 2
    fi

    if ! APPS=$(smi_apps); then
        log "ABORT: nvidia-smi --query-compute-apps failed twice"
        save_state
        exit 2
    fi
    FOREIGN=$(printf '%s\n' "${APPS}" | python3 -c '
import sys
targets = set(sys.argv[1].split())
swarm_pid = sys.argv[2]
for line in sys.stdin:
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 3 or not parts[0]:
        continue
    uuid, pid, name = parts[0], parts[1], parts[2]
    if uuid not in targets:
        continue
    if pid == swarm_pid or "swarmui" in name.lower():
        continue
    mem = parts[3] if len(parts) > 3 else ""
    print(f"{uuid} pid={pid} {name} {mem}")' "${TARGET_UUIDS[*]}" "${SWARM_PID}")
    if [[ -n "${FOREIGN}" ]]; then
        REASONS+=("foreign process on a target card")
    fi

    LOG_NEW=0
    LOG_FOREIGN=""
    if [[ ${LOG_CHECK} -eq 1 ]]; then
        if ! DIFF=$(log_fetch diff "${LOG_CURSOR}"); then
            log "ABORT: ListRecentLogMessages unusable after a retry (--no-log-check to skip)"
            save_state
            exit 2
        fi
        NEXT=$(printf '%s\n' "${DIFF}" | sed -n 's/^CURSOR //p' | head -1)
        [[ -n "${NEXT}" ]] && LOG_CURSOR="${NEXT}"
        LOG_NEW=$(printf '%s\n' "${DIFF}" | grep -c '^\(benign\|FOREIGN\) ')
        LOG_FOREIGN=$(printf '%s\n' "${DIFF}" | grep '^FOREIGN \|^RESTART')
        if [[ -n "${LOG_FOREIGN}" ]]; then
            REASONS+=("Swarm log shows activity")
        fi
    fi

    if [[ ${#REASONS[@]} -eq 0 ]]; then
        if [[ -z "${QUIET_SINCE}" ]]; then
            QUIET_SINCE=${NOW}
        fi
        HELD=$((NOW - QUIET_SINCE))
    else
        QUIET_SINCE=""
        HELD=0
    fi

    log "poll ${POLLS}: live_gens=${LIVE} waiting_gens=${WAITING} loading_models=${LOADING} waiting_backends=${WBACK} | 3060 backend: ${FORBID_STATE} | foreign on targets: ${FOREIGN:-none} | log: $( [[ ${LOG_CHECK} -eq 1 ]] && echo "${LOG_NEW} new line(s)" || echo off ) | quiet held ${HELD}s / ${WINDOW_S}s"
    if [[ ${#REASONS[@]} -gt 0 ]]; then
        printf '    reset: %s\n' "${REASONS[@]}"
    fi
    if [[ -n "${FOREIGN}" ]]; then
        printf '    foreign: %s\n' "${FOREIGN}"
    fi
    if [[ -n "${LOG_FOREIGN}" ]]; then
        printf '%s\n' "${LOG_FOREIGN}" | head -8 | sed 's/^/    swarm log: /'
    fi
    save_state

    if [[ ${#REASONS[@]} -eq 0 && ${HELD} -ge ${WINDOW_S} ]]; then
        log "QUIET WINDOW OK: ${HELD}s (>= ${WINDOW_S}s): Swarm queue empty, no request in its log$( [[ ${LOG_CHECK} -eq 1 ]] || echo " (log check OFF)" ), no foreign process on:"
        printf '%s\n' "${TARGET_DESC}"
        log "  checked ${POLLS} poll(s) since $(date -u -d "@${START}" +%H:%M:%SZ); backend '${FORBID_TITLE}' stayed ${FORBID_STATE%% *}."
        if [[ -n "${STATE}" ]]; then
            rm -f "${STATE}"
        fi
        exit 0
    fi

    if [[ -n "${MAX_S}" ]] && (( NOW - INVOCATION_START >= MAX_S )); then
        log "NO WINDOW: this invocation waited $(( NOW - INVOCATION_START ))s (cap ${MAX_S}s), $(( NOW - START ))s in total; quiet held ${HELD}s of ${WINDOW_S}s.${STATE:+ State kept in ${STATE}; re-run to continue the window.}"
        exit 3
    fi

    sleep "${INTERVAL_S}"
done
