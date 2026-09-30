#!/usr/bin/env bash
# Swarm quiet-window gate — READ-ONLY against a running SwarmUI. Never stops, restarts or reconfigures
# SwarmUI or its backends; the only API calls are GetNewSession (creates a session record, nothing else),
# GetCurrentStatus and ListBackends.
#
# Exits 0 only after `live_gens==0 && waiting_gens==0 && loading_models==0` has held on EVERY poll for
# --minutes consecutive minutes (default 10) AND no foreign process was seen on any target card for that
# whole window. Polls every --interval seconds (default 30).
#
# Checked on every poll (all three printed):
#   1. POST /API/GetCurrentStatus   -> status.{live_gens,waiting_gens,loading_models,waiting_backends}
#   2. POST /API/ListBackends       -> ABORT (exit 2) if a backend whose title equals --forbid-backend-title
#                                      (default "HartsyInference GPU1 (3060)") reports enabled=true. That backend
#                                      is disabled today; enabled would mean Swarm can place gens on the 3060
#                                      underneath an in-process Phase 1 run.
#   3. nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory
#                                   -> every process on a target card (--gpu, matched by UUID or name substring)
#                                      must be SwarmUI itself: the swarmui.service MainPID or a process whose
#                                      name contains "SwarmUI". Anything else is FOREIGN and resets the window.
#
# WHY THE UUID CHECK EXISTS: GetCurrentStatus only knows SwarmUI's own generation queue. Another agent's
# in-process CLI run, a `dotnet test` GPU suite or a stray `hartsy-bench` worker never appears there. The
# compute-apps listing keyed by GPU UUID is what covers those (the "GPU turn-taking" and "GPU shared hard gate"
# directives) — this script gates on the CARD, not only on SwarmUI. Known blind spot: an AudioLab or
# LLMAssistant request served INSIDE the SwarmUI process is neither a T2I gen nor a new process, so neither
# check sees it; the bench scripts watch used_memory deltas on the SwarmUI rows for that.
#
# Usage:
#   tests/swarm-quiet-window.sh [--gpu <uuid|name-substring>]... [--minutes 10] [--interval 30]
#                               [--max-minutes N] [--state FILE] [--host H] [--port P]
#                               [--forbid-backend-title "HartsyInference GPU1 (3060)"]
#   --gpu           target card(s), repeatable; default = every card nvidia-smi lists.
#   --max-minutes   give up (exit 3) after this much wall time without a window; default: wait forever.
#   --state FILE    persist the running window across invocations: a call within 2*interval of the previous
#                   poll continues the same window instead of starting over. Exists because some harnesses cap
#                   one foreground command below 10 minutes; the window stays continuous (the gap between
#                   invocations is bounded by one poll interval, and a stale state file is discarded).
#
# Exit codes: 0 quiet window achieved (state file removed)
#             2 aborted: forbidden backend enabled, API or nvidia-smi failure, bad arguments
#             3 --max-minutes reached without a window (state kept when --state is set)
set -uo pipefail

HOST="${SWARM_HOST:-192.168.10.188}"
PORT="${SWARM_PORT:-7801}"
WINDOW_MIN=10
INTERVAL_S=30
MAX_MIN=""
STATE=""
FORBID_TITLE="HartsyInference GPU1 (3060)"
GPU_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) GPU_ARGS+=("$2"); shift 2 ;;
        --minutes) WINDOW_MIN="$2"; shift 2 ;;
        --interval) INTERVAL_S="$2"; shift 2 ;;
        --max-minutes) MAX_MIN="$2"; shift 2 ;;
        --state) STATE="$2"; shift 2 ;;
        --host) HOST="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --forbid-backend-title) FORBID_TITLE="$2"; shift 2 ;;
        -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
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
SWARM_PID=$(systemctl --user show -p MainPID --value swarmui.service 2>/dev/null || echo 0)
SWARM_PID="${SWARM_PID:-0}"

SESSION=$(api GetNewSession '{}' | python3 -c 'import sys, json; print(json.load(sys.stdin)["session_id"])' 2>/dev/null)
if [[ -z "${SESSION}" ]]; then
    log "ABORT: GetNewSession failed against ${BASE}"
    exit 2
fi

log "quiet-window gate: host=${BASE} session=${SESSION:0:8}... window=${WINDOW_MIN}min interval=${INTERVAL_S}s max=${MAX_MIN:-unbounded}min"
log "  forbidden backend title: '${FORBID_TITLE}' (must stay disabled)"
log "  swarmui.service MainPID: ${SWARM_PID} (allowed on the target cards, plus any process named *SwarmUI*)"
log "  target card(s):"
printf '%s\n' "${TARGET_DESC}"

# ---- resume state ----------------------------------------------------------------------------------------
NOW=$(date +%s)
START=${NOW}
QUIET_SINCE=""
POLLS=0
if [[ -n "${STATE}" && -f "${STATE}" ]]; then
    # shellcheck disable=SC1090
    source "${STATE}"
    if [[ -n "${STATE_LAST_POLL:-}" ]] && (( NOW - STATE_LAST_POLL <= 2 * INTERVAL_S )); then
        QUIET_SINCE="${STATE_QUIET_SINCE:-}"
        START="${STATE_START:-${NOW}}"
        POLLS="${STATE_POLLS:-0}"
        log "  resumed state from ${STATE}: quiet_since=${QUIET_SINCE:-none} polls=${POLLS} (last poll $(( NOW - STATE_LAST_POLL ))s ago)"
    else
        log "  stale state in ${STATE} (last poll too old) — starting a fresh window"
        rm -f "${STATE}"
    fi
fi

save_state() {
    if [[ -n "${STATE}" ]]; then
        printf 'STATE_QUIET_SINCE=%s\nSTATE_LAST_POLL=%s\nSTATE_START=%s\nSTATE_POLLS=%s\n' \
            "${QUIET_SINCE}" "$(date +%s)" "${START}" "${POLLS}" > "${STATE}"
    fi
}

# ---- poll loop -------------------------------------------------------------------------------------------
while true; do
    NOW=$(date +%s)
    POLLS=$((POLLS + 1))

    STATUS_JSON=$(api GetCurrentStatus "{\"session_id\":\"${SESSION}\"}")
    STATUS_LINE=$(printf '%s' "${STATUS_JSON}" | python3 -c '
import sys, json
try:
    s = json.load(sys.stdin)["status"]
    print(s.get("live_gens", -1), s.get("waiting_gens", -1), s.get("loading_models", -1), s.get("waiting_backends", -1))
except Exception as e:
    print("ERR", e)' 2>/dev/null)
    if [[ -z "${STATUS_LINE}" || "${STATUS_LINE}" == ERR* ]]; then
        log "ABORT: GetCurrentStatus unusable: ${STATUS_JSON:0:200}"
        save_state
        exit 2
    fi
    read -r LIVE WAITING LOADING WBACK <<<"${STATUS_LINE}"

    BACKENDS_JSON=$(api ListBackends "{\"session_id\":\"${SESSION}\"}")
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
        log "ABORT: ListBackends unusable: ${BACKENDS_JSON:0:200}"
        save_state
        exit 2
    fi
    if [[ "${FORBID_STATE}" == ENABLED* ]]; then
        log "ABORT: backend '${FORBID_TITLE}' is ENABLED (${FORBID_STATE#ENABLED }) — Swarm could place gens on the 3060."
        save_state
        exit 2
    fi

    APPS=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader 2>/dev/null)
    if [[ $? -ne 0 ]]; then
        log "ABORT: nvidia-smi --query-compute-apps failed"
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

    QUIET=1
    if [[ "${LIVE}" != "0" || "${WAITING}" != "0" || "${LOADING}" != "0" ]]; then
        QUIET=0
    fi
    if [[ -n "${FOREIGN}" ]]; then
        QUIET=0
    fi

    if [[ ${QUIET} -eq 1 ]]; then
        if [[ -z "${QUIET_SINCE}" ]]; then
            QUIET_SINCE=${NOW}
        fi
        HELD=$((NOW - QUIET_SINCE))
    else
        QUIET_SINCE=""
        HELD=0
    fi

    log "poll ${POLLS}: live_gens=${LIVE} waiting_gens=${WAITING} loading_models=${LOADING} waiting_backends=${WBACK} | 3060 backend: ${FORBID_STATE} | foreign on targets: ${FOREIGN:-none} | quiet held ${HELD}s / ${WINDOW_S}s"
    if [[ -n "${FOREIGN}" ]]; then
        printf '    foreign: %s\n' "${FOREIGN}"
    fi
    save_state

    if [[ ${QUIET} -eq 1 && ${HELD} -ge ${WINDOW_S} ]]; then
        log "QUIET WINDOW OK: ${HELD}s (>= ${WINDOW_S}s) with live_gens/waiting_gens/loading_models all 0 and no foreign process on:"
        printf '%s\n' "${TARGET_DESC}"
        log "  checked ${POLLS} poll(s) since $(date -u -d "@${START}" +%H:%M:%SZ); backend '${FORBID_TITLE}' stayed ${FORBID_STATE%% *}."
        if [[ -n "${STATE}" ]]; then
            rm -f "${STATE}"
        fi
        exit 0
    fi

    if [[ -n "${MAX_S}" ]] && (( NOW - START >= MAX_S )); then
        log "NO WINDOW: waited $(( NOW - START ))s (cap ${MAX_S}s); quiet held ${HELD}s of ${WINDOW_S}s.${STATE:+ State kept in ${STATE}; re-run to continue the window.}"
        exit 3
    fi

    sleep "${INTERVAL_S}"
done
