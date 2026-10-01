#!/usr/bin/env bash
# Swarm quiet-window gate — READ-ONLY against a running SwarmUI. Never stops, restarts or reconfigures
# SwarmUI or its backends; the only API calls are GetNewSession (creates a session record, nothing else),
# GetGlobalStatus and ListBackends, plus `journalctl` and `nvidia-smi` reads.
#
# Exits 0 only after --minutes consecutive minutes (default 10) in which EVERY poll (every --interval seconds,
# default 30) found the checks below clean. Any dirty poll restarts the window.
#
# Checked on every poll (all printed):
#   1. POST /API/GetGlobalStatus    -> live_gens, waiting_gens and loading_models summed over EVERY session must
#                                      be 0. (GetCurrentStatus is per-session: it only counts the calling
#                                      session's own gens, so it reads 0 while the user generates.)
#   2. POST /API/ListBackends       -> ABORT (exit 2) if a backend whose title equals --forbid-backend-title
#                                      (default "HartsyInference GPU1 (3060)") reports enabled=true. That backend
#                                      is disabled today; enabled would mean Swarm can place gens on the 3060
#                                      underneath an in-process run.
#   3. nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory
#                                   -> every process on a target card (--gpu, matched by UUID or name substring)
#                                      must be SwarmUI itself (the swarmui.service MainPID or a process named
#                                      *SwarmUI*); anything else is FOREIGN. SwarmUI appearing on, leaving, or
#                                      moving more than 256 MiB on a target card between polls also resets the
#                                      window (a model load or gen inside Swarm — LLMAssistant and AudioLab
#                                      included).
#   4. journalctl --user -t <--journal-id, default swarmui-run.sh>, from a cursor (skip with --no-journal-check)
#                                   -> no Swarm request/activity line since the previous poll: "requested"
#                                      (T2I "User X requested N image(s)", AudioLab ProcessTTS), "Generated an
#                                      image", "Backend request #N", "[AudioLab] Process…", sampler "step N/M".
#                                      A Swarm restart (new MainPID) resets the window too.
#
# WHY SEVERAL CHECKS: 1 and 3 are snapshots — a gen or a foreign process that starts and ends between two polls
# is invisible to them. The journal is continuous, so check 4 sees every T2I request and every AudioLab TTS
# request in the window. The compute-apps listing keyed by GPU UUID is what covers another agent's in-process
# CLI run or `dotnet test` GPU suite, which never reaches Swarm (the "GPU turn-taking" and "GPU shared hard
# gate" directives). Remaining blind spots: AudioLab STT and LLMAssistant chat turns write no journal line and
# show up only as SwarmUI memory moving on a card (check 3); a non-Swarm GPU process that starts and exits
# between two polls.
#
# AFTER A TIMED ARM run `--verify-since <UTC start of the arm>`: it scans the journal from that instant for
# request lines and Swarm restarts and reads the global queue once; exit 0 = nothing landed, exit 4 = redo the
# arm. It cannot see past GPU processes — sample nvidia-smi during the arm for that.
#
# Usage:
#   tests/swarm-quiet-window.sh [--gpu <uuid|name-substring>]... [--minutes 10] [--interval 30]
#                               [--max-minutes N] [--state FILE] [--host H] [--port P]
#                               [--journal-id swarmui-run.sh] [--no-journal-check]
#                               [--forbid-backend-title "HartsyInference GPU1 (3060)"]
#   tests/swarm-quiet-window.sh --verify-since "2026-10-01T01:17:00Z"   (or "@<epoch>")
#   --gpu           target card(s), repeatable; default = every card nvidia-smi lists.
#   --minutes 0     a spot check: exits 0 when the first poll is clean.
#   --max-minutes   give up (exit 3) after this much wall time in THIS invocation; default: wait forever.
#   --state FILE    persist the running window (and the journal cursor) across invocations: a call within
#                   2*interval of the previous poll continues the same window instead of starting over, and its
#                   first poll checks the journal lines written in between. Exists because some harnesses cap one
#                   foreground command below 10 minutes. A stale state file, or one from before a Swarm restart,
#                   is discarded.
#   --host/--port   default 192.168.10.188:7801 (or SWARM_HOST / SWARM_PORT); --journal-id or SWARM_JOURNAL_ID.
#
# Exit codes: 0 quiet window achieved (state file removed) / --verify-since found nothing
#             2 aborted: forbidden backend enabled, Swarm API, journal or nvidia-smi unusable after one retry,
#               bad arguments
#             3 --max-minutes reached without a window (state kept when --state is set)
#             4 --verify-since found a request, a Swarm restart or a busy queue: redo the arm
set -uo pipefail

HOST_DEFAULTED=1
[[ -n "${SWARM_HOST:-}" ]] && HOST_DEFAULTED=0
HOST="${SWARM_HOST:-192.168.10.188}"
PORT="${SWARM_PORT:-7801}"
JOURNAL_ID="${SWARM_JOURNAL_ID:-swarmui-run.sh}"
JOURNAL_CHECK=1
VERIFY_SINCE=""
WINDOW_MIN=10
INTERVAL_S=30
MAX_MIN=""
STATE=""
FORBID_TITLE="HartsyInference GPU1 (3060)"
GPU_ARGS=()
SWARM_MEM_MOVE_MIB=256
REQUEST_RE='requested|Generated an image|Backend request #[0-9]+|\[AudioLab\] Process|Process(STT|TTS|Audio)|step [0-9]+/[0-9]+'

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu) GPU_ARGS+=("$2"); shift 2 ;;
        --minutes) WINDOW_MIN="$2"; shift 2 ;;
        --interval) INTERVAL_S="$2"; shift 2 ;;
        --max-minutes) MAX_MIN="$2"; shift 2 ;;
        --state) STATE="$2"; shift 2 ;;
        --host) HOST="$2"; HOST_DEFAULTED=0; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --journal-id) JOURNAL_ID="$2"; shift 2 ;;
        --no-journal-check|--no-log-check) JOURNAL_CHECK=0; shift ;;
        --verify-since) VERIFY_SINCE="$2"; shift 2 ;;
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

# global_status — "live waiting loading waiting_backends" summed over every session, or "ERR …".
global_status() {
    local out
    out=$(swarm_call GetGlobalStatus)
    printf '%s' "${out}" | python3 -c '
import sys, json
try:
    s = json.load(sys.stdin)["status"]
    print(s.get("live_gens", -1), s.get("waiting_gens", -1), s.get("loading_models", -1), s.get("waiting_backends", -1))
except Exception as e:
    print("ERR", e)' 2>/dev/null
}

smi_apps() {
    local out attempt
    for attempt in 1 2; do
        if out=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null); then
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

journal_newest_cursor() {
    journalctl --user -t "${JOURNAL_ID}" -n 1 --show-cursor --no-pager -o cat 2>/dev/null | sed -n 's/^-- cursor: //p' | tail -1
}

if ! new_session; then
    log "ABORT: GetNewSession failed against ${BASE}"
    exit 2
fi

# ---- --verify-since: after a timed arm, did anything land in Swarm since it started? ---------------------------
if [[ -n "${VERIFY_SINCE}" ]]; then
    if ! SINCE_EPOCH=$(date -u -d "${VERIFY_SINCE}" +%s 2>/dev/null); then
        log "ABORT: cannot parse --verify-since '${VERIFY_SINCE}' (use e.g. 2026-10-01T01:17:00Z or @<epoch>)"
        exit 2
    fi
    if [[ -z "$(journal_newest_cursor)" ]]; then
        log "ABORT: the journal has no entries for identifier '${JOURNAL_ID}' (--journal-id to change)"
        exit 2
    fi
    JLINES=$(journalctl --user -t "${JOURNAL_ID}" --since "@${SINCE_EPOCH}" --no-pager -o short-iso 2>/dev/null)
    REQS=$(printf '%s\n' "${JLINES}" | grep -E "${REQUEST_RE}")
    NREQ=$( [[ -n "${REQS}" ]] && printf '%s\n' "${REQS}" | wc -l || echo 0 )
    PIDS=$(printf '%s\n' "${JLINES}" | grep -oE "${JOURNAL_ID//./\\.}\[[0-9]+\]" | sort -u | tr '\n' ' ')
    NPIDS=$(printf '%s' "${PIDS}" | wc -w)
    QUEUE=$(global_status)
    read -r LIVE WAITING LOADING WBACK <<<"${QUEUE}"
    log "VERIFY since $(date -u -d "@${SINCE_EPOCH}" +%Y-%m-%dT%H:%M:%SZ) ($(( ($(date +%s) - SINCE_EPOCH) / 60 )) min): ${NREQ} Swarm request/activity line(s); launcher pid(s) in window: ${PIDS:-none} | queue now: live=${LIVE} waiting=${WAITING} loading=${LOADING}"
    if [[ -n "${REQS}" ]]; then
        ASKED=$(printf '%s\n' "${REQS}" | grep -i 'requested')
        NASKED=$( [[ -n "${ASKED}" ]] && printf '%s\n' "${ASKED}" | wc -l || echo 0 )
        if [[ -n "${ASKED}" ]]; then
            printf '%s\n' "${ASKED}" | head -20 | cut -c1-220 | sed 's/^/    requested: /'
        fi
        log "  ${NASKED} 'requested' line(s) (first 20 shown) + $(( NREQ - NASKED )) other activity line(s) (Backend request / Generated / steps)"
    fi
    DIRTY=0
    [[ ${NREQ} -gt 0 ]] && DIRTY=1
    [[ ${NPIDS} -gt 1 ]] && { log "  Swarm restarted inside the window"; DIRTY=1; }
    [[ "${QUEUE}" == ERR* ]] && { log "ABORT: GetGlobalStatus unusable: ${QUEUE}"; exit 2; }
    [[ "${LIVE}" != "0" || "${WAITING}" != "0" || "${LOADING}" != "0" ]] && { log "  Swarm queue is busy right now"; DIRTY=1; }
    if [[ ${DIRTY} -eq 1 ]]; then
        log "VERIFY FAILED: something landed in Swarm during the arm — redo it (GPU processes are not covered retroactively)"
        exit 4
    fi
    log "VERIFY OK: no Swarm request, restart or queued gen since then (GPU processes are not covered retroactively)"
    exit 0
fi

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

log "quiet-window gate: host=${BASE}$( [[ ${HOST_DEFAULTED} -eq 1 ]] && echo " (default; --host or SWARM_HOST to change)" ) session=${SESSION:0:8}... window=${WINDOW_MIN}min interval=${INTERVAL_S}s max=${MAX_MIN:-unbounded}min"
log "  queue counters: GetGlobalStatus (every session); forbidden backend title: '${FORBID_TITLE}' (must stay disabled)"
log "  swarmui.service MainPID: ${SWARM_PID} (allowed on the target cards, plus any process named *SwarmUI*; its memory moving > ${SWARM_MEM_MOVE_MIB} MiB counts as activity)"
log "  journal request check: $( [[ ${JOURNAL_CHECK} -eq 1 ]] && echo "on (journalctl --user -t ${JOURNAL_ID})" || echo "OFF (--no-journal-check)" )"
log "  target card(s):"
printf '%s\n' "${TARGET_DESC}"

# ---- resume state ----------------------------------------------------------------------------------------
NOW=$(date +%s)
START=${NOW}
# --max-minutes caps THIS invocation; START (restored from --state) only reports the total wait.
INVOCATION_START=${NOW}
QUIET_SINCE=""
POLLS=0
JCURSOR=""
if [[ -n "${STATE}" && -f "${STATE}" ]]; then
    STATE_QUIET_SINCE="" STATE_LAST_POLL="" STATE_START="" STATE_POLLS="" STATE_JOURNAL_CURSOR="" STATE_SWARM_PID=""
    # Parsed, not sourced: known keys only; digits for numbers, the journald cursor alphabet for the cursor.
    while IFS='=' read -r key value; do
        case "${key}" in
            STATE_QUIET_SINCE|STATE_LAST_POLL|STATE_START|STATE_POLLS|STATE_SWARM_PID)
                [[ "${value}" =~ ^[0-9]*$ ]] && printf -v "${key}" '%s' "${value}" ;;
            STATE_JOURNAL_CURSOR)
                [[ "${value}" =~ ^[A-Za-z0-9=\;]*$ ]] && printf -v "${key}" '%s' "${value}" ;;
        esac
    done < "${STATE}"
    if [[ -n "${STATE_LAST_POLL}" ]] && (( NOW - STATE_LAST_POLL <= 2 * INTERVAL_S )) && [[ "${STATE_SWARM_PID}" == "${SWARM_PID}" ]]; then
        QUIET_SINCE="${STATE_QUIET_SINCE}"
        START="${STATE_START:-${NOW}}"
        POLLS="${STATE_POLLS:-0}"
        JCURSOR="${STATE_JOURNAL_CURSOR}"
        log "  resumed state from ${STATE}: quiet_since=${QUIET_SINCE:-none} polls=${POLLS} (last poll $(( NOW - STATE_LAST_POLL ))s ago)"
    else
        log "  stale state in ${STATE} (last poll too old, or Swarm restarted since) — starting a fresh window"
        rm -f "${STATE}"
    fi
fi
if [[ ${JOURNAL_CHECK} -eq 1 && -z "${JCURSOR}" ]]; then
    JCURSOR=$(journal_newest_cursor)
    if [[ -z "${JCURSOR}" ]]; then
        log "ABORT: the journal has no entries for identifier '${JOURNAL_ID}' (--journal-id to change, --no-journal-check to skip)"
        exit 2
    fi
fi

save_state() {
    if [[ -n "${STATE}" ]]; then
        printf 'STATE_QUIET_SINCE=%s\nSTATE_LAST_POLL=%s\nSTATE_START=%s\nSTATE_POLLS=%s\nSTATE_JOURNAL_CURSOR=%s\nSTATE_SWARM_PID=%s\n' \
            "${QUIET_SINCE}" "$(date +%s)" "${START}" "${POLLS}" "${JCURSOR}" "${SWARM_PID}" > "${STATE}"
    fi
}

# ---- poll loop -------------------------------------------------------------------------------------------
SWARM_MEM_PREV=""
SWARM_MEM_BASELINE=0
while true; do
    NOW=$(date +%s)
    POLLS=$((POLLS + 1))
    REASONS=()

    CURRENT_PID=$(swarm_pid)
    if [[ "${CURRENT_PID}" != "${SWARM_PID}" ]]; then
        REASONS+=("Swarm restarted (MainPID ${SWARM_PID} -> ${CURRENT_PID})")
        SWARM_PID="${CURRENT_PID}"
        SWARM_MEM_PREV=""
        SWARM_MEM_BASELINE=0
        new_session || true
    fi

    STATUS_LINE=$(global_status)
    if [[ -z "${STATUS_LINE}" || "${STATUS_LINE}" == ERR* ]]; then
        log "ABORT: GetGlobalStatus unusable after a retry: ${STATUS_LINE:0:200}"
        save_state
        exit 2
    fi
    read -r LIVE WAITING LOADING WBACK <<<"${STATUS_LINE}"
    if [[ "${LIVE}" != "0" || "${WAITING}" != "0" || "${LOADING}" != "0" ]]; then
        REASONS+=("Swarm queue (all sessions) live=${LIVE} waiting=${WAITING} loading=${LOADING}")
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
    # Line 1: foreign processes on the targets; line 2: SwarmUI memory per target card ("uuid=MiB …").
    GPU_EVAL=$(printf '%s\n' "${APPS}" | python3 -c '
import sys
targets = set(sys.argv[1].split())
swarm_pid = sys.argv[2]
foreign, swarm = [], {}
for line in sys.stdin:
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 3 or not parts[0] or parts[0] not in targets:
        continue
    uuid, pid, name = parts[0], parts[1], parts[2]
    mem = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else 0
    if pid == swarm_pid or "swarmui" in name.lower():
        swarm[uuid] = swarm.get(uuid, 0) + mem
    else:
        foreign.append(f"{uuid} pid={pid} {name} {mem} MiB")
print("; ".join(foreign))
print(" ".join(f"{u}={m}" for u, m in sorted(swarm.items())))' "${TARGET_UUIDS[*]}" "${SWARM_PID}")
    FOREIGN=$(printf '%s\n' "${GPU_EVAL}" | sed -n 1p)
    SWARM_MEM_NOW=$(printf '%s\n' "${GPU_EVAL}" | sed -n 2p)
    if [[ -n "${FOREIGN}" ]]; then
        REASONS+=("foreign process on a target card")
    fi
    # Compared only against a baseline taken earlier in THIS invocation (none on its first poll or after a restart).
    if [[ ${SWARM_MEM_BASELINE} -eq 1 ]]; then
        MOVED=$(python3 -c '
import sys
def parse(s):
    return {k: int(v) for k, v in (kv.split("=") for kv in s.split())} if s.strip() else {}
def show(v):
    return "absent" if v is None else str(v)
prev, now, limit = parse(sys.argv[1]), parse(sys.argv[2]), int(sys.argv[3])
moves = []
for u in sorted(set(prev) | set(now)):
    a, b = prev.get(u), now.get(u)
    if a is None or b is None or abs(b - a) > limit:
        moves.append(u[:12] + " " + show(a) + "->" + show(b) + " MiB")
print("; ".join(moves))' "${SWARM_MEM_PREV}" "${SWARM_MEM_NOW}" "${SWARM_MEM_MOVE_MIB}")
        if [[ -n "${MOVED}" ]]; then
            REASONS+=("SwarmUI memory moved on a target card: ${MOVED}")
        fi
    fi
    SWARM_MEM_PREV="${SWARM_MEM_NOW}"
    SWARM_MEM_BASELINE=1

    JNEW=0
    JREQS=""
    if [[ ${JOURNAL_CHECK} -eq 1 ]]; then
        if ! JOUT=$(journalctl --user -t "${JOURNAL_ID}" --after-cursor="${JCURSOR}" --show-cursor --no-pager -o short-iso 2>/dev/null); then
            log "ABORT: journalctl failed for '${JOURNAL_ID}' (--no-journal-check to skip)"
            save_state
            exit 2
        fi
        NEXT=$(printf '%s\n' "${JOUT}" | sed -n 's/^-- cursor: //p' | tail -1)
        [[ -n "${NEXT}" ]] && JCURSOR="${NEXT}"
        JBODY=$(printf '%s\n' "${JOUT}" | grep -v '^-- ')
        JNEW=$( [[ -n "${JBODY}" ]] && printf '%s\n' "${JBODY}" | wc -l || echo 0 )
        JREQS=$(printf '%s\n' "${JBODY}" | grep -E "${REQUEST_RE}")
        if [[ -n "${JREQS}" ]]; then
            REASONS+=("Swarm journal shows request/activity lines")
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

    log "poll ${POLLS}: global live=${LIVE} waiting=${WAITING} loading=${LOADING} waiting_backends=${WBACK} | 3060 backend: ${FORBID_STATE} | foreign on targets: ${FOREIGN:-none} | SwarmUI on targets: ${SWARM_MEM_NOW:-none} | journal: $( [[ ${JOURNAL_CHECK} -eq 1 ]] && echo "${JNEW} new line(s)" || echo off ) | quiet held ${HELD}s / ${WINDOW_S}s"
    if [[ ${#REASONS[@]} -gt 0 ]]; then
        printf '    reset: %s\n' "${REASONS[@]}"
    fi
    if [[ -n "${JREQS}" ]]; then
        printf '%s\n' "${JREQS}" | head -6 | cut -c1-200 | sed 's/^/    journal: /'
    fi
    save_state

    if [[ ${#REASONS[@]} -eq 0 && ${HELD} -ge ${WINDOW_S} ]]; then
        log "QUIET WINDOW OK: ${HELD}s (>= ${WINDOW_S}s): global queue empty, no request in the journal$( [[ ${JOURNAL_CHECK} -eq 1 ]] || echo " (journal check OFF)" ), no foreign process on:"
        printf '%s\n' "${TARGET_DESC}"
        log "  checked ${POLLS} poll(s) since $(date -u -d "@${START}" +%H:%M:%SZ); backend '${FORBID_TITLE}' stayed ${FORBID_STATE%% *}."
        log "  after the timed arm, confirm with: $0 --verify-since @$(date +%s)"
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
