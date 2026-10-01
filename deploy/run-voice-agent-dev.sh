#!/usr/bin/env bash
# No-root LAN softphone check for the phone-call voice agent (deploy/README.md, "LAN softphone call"). Publishes both
# executables, writes a LAN-profile voice.json/phone.json under $XDG_RUNTIME_DIR and ~/.config/hartsyinference, and
# runs them as plain user processes (no systemd, no sudo). For the production install see install-voice-agent.sh.
#
# With no option: publish, generate secrets and configs if missing, check SwarmUI is quiet, start the host then the
# gateway, print the dial target, and tail both logs until Ctrl+C.
#   --no-build       reuse the last publish output under the publish directory instead of running `dotnet publish`
#   --force          start even if the quiet-window spot check finds SwarmUI busy (or can't tell)
#   --host-timeout N seconds to wait for the voice host to warm up and open its socket (default 300, a cold GPU load)
#   --stop           stop a running instance (both processes, cleanly) and exit; also what Ctrl+C does
#
# State:
#   publish output   ~/.local/share/hartsyinference/voice-agent-dev/{voice-host,phone-gateway}/
#   configs+secrets  ~/.config/hartsyinference/voice-agent-dev/{voice.json,phone.json,secrets/,logs/}
#   socket+pidfiles  $XDG_RUNTIME_DIR/hartsyinference/ (falls back to /run/user/<uid> if XDG_RUNTIME_DIR is unset)
# An existing voice.json/phone.json is never overwritten: a second run that finds one already there leaves it alone
# and prints a diff against what the LAN profile would write. Secrets already present are reused as-is; a missing
# one is generated with `openssl rand -hex 32` (0600, umask 077) and its value is never printed.

set -euo pipefail

readonly repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
readonly documented_lan_ip="192.168.10.188"

no_build=0
force=0
do_stop=0
host_timeout=300
readonly gateway_timeout=30

say() { printf '[voice-agent-dev] %s\n' "$*"; }
die() {
    printf '[voice-agent-dev] error: %s\n' "$*" >&2
    exit 1
}

usage() {
    cat <<'EOF'
Usage: deploy/run-voice-agent-dev.sh [--no-build] [--force] [--host-timeout SECONDS] | --stop
  (no option)       publish, generate secrets/configs if missing, check SwarmUI is quiet, start both
                     processes, print the dial target, tail both logs until Ctrl+C
  --no-build        reuse the existing publish output instead of running dotnet publish
  --force           start even if SwarmUI looks busy (or the quiet-window check itself failed)
  --host-timeout N  seconds to wait for the voice host to warm up (default 300)
  --stop            stop a running instance and exit (same as Ctrl+C on the foreground run)
No root, no sudo, no systemd: a plain-user LAN check. See install-voice-agent.sh for the production install.
EOF
}

# --- paths --------------------------------------------------------------------------------------------------------
publish_dir=${HARTSY_VOICE_DEV_PUBLISH_DIR:-"$HOME/.local/share/hartsyinference/voice-agent-dev"}
state_dir=${HARTSY_VOICE_DEV_STATE_DIR:-"$HOME/.config/hartsyinference/voice-agent-dev"}
secrets_dir="$state_dir/secrets"
log_dir="$state_dir/logs"
runtime_base=${XDG_RUNTIME_DIR:-"/run/user/$(id -u)"}
socket_dir="$runtime_base/hartsyinference"
voice_json="$state_dir/voice.json"
phone_json="$state_dir/phone.json"
socket_path="$socket_dir/phone.sock"
host_pidfile="$socket_dir/voice-host.pid"
gw_pidfile="$socket_dir/phone-gateway.pid"
host_log="$log_dir/voice-host.log"
gw_log="$log_dir/phone-gateway.log"

host_pid=""
gw_pid=""
tail_host_pid=""
tail_gw_pid=""

# --- process helpers -----------------------------------------------------------------------------------------------

# True when $1's pidfile names a live process whose cmdline contains $2 (guards against a recycled, unrelated pid).
is_running() {
    local pidfile=$1 marker=$2 pid
    [[ -f $pidfile ]] || return 1
    pid=$(<"$pidfile")
    [[ -n $pid ]] || return 1
    kill -0 "$pid" 2>/dev/null || return 1
    tr '\0' '\n' <"/proc/$pid/cmdline" 2>/dev/null | grep -qF "$marker"
}

# Stops the process named by $1's pidfile (cmdline must contain $2), TERM then wait up to $3 s, then KILL.
stop_one() {
    local pidfile=$1 marker=$2 timeout=$3 pid waited=0
    [[ -f $pidfile ]] || return 0
    pid=$(<"$pidfile")
    if [[ -n $pid ]] && kill -0 "$pid" 2>/dev/null && tr '\0' '\n' <"/proc/$pid/cmdline" 2>/dev/null | grep -qF "$marker"; then
        say "stopping pid $pid ($marker)"
        kill -TERM "$pid" 2>/dev/null || true
        while kill -0 "$pid" 2>/dev/null && ((waited < timeout)); do
            sleep 1
            waited=$((waited + 1))
        done
        if kill -0 "$pid" 2>/dev/null; then
            say "pid $pid did not exit after ${timeout}s; sending SIGKILL"
            kill -KILL "$pid" 2>/dev/null || true
        fi
    fi
    rm -f -- "$pidfile"
}

stop_all() {
    if [[ -n $tail_host_pid ]]; then
        kill "$tail_host_pid" 2>/dev/null || true
    fi
    if [[ -n $tail_gw_pid ]]; then
        kill "$tail_gw_pid" 2>/dev/null || true
    fi
    # Gateway first: it owns no models and nothing else depends on it; stopping it first means it is not left
    # trying to talk to a host that just disappeared.
    stop_one "$gw_pidfile" "HartsyInference.PhoneGateway.dll" 15
    stop_one "$host_pidfile" "HartsyInference.VoiceHost.dll" 30
}

# --- setup ----------------------------------------------------------------------------------------------------------

gen_secrets() {
    install -d -m 0700 -- "$secrets_dir"
    local name f
    for name in phone-link-token phone-admin-token sip-password; do
        f="$secrets_dir/$name"
        if [[ -f $f ]]; then
            say "$f: already present"
        else
            (
                umask 077
                openssl rand -hex 32 >"$f"
            )
            chmod 0600 -- "$f"
            say "generated $f (0600; value not printed)"
        fi
    done
}

# Installs $1 (a freshly rendered temp file) as $2 only if $2 does not exist yet; otherwise leaves $2 alone and
# prints a diff. Removes $1 either way.
install_rendered() {
    local tmp=$1 dest=$2
    if [[ -f $dest ]]; then
        if cmp -s -- "$tmp" "$dest"; then
            say "$dest: already matches the LAN profile"
        else
            say "$dest already exists and differs from the LAN profile this run would write; leaving it alone."
            say "diff (existing vs. what the LAN profile would write):"
            diff -u -- "$dest" "$tmp" | sed 's/^/    /' || true
        fi
        rm -f -- "$tmp"
    else
        install -m 0644 -- "$tmp" "$dest"
        rm -f -- "$tmp"
        say "wrote $dest"
    fi
}

# Both example templates carry `//` comments System.Text.Json is told to skip (JsonCommentHandling.Skip); python's
# json module is not, so full-line comments are stripped before parsing. Never regex `//` mid-line: a future
# registrar or STUN value could legitimately contain it.
render_voice_json() {
    local template="$repo_root/src/HartsyInference.VoiceHost/voice.example.json" tmp
    [[ -f $template ]] || die "missing $template"
    tmp=$(mktemp)
    python3 - "$template" "$tmp" "$socket_path" "$secrets_dir/phone-link-token" <<'PY'
import json, sys
template, out, socket_path, token_file = sys.argv[1:5]
with open(template, encoding="utf-8") as f:
    text = "\n".join(line for line in f if line.strip()[:2] != "//")
cfg = json.loads(text)
cfg["link"]["socketPath"] = socket_path
cfg["link"]["tokenFile"] = token_file
# The LAN profile always keeps the denoiser on, matching VoiceAgentOptions' own default: the template carrying a
# conflicting hard-coded value (as it once did) must not silently turn it off for a LAN check either.
cfg["models"]["denoise"] = True
with open(out, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
    f.write("\n")
PY
    install_rendered "$tmp" "$voice_json"
}

render_phone_json() {
    local template="$repo_root/src/HartsyInference.PhoneGateway/phone.example.json" tmp
    [[ -f $template ]] || die "missing $template"
    tmp=$(mktemp)
    python3 - "$template" "$tmp" "$socket_path" "$secrets_dir/phone-link-token" "$secrets_dir/phone-admin-token" \
        "$secrets_dir/sip-password" <<'PY'
import json, sys
template, out, socket_path, link_token, admin_token, sip_password = sys.argv[1:7]
with open(template, encoding="utf-8") as f:
    text = "\n".join(line for line in f if line.strip()[:2] != "//")
cfg = json.loads(text)
# The LAN profile: no registrar, no trunk, so no destination restriction; the sip-password file is a placeholder
# (never read without a registrar) generated anyway, for parity with how the production install writes one.
cfg["sip"]["registrar"] = ""
cfg["sip"]["destinationPrefixes"] = []
cfg["sip"]["allowAnyDestination"] = True
cfg["sip"]["passwordFile"] = sip_password
cfg["link"]["socketPath"] = socket_path
cfg["link"]["tokenFile"] = link_token
cfg["admin"]["tokenFile"] = admin_token
with open(out, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
    f.write("\n")
PY
    install_rendered "$tmp" "$phone_json"
}

check_quiet() {
    local rc=0
    say "quiet-window spot check (SwarmUI puts Qwen3 on the 4090, speech models on the 3060)..."
    # --max-minutes 0 alongside --minutes 0: without it, a busy Swarm leaves the script polling forever instead of
    # reporting "busy" (--max-minutes unset means wait forever; --minutes 0 only shortens the window once it's quiet).
    "$repo_root/tests/swarm-quiet-window.sh" --minutes 0 --max-minutes 0 || rc=$?
    if ((rc == 0)); then
        say "quiet-window spot check: clean"
        return 0
    fi
    say "quiet-window spot check: not clean (exit $rc) -- SwarmUI looks busy, or the check could not tell"
    if ((force)); then
        say "--force: starting anyway"
        return 0
    fi
    die "refusing to start while SwarmUI may be busy; pass --force once you've checked yourself, or wait and retry"
}

verify_published() {
    local v="$publish_dir/voice-host/HartsyInference.VoiceHost.dll" g="$publish_dir/phone-gateway/HartsyInference.PhoneGateway.dll"
    [[ -f $v ]] || die "--no-build but $v is missing; run once without --no-build first"
    [[ -f $g ]] || die "--no-build but $g is missing; run once without --no-build first"
    say "--no-build: reusing $publish_dir"
}

publish_all() {
    say "publishing voice host -> $publish_dir/voice-host"
    dotnet publish -c Release "$repo_root/src/HartsyInference.VoiceHost" -o "$publish_dir/voice-host"
    say "publishing phone gateway -> $publish_dir/phone-gateway"
    dotnet publish -c Release "$repo_root/src/HartsyInference.PhoneGateway" -o "$publish_dir/phone-gateway"
}

note_fifo() {
    local rt
    rt=$(ulimit -r 2>/dev/null || echo 0)
    if ((rt < 50)); then
        say "note: ulimit -r is ${rt} (<50); the gateway's RTP tick thread runs on the default scheduler, not" \
            "SCHED_FIFO (it logs its own refusal and continues; fine for this check). See" \
            "deploy/install-host-tuning.sh for the production rtprio limit."
    fi
}

# --- start/wait -------------------------------------------------------------------------------------------------

start_host() {
    : >"$host_log"
    say "starting voice host -> $host_log"
    (exec dotnet "$publish_dir/voice-host/HartsyInference.VoiceHost.dll" --config "$voice_json") >>"$host_log" 2>&1 &
    host_pid=$!
    echo "$host_pid" >"$host_pidfile"
    local deadline=$((SECONDS + host_timeout)) warm=0 sock=0
    while ((SECONDS < deadline)); do
        if ! kill -0 "$host_pid" 2>/dev/null; then
            say "the voice host exited before it was ready; last lines:"
            tail -n 40 -- "$host_log" | sed 's/^/    /'
            die "voice host start failed"
        fi
        if grep -q "loaded and warm in" "$host_log" 2>/dev/null; then
            warm=1
        fi
        if [[ -S $socket_path ]]; then
            sock=1
        fi
        if ((warm && sock)); then
            say "voice host ready: $(grep "loaded and warm in" "$host_log" | tail -1)"
            return 0
        fi
        sleep 1
    done
    say "timed out after ${host_timeout}s waiting for the voice host; last lines:"
    tail -n 40 -- "$host_log" | sed 's/^/    /'
    die "voice host never became ready (--host-timeout to wait longer)"
}

start_gateway() {
    : >"$gw_log"
    say "starting phone gateway -> $gw_log"
    (exec dotnet "$publish_dir/phone-gateway/HartsyInference.PhoneGateway.dll" --config "$phone_json") >>"$gw_log" 2>&1 &
    gw_pid=$!
    echo "$gw_pid" >"$gw_pidfile"
    local deadline=$((SECONDS + gateway_timeout)) started=0 linked=0
    while ((SECONDS < deadline)); do
        if ! kill -0 "$gw_pid" 2>/dev/null; then
            say "the phone gateway exited before it was ready; last lines:"
            tail -n 40 -- "$gw_log" | sed 's/^/    /'
            die "phone gateway start failed"
        fi
        if grep -qE "Gateway .* started\." "$gw_log" 2>/dev/null; then
            started=1
        fi
        if grep -q "Phone gateway connected" "$host_log" 2>/dev/null; then
            linked=1
        fi
        if ((started && linked)); then
            say "phone gateway started and linked to the host"
            return 0
        fi
        sleep 1
    done
    say "timed out after ${gateway_timeout}s waiting for the phone gateway; last lines (gateway, then host):"
    tail -n 20 -- "$gw_log" | sed 's/^/    /'
    tail -n 20 -- "$host_log" | sed 's/^/    /'
    die "phone gateway never became ready"
}

dial_target() {
    local ip
    ip=$(ip -4 route get 1.1.1.1 2>/dev/null | sed -n 's/.* src \([0-9.]*\).*/\1/p') || true
    if [[ -z $ip ]]; then
        ip=$(hostname -I 2>/dev/null | awk '{print $1}') || true
    fi
    if [[ -z $ip ]]; then
        say "could not detect a LAN address; using the documented box address"
        ip=$documented_lan_ip
    elif [[ $ip != "$documented_lan_ip" ]]; then
        say "note: detected LAN address $ip differs from the documented $documented_lan_ip; dialling the detected one"
    fi
    say "dial sip:agent@${ip}:5060 (PCMU/PCMA) from a softphone on the LAN"
}

tail_both() {
    tail -n0 -F -- "$host_log" 2>/dev/null | sed -u 's/^/[voice-host]    /' &
    tail_host_pid=$!
    tail -n0 -F -- "$gw_log" 2>/dev/null | sed -u 's/^/[phone-gateway] /' &
    tail_gw_pid=$!
    say "tailing both logs below; Ctrl+C ends both cleanly (or run with --stop from another shell)."
    while kill -0 "$host_pid" 2>/dev/null && kill -0 "$gw_pid" 2>/dev/null; do
        sleep 1
    done
    say "a process exited on its own; stopping the other."
}

main() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --no-build) no_build=1; shift ;;
            --force) force=1; shift ;;
            --stop) do_stop=1; shift ;;
            --host-timeout)
                [[ ${2:-} =~ ^[0-9]+$ ]] || die "--host-timeout needs a number of seconds"
                host_timeout=$2
                shift 2
                ;;
            -h | --help) usage; exit 0 ;;
            *) usage >&2; die "unknown option: $1" ;;
        esac
    done

    if ((do_stop)); then
        stop_all
        say "stopped (if it was running)"
        exit 0
    fi

    if is_running "$host_pidfile" "HartsyInference.VoiceHost.dll" || is_running "$gw_pidfile" "HartsyInference.PhoneGateway.dll"; then
        die "already running; stop it first: $0 --stop"
    fi

    trap 'stop_all; exit 0' INT TERM

    install -d -m 0700 -- "$state_dir" "$socket_dir"
    install -d -- "$log_dir"
    gen_secrets
    render_voice_json
    render_phone_json
    check_quiet
    if ((no_build)); then
        verify_published
    else
        publish_all
    fi
    start_host
    note_fifo
    start_gateway
    dial_target
    tail_both
    stop_all
}

main "$@"
