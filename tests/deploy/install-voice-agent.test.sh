#!/usr/bin/env bash
# Unprivileged-namespace regression test for deploy/install-voice-agent.sh (same verification style as
# install-host-tuning.sh's own PR, generalized: this script has more to break -- the default publish path,
# restart-on-change, stale-file pruning, an ERR-trap cleanup -- so it gets a committed, repeatable harness instead
# of a one-off run quoted in a PR body).
#
# Needs unprivileged user+mount namespaces (CONFIG_USER_NS; on by default on current kernels; some hardened
# configs disable it for non-root, in which case run as root instead):
#
#   bash tests/deploy/install-voice-agent.test.sh            # re-execs itself under `unshare -rm`
#   unshare -rm -- bash tests/deploy/install-voice-agent.test.sh   # equivalent, explicit
#
# A new mount namespace means every mount this makes disappears when the process exits; nothing here touches the
# real host. /opt, /etc and /tmp are each replaced with their own tmpfs (with /etc/passwd, /etc/group restored
# from the real ones so root/$SUDO_USER name resolution still works) -- /tmp too, or a leaked-temp-dir check would
# be counting entries in the real, shared /tmp, where another process (this box runs several agents at once) can
# add or remove its own tmp.* between the "before" and "after" snapshot and produce a false pass or fail. A fake
# systemctl on PATH records calls and tracks per-unit enabled/active state in files, so apply/revert/idempotence
# can be checked without a real systemd; a fake sudo and dotnet exercise the default (no --publish-dir) publish
# path without a real, slow, disk-heavy build.
set -uo pipefail

if [[ "${1:-}" != "--in-namespace" ]]; then
    if [[ $(id -u) -ne 0 ]] && ! unshare -rm -- true 2>/dev/null; then
        echo "unprivileged user+mount namespaces are not available (and this is not root); see the comment at" >&2
        echo "the top of this script for how to run it as root instead." >&2
        exit 2
    fi
    exec unshare -rm -- bash "${BASH_SOURCE[0]}" --in-namespace
fi

REPO=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
SCRIPT="$REPO/deploy/install-voice-agent.sh"

pass=0
fail=0
check() {
    local desc=$1
    shift
    if "$@"; then
        echo "PASS: $desc"
        pass=$((pass + 1))
    else
        echo "FAIL: $desc"
        fail=$((fail + 1))
    fi
}
check_not() {
    local desc=$1
    shift
    if "$@"; then
        echo "FAIL: $desc"
        fail=$((fail + 1))
    else
        echo "PASS: $desc"
        pass=$((pass + 1))
    fi
}

# ---- isolate /opt, /etc and /tmp; keep root/name resolution working --------------------------------------------
passwd_snap=$(cat /etc/passwd)
group_snap=$(cat /etc/group)
nsswitch_snap=$(cat /etc/nsswitch.conf 2>/dev/null || true)

mount -t tmpfs tmpfs /opt
mount -t tmpfs tmpfs /etc
mount -t tmpfs tmpfs /tmp
printf '%s\n' "$passwd_snap" >/etc/passwd
printf '%s\n' "$group_snap" >/etc/group
[[ -n $nsswitch_snap ]] && printf '%s\n' "$nsswitch_snap" >/etc/nsswitch.conf
mkdir -p /etc/systemd/system

# Only now, so it lands inside the fresh, isolated /tmp above, not the real one.
SCRATCH=$(mktemp -d)
FAKE_BIN="$SCRATCH/bin"
export FAKE_SYSTEMCTL_STATE="$SCRATCH/systemctl-state"
FAKE_PUBLISH="$SCRATCH/fake-publish"
trap 'rm -rf -- "$SCRATCH"' EXIT

echo "id inside the namespace: $(id)"
echo "/opt, /etc and /tmp are now tmpfs: $(findmnt -no SOURCE,FSTYPE /opt 2>/dev/null) / $(findmnt -no SOURCE,FSTYPE /etc 2>/dev/null) / $(findmnt -no SOURCE,FSTYPE /tmp 2>/dev/null)"

mkdir -p "$FAKE_BIN" "$FAKE_SYSTEMCTL_STATE" "$FAKE_PUBLISH/voice-host" "$FAKE_PUBLISH/phone-gateway"

# Fake systemctl: records every call and tracks per-unit enabled/active state with marker files, so repeated
# apply/revert calls see the same enabled/active/already-installed branches the real systemctl would report.
cat >"$FAKE_BIN/systemctl" <<'SYSTEMCTL_EOF'
#!/usr/bin/env bash
set -euo pipefail
state="${FAKE_SYSTEMCTL_STATE:?FAKE_SYSTEMCTL_STATE must be set}"
mkdir -p "$state"
echo "systemctl $*" >>"$state/calls.log"
cmd=${1:-}
shift || true
args=()
now=0
for a in "$@"; do
    case "$a" in
        --now) now=1 ;;
        --quiet | --no-pager | --no-legend) : ;;
        *) args+=("$a") ;;
    esac
done
case "$cmd" in
    daemon-reload) exit 0 ;;
    enable)
        for u in "${args[@]}"; do touch "$state/$u.enabled"; ((now)) && touch "$state/$u.active"; done
        exit 0 ;;
    start | restart)
        for u in "${args[@]}"; do touch "$state/$u.active"; done
        exit 0 ;;
    stop)
        for u in "${args[@]}"; do rm -f "$state/$u.active"; done
        exit 0 ;;
    disable)
        for u in "${args[@]}"; do rm -f "$state/$u.enabled"; ((now)) && rm -f "$state/$u.active"; done
        exit 0 ;;
    is-enabled)
        [[ -f "$state/${args[0]}.enabled" ]] && exit 0 || exit 1 ;;
    is-active)
        [[ -f "$state/${args[0]}.active" ]] && exit 0 || exit 1 ;;
    status)
        for u in "${args[@]}"; do
            if [[ -f "$state/$u.active" ]]; then echo "$u: active (fake)"; else echo "$u: inactive (fake)"; fi
        done
        exit 0 ;;
    *) exit 0 ;;
esac
SYSTEMCTL_EOF

# Fake sudo: drops -u NAME (we're already root in the namespace) and execs the rest.
cat >"$FAKE_BIN/sudo" <<'SUDO_EOF'
#!/usr/bin/env bash
while [[ $# -gt 0 && $1 == -* ]]; do
    case $1 in
        -u) shift 2 ;;
        *) shift ;;
    esac
done
exec "$@"
SUDO_EOF

# Fake dotnet: `dotnet publish ... -o DIR` drops one stub dll named for DIR's basename, so publish_as_user()'s
# real code path runs without a real (slow, disk-heavy) build.
cat >"$FAKE_BIN/dotnet" <<'DOTNET_EOF'
#!/usr/bin/env bash
if [[ ${1:-} == publish ]]; then
    out=""
    prev=""
    for a in "$@"; do
        [[ $prev == "-o" ]] && out=$a
        prev=$a
    done
    [[ -n $out ]] || exit 1
    mkdir -p "$out"
    case "$(basename "$out")" in
        voice-host) echo stub >"$out/HartsyInference.VoiceHost.dll" ;;
        phone-gateway) echo stub >"$out/HartsyInference.PhoneGateway.dll" ;;
        *) echo stub >"$out/stub.dll" ;;
    esac
    exit 0
fi
exit 0
DOTNET_EOF

chmod +x "$FAKE_BIN/systemctl" "$FAKE_BIN/sudo" "$FAKE_BIN/dotnet"
echo stub >"$FAKE_PUBLISH/voice-host/HartsyInference.VoiceHost.dll"
echo stub >"$FAKE_PUBLISH/phone-gateway/HartsyInference.PhoneGateway.dll"
export PATH="$FAKE_BIN:$PATH"
echo "fake systemctl resolves to: $(command -v systemctl)"

run_script() { bash "$SCRIPT" "$@"; }

echo
echo "=== scenario: --dry-run --apply as EUID 0 (preview; nothing exists yet) ==="
check "dry-run --apply exits 0" run_script --dry-run --apply --publish-dir "$FAKE_PUBLISH"
check_not "dry run wrote nothing to /opt" test -e /opt/hartsyinference
check_not "dry run wrote nothing to /etc/hartsyinference" test -e /etc/hartsyinference/voice.json

echo
echo "=== scenario: unknown option is refused ==="
check_not "unknown option dies" run_script --bogus

echo
echo "=== scenario: --apply --revert together is refused ==="
check_not "--apply --revert together dies" run_script --apply --revert

echo
echo "=== scenario: --purge without --revert is refused ==="
check_not "--purge without --revert dies" run_script --purge --apply --publish-dir "$FAKE_PUBLISH"

echo
echo "=== scenario: real --apply (--publish-dir skips dotnet) ==="
check "apply #1 exits 0" run_script --apply --publish-dir "$FAKE_PUBLISH"
check "voice-host dll installed" test -f /opt/hartsyinference/voice-host/HartsyInference.VoiceHost.dll
check "phone-gateway dll installed" test -f /opt/hartsyinference/phone-gateway/HartsyInference.PhoneGateway.dll
check "opt tree is root:root" test "$(stat -c '%U:%G' /opt/hartsyinference)" = "root:root"
check "secrets dir is 0700" test "$(stat -c '%a' /etc/hartsyinference/secrets)" = "700"
check "phone-link-token is 0600" test "$(stat -c '%a' /etc/hartsyinference/secrets/phone-link-token)" = "600"
check "phone-link-token has content" test -s /etc/hartsyinference/secrets/phone-link-token
check "voice.json written" test -f /etc/hartsyinference/voice.json
check "phone.json written" test -f /etc/hartsyinference/phone.json
check "phone.json LAN profile: registrar empty" python3 -c "import json,sys; d=json.load(open('/etc/hartsyinference/phone.json')); sys.exit(0 if d['sip']['registrar']=='' else 1)"
check "phone.json LAN profile: allowAnyDestination true" python3 -c "import json,sys; d=json.load(open('/etc/hartsyinference/phone.json')); sys.exit(0 if d['sip']['allowAnyDestination'] is True else 1)"
check "phone.json LAN profile: destinationPrefixes empty" python3 -c "import json,sys; d=json.load(open('/etc/hartsyinference/phone.json')); sys.exit(0 if d['sip']['destinationPrefixes']==[] else 1)"
check "phone.json tickCpu is 7" python3 -c "import json,sys; d=json.load(open('/etc/hartsyinference/phone.json')); sys.exit(0 if d['media']['tickCpu']==7 else 1)"
check "voice.json denoise forced true" python3 -c "import json,sys; d=json.load(open('/etc/hartsyinference/voice.json')); sys.exit(0 if d['models']['denoise'] is True else 1)"
check "voice.json keeps template's /run/credentials path" grep -q "/run/credentials/hartsyinference-voice-host.service/phone-link-token" /etc/hartsyinference/voice.json
check "voice.json has no // comments left" bash -c '! grep -q "//" /etc/hartsyinference/voice.json'
check "units installed" test -f /etc/systemd/system/hartsyinference-voice-host.service -a -f /etc/systemd/system/hartsyinference-phone-gateway.service
check "server unit NOT installed (none pre-existed)" test ! -e /etc/systemd/system/hartsyinference-server.service
check "host unit enabled (fake)" test -f "$FAKE_SYSTEMCTL_STATE/hartsyinference-voice-host.service.enabled"
check "host unit active (fake)" test -f "$FAKE_SYSTEMCTL_STATE/hartsyinference-voice-host.service.active"
check "gateway unit enabled (fake)" test -f "$FAKE_SYSTEMCTL_STATE/hartsyinference-phone-gateway.service.enabled"
check "gateway unit active (fake)" test -f "$FAKE_SYSTEMCTL_STATE/hartsyinference-phone-gateway.service.active"
check "daemon-reload was called" grep -q "daemon-reload" "$FAKE_SYSTEMCTL_STATE/calls.log"

echo
echo "=== scenario: idempotence -- apply #2 changes nothing ==="
: >"$FAKE_SYSTEMCTL_STATE/calls.log"
secrets_before=$(sha256sum /etc/hartsyinference/secrets/* | sort)
check "apply #2 exits 0" run_script --apply --publish-dir "$FAKE_PUBLISH"
secrets_after=$(sha256sum /etc/hartsyinference/secrets/* | sort)
check "secrets unchanged by apply #2" test "$secrets_before" = "$secrets_after"
check_not "apply #2 did not call daemon-reload (nothing changed)" grep -q "daemon-reload" "$FAKE_SYSTEMCTL_STATE/calls.log"
check_not "apply #2 did not re-enable the host (already enabled)" grep -q "^systemctl enable .*hartsyinference-voice-host" "$FAKE_SYSTEMCTL_STATE/calls.log"
check_not "apply #2 did not restart the host (tree unchanged)" grep -q "^systemctl restart .*hartsyinference-voice-host" "$FAKE_SYSTEMCTL_STATE/calls.log"

echo
echo "=== scenario: a changed tree restarts an already-active unit ==="
echo "changed stub" >"$FAKE_PUBLISH/voice-host/HartsyInference.VoiceHost.dll"
: >"$FAKE_SYSTEMCTL_STATE/calls.log"
check "apply #3 (changed tree) exits 0" run_script --apply --publish-dir "$FAKE_PUBLISH"
check "the changed binary was installed" grep -q "changed stub" /opt/hartsyinference/voice-host/HartsyInference.VoiceHost.dll
check "the host was restarted, not left on the old binary" grep -q "^systemctl restart .*hartsyinference-voice-host" "$FAKE_SYSTEMCTL_STATE/calls.log"
check_not "the gateway (unchanged) was not restarted" grep -q "^systemctl restart .*hartsyinference-phone-gateway" "$FAKE_SYSTEMCTL_STATE/calls.log"
echo stub >"$FAKE_PUBLISH/voice-host/HartsyInference.VoiceHost.dll"

echo
echo "=== scenario: hand-edited voice.json (operator config) survives re-apply ==="
python3 -c "
import json
d = json.load(open('/etc/hartsyinference/voice.json'))
d['agent']['greeting'] = 'Operator edited this, must survive.'
json.dump(d, open('/etc/hartsyinference/voice.json', 'w'), indent=2)
"
check "apply #4 exits 0" run_script --apply --publish-dir "$FAKE_PUBLISH"
check "hand-edited greeting survived" grep -q "Operator edited this, must survive." /etc/hartsyinference/voice.json

echo
echo "=== scenario: hand-edited unit file is put back (units are code, not operator config) ==="
echo "# drifted" >>/etc/systemd/system/hartsyinference-voice-host.service
: >"$FAKE_SYSTEMCTL_STATE/calls.log"
check "apply #5 exits 0" run_script --apply --publish-dir "$FAKE_PUBLISH"
check_not "drifted unit content was overwritten" grep -q "# drifted" /etc/systemd/system/hartsyinference-voice-host.service
check "apply #5 called daemon-reload (unit changed)" grep -q "daemon-reload" "$FAKE_SYSTEMCTL_STATE/calls.log"

echo
echo "=== scenario: symlinked destination is refused, not followed ==="
rm -f /etc/hartsyinference/phone.json
ln -s /etc/hartsyinference/secrets/sip-password /etc/hartsyinference/phone.json
check_not "apply refuses to write through a symlinked config" run_script --apply --publish-dir "$FAKE_PUBLISH"
check "the symlink itself is untouched" test -L /etc/hartsyinference/phone.json
rm -f /etc/hartsyinference/phone.json
check "apply #6 recreates phone.json after the symlink is removed" run_script --apply --publish-dir "$FAKE_PUBLISH"

echo
echo "=== scenario: a pre-existing server unit is refreshed; one that never existed stays absent ==="
echo "# stale pre-existing server unit" >/etc/systemd/system/hartsyinference-server.service
check "apply #7 exits 0" run_script --apply --publish-dir "$FAKE_PUBLISH"
check_not "the pre-existing server unit was refreshed from the repo" grep -q "stale pre-existing" /etc/systemd/system/hartsyinference-server.service
check "the refreshed server unit matches the repo source" cmp -s /etc/systemd/system/hartsyinference-server.service "$REPO/deploy/systemd/hartsyinference-server.service"

echo
echo "=== scenario: stale-file pruning removes a dropped file but keeps a legitimate empty directory ==="
# put_tree only compares files; an empty directory alone is not a "change" that would make it re-copy at all, so
# this also adds a real file -- a believable new release (adds plugins/, a readme in it, and an empty subfolder
# for drop-in extensions), not just an empty directory in isolation.
mkdir -p "$FAKE_PUBLISH/voice-host/plugins/empty-on-purpose"
echo "drop .dll plugins here" >"$FAKE_PUBLISH/voice-host/plugins/README.txt"
check "apply #8 (adds plugins/ and the empty-on-purpose dir) exits 0" run_script --apply --publish-dir "$FAKE_PUBLISH"
check "the empty-on-purpose directory exists" test -d /opt/hartsyinference/voice-host/plugins/empty-on-purpose
touch /opt/hartsyinference/voice-host/stale-leftover.dll
check "apply #9 (with a stale leftover file) exits 0" run_script --apply --publish-dir "$FAKE_PUBLISH"
check_not "the stale leftover file was removed" test -e /opt/hartsyinference/voice-host/stale-leftover.dll
check "the legitimate empty directory survived" test -d /opt/hartsyinference/voice-host/plugins/empty-on-purpose
rm -rf "$FAKE_PUBLISH/voice-host/plugins"

echo
echo "=== scenario: a failed publish (second dotnet call fails) does not leak the temp dir ==="
cat >"$FAKE_BIN/dotnet" <<'DOTNET_FAIL_EOF'
#!/usr/bin/env bash
if [[ ${1:-} == publish ]]; then
    out=""; prev=""
    for a in "$@"; do [[ $prev == "-o" ]] && out=$a; prev=$a; done
    case "$(basename "$out")" in
        voice-host) mkdir -p "$out"; echo stub >"$out/HartsyInference.VoiceHost.dll"; exit 0 ;;
        phone-gateway) echo "simulated publish failure" >&2; exit 1 ;;
    esac
fi
exit 0
DOTNET_FAIL_EOF
export SUDO_USER=hartsy
tmp_before=$(find /tmp -maxdepth 1 -name 'tmp.*' 2>/dev/null | wc -l)
check_not "apply with a failing publish exits non-zero" run_script --apply
tmp_after=$(find /tmp -maxdepth 1 -name 'tmp.*' 2>/dev/null | wc -l)
check "no temp dir leaked by the failed publish" test "$tmp_before" = "$tmp_after"
unset SUDO_USER

echo
echo "=== scenario: the DEFAULT apply path (no --publish-dir), real publish_as_user() ==="
cat >"$FAKE_BIN/dotnet" <<'DOTNET_EOF2'
#!/usr/bin/env bash
if [[ ${1:-} == publish ]]; then
    out=""; prev=""
    for a in "$@"; do [[ $prev == "-o" ]] && out=$a; prev=$a; done
    [[ -n $out ]] || exit 1
    mkdir -p "$out"
    case "$(basename "$out")" in
        voice-host) echo stub >"$out/HartsyInference.VoiceHost.dll" ;;
        phone-gateway) echo stub >"$out/HartsyInference.PhoneGateway.dll" ;;
    esac
    exit 0
fi
exit 0
DOTNET_EOF2
export SUDO_USER=hartsy
run_script --revert --purge >/dev/null 2>&1 || true
check "default-path apply (no --publish-dir) exits 0" run_script --apply
check "voice-host dll installed via the default path" test -f /opt/hartsyinference/voice-host/HartsyInference.VoiceHost.dll
check "phone-gateway dll installed via the default path" test -f /opt/hartsyinference/phone-gateway/HartsyInference.PhoneGateway.dll

echo
echo "=== scenario: a symlinked destination with the default publish path leaks no temp dir ==="
# put_tree's refuse_symlink dies via exit, not a plain command failure -- the specific path an ERR trap would
# miss and only an EXIT trap catches. Still the default (no --publish-dir) path, so this exercises
# publish_as_user()'s real mktemp -d too, not a fake one.
rm -f /opt/hartsyinference/voice-host/HartsyInference.VoiceHost.dll
ln -s /etc/hartsyinference/secrets/sip-password /opt/hartsyinference/voice-host/HartsyInference.VoiceHost.dll
tmp_before=$(find /tmp -maxdepth 1 -name 'tmp.*' 2>/dev/null | wc -l)
check_not "apply refuses to write through the symlinked binary" run_script --apply
tmp_after=$(find /tmp -maxdepth 1 -name 'tmp.*' 2>/dev/null | wc -l)
check "no temp dir leaked by the symlink refusal" test "$tmp_before" = "$tmp_after"
rm -f /opt/hartsyinference/voice-host/HartsyInference.VoiceHost.dll
check "apply recovers once the symlink is removed" run_script --apply
unset SUDO_USER

echo
echo "=== scenario: --revert (no --purge) stops/disables/removes units, leaves binaries+configs+secrets ==="
check "revert exits 0" run_script --revert
check_not "host unit file removed" test -e /etc/systemd/system/hartsyinference-voice-host.service
check_not "gateway unit file removed" test -e /etc/systemd/system/hartsyinference-phone-gateway.service
check "server unit NOT touched by revert" test -f /etc/systemd/system/hartsyinference-server.service
check_not "host unit no longer active (fake)" test -f "$FAKE_SYSTEMCTL_STATE/hartsyinference-voice-host.service.active"
check_not "gateway unit no longer enabled (fake)" test -f "$FAKE_SYSTEMCTL_STATE/hartsyinference-phone-gateway.service.enabled"
check "binaries left in place" test -f /opt/hartsyinference/voice-host/HartsyInference.VoiceHost.dll
check "voice.json left in place" test -f /etc/hartsyinference/voice.json
check "secrets left in place" test -f /etc/hartsyinference/secrets/phone-link-token

echo
echo "=== scenario: idempotent second revert ==="
check "revert #2 exits 0" run_script --revert

echo
echo "=== scenario: --revert --purge removes binaries, configs and secrets ==="
check "revert --purge exits 0" run_script --revert --purge
check_not "voice-host binaries purged" test -e /opt/hartsyinference/voice-host
check_not "voice.json purged" test -e /etc/hartsyinference/voice.json
check_not "secrets dir purged" test -e /etc/hartsyinference/secrets

echo
echo "=== scenario: clean re-apply from scratch after purge ==="
check "apply after purge exits 0" run_script --apply --publish-dir "$FAKE_PUBLISH"
check "voice.json exists again" test -f /etc/hartsyinference/voice.json
check "host unit active again (fake)" test -f "$FAKE_SYSTEMCTL_STATE/hartsyinference-voice-host.service.active"
run_script --revert --purge >/dev/null

echo
echo "================================================================"
echo "RESULT: $pass passed, $fail failed"
exit "$((fail > 0))"
