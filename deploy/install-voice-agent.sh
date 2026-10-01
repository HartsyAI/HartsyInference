#!/usr/bin/env bash
# Production install of the phone-call voice agent (deploy/README.md, "Deploying the phone-call voice agent"): the
# voice host and phone gateway, published, installed under /opt and /etc, and run under systemd. Same conventions as
# install-host-tuning.sh.
#
# No option, or --dry-run: prints every file it would write, in full or by SHA-256, and every change; needs no root.
# --apply:  publishes both executables as the invoking (non-root) user, installs them under
#           /opt/hartsyinference/{voice-host,phone-gateway} (root:root, 0755), generates any missing secret under
#           /etc/hartsyinference/secrets (root, 0700 dir, 0600 files, never printed), writes
#           /etc/hartsyinference/{voice.json,phone.json} from the example templates with the LAN profile ONLY if
#           absent (an existing file is never overwritten), copies the two units (and the server unit, but only if
#           one is already installed), reloads systemd, then enables and starts the voice host and the gateway.
#           Running it again changes nothing that is already correct.
# --revert: stops, disables and removes the two units. Binaries, configs and secrets are left in place unless
#           --purge is also given, in which case they are removed too. Never touches the server unit.
# --publish-dir DIR   use DIR/voice-host and DIR/phone-gateway (already published, e.g. by run-voice-agent-dev.sh)
#                      instead of running `dotnet publish`; skips the publish step entirely.
# --apply and --revert need root (sudo deploy/install-voice-agent.sh --apply), run as the account that will own
# /opt/hartsyinference and /etc/hartsyinference's secrets (the units' User=); the build itself still runs as the
# invoking user via $SUDO_USER, never as root.

set -euo pipefail

readonly repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
readonly opt_dir=/opt/hartsyinference
readonly etc_dir=/etc/hartsyinference
readonly secrets_dir=$etc_dir/secrets
readonly unit_dir=/etc/systemd/system
readonly wants_dir=$unit_dir/multi-user.target.wants
readonly host_unit=hartsyinference-voice-host.service
readonly gateway_unit=hartsyinference-phone-gateway.service
readonly server_unit=hartsyinference-server.service
readonly voice_host_app=voice-host
readonly phone_gateway_app=phone-gateway

dry_run=1
purge=0
failures=0
wrote=0
publish_dir=""

say() { printf '[install-voice-agent] %s\n' "$*"; }
die() {
    printf '[install-voice-agent] error: %s\n' "$*" >&2
    exit 1
}

usage() {
    cat <<'EOF'
Usage: deploy/install-voice-agent.sh [--dry-run] [--apply | --revert] [--purge] [--publish-dir DIR]
  (no option), --dry-run  print what --apply (or --revert) would write and change; needs no root
  --apply                 publish, install under /opt and /etc, enable and start both units
  --revert                stop, disable and remove the two units; binaries/configs/secrets stay unless --purge
  --purge                 with --revert: also remove /opt/hartsyinference, the configs and the secrets
  --publish-dir DIR       reuse DIR/voice-host and DIR/phone-gateway instead of running dotnet publish
--apply and --revert need root; running either again changes nothing already correct.
EOF
}

# ---- generic helpers (same contract as install-host-tuning.sh) ---------------------------------------------------

run() {
    if ((dry_run)); then
        say "would run: $*"
    else
        say "running: $*"
        "$@"
    fi
}

refuse_symlink() {
    if [[ -L $1 ]]; then
        die "$1 is a symlink; refusing to write through it (remove it, then run again)"
    fi
}

# Installs $1 as $2 (owner:group $3, mode $4) unless $2 already holds the same bytes. Sets wrote=1 when it writes.
put_file() {
    local src=$1 dst=$2 owner=$3 mode=$4
    wrote=0
    refuse_symlink "$dst"
    if [[ -f $dst ]] && cmp -s -- "$src" "$dst"; then
        say "$dst: already installed"
        return 0
    fi
    wrote=1
    if ((dry_run)); then
        say "would write $dst ($owner, $mode):"
        sed 's/^/    /' <"$src"
    else
        install -D -o "${owner%%:*}" -g "${owner##*:}" -m "$mode" -- "$src" "$dst"
        say "wrote $dst"
    fi
}

drop_file() {
    local dst=$1
    wrote=0
    if [[ ! -e $dst && ! -L $dst ]]; then
        say "$dst: not present"
        return 0
    fi
    wrote=1
    run rm -f -- "$dst"
}

# ---- publish ------------------------------------------------------------------------------------------------------

# Publishes both executables as $SUDO_USER (never as root) into a temp dir that user owns; echoes its path. Needs
# real root to actually run (dry run only describes it).
publish_as_user() {
    [[ -n ${SUDO_USER:-} ]] || die "no \$SUDO_USER; run via sudo as your login user, not logged in as root: sudo bash $0 --apply"
    command -v dotnet >/dev/null || die "dotnet not found on PATH (sudo's secure_path can hide a user-local" \
        "~/.dotnet install that \$SUDO_USER's own shell finds fine; publish yourself with" \
        "run-voice-agent-dev.sh and pass its output to --publish-dir instead)"
    local tmp
    tmp=$(sudo -u "$SUDO_USER" mktemp -d)
    # Belt and suspenders, not the primary defense: called as `src_dir=$(publish_as_user)`, a plain failing
    # command below (dotnet publish returning non-zero, not an explicit exit) would not actually abort this
    # function under set -e -- bash does not apply errexit's abort inside a command substitution assigned to a
    # variable, a well-known gotcha, confirmed by testing it directly: execution would fall through to
    # `trap - EXIT` and return an incomplete $tmp as if nothing had failed. Hence the explicit `|| { ...; die; }`
    # on each publish below, rather than trusting set -e to stop at the first one that fails.
    trap 'rm -rf -- "$tmp"' EXIT
    # Everything here must go to stderr: the caller captures this function's stdout as the return value
    # (src_dir=$(publish_as_user)), and say()/dotnet publish both write to stdout otherwise, which would corrupt
    # the path with log lines.
    say "publishing as $SUDO_USER -> $tmp" >&2
    sudo -u "$SUDO_USER" dotnet publish -c Release "$repo_root/src/HartsyInference.VoiceHost" -o "$tmp/$voice_host_app" >&2 \
        || { rm -rf -- "$tmp"; die "dotnet publish failed for HartsyInference.VoiceHost"; }
    sudo -u "$SUDO_USER" dotnet publish -c Release "$repo_root/src/HartsyInference.PhoneGateway" -o "$tmp/$phone_gateway_app" >&2 \
        || { rm -rf -- "$tmp"; die "dotnet publish failed for HartsyInference.PhoneGateway"; }
    trap - EXIT
    printf '%s' "$tmp"
}

verify_publish_dir() {
    local dir=$1
    [[ -f "$dir/$voice_host_app/HartsyInference.VoiceHost.dll" ]] || die "--publish-dir $dir: missing $voice_host_app/HartsyInference.VoiceHost.dll"
    [[ -f "$dir/$phone_gateway_app/HartsyInference.PhoneGateway.dll" ]] || die "--publish-dir $dir: missing $phone_gateway_app/HartsyInference.PhoneGateway.dll"
    say "--publish-dir: reusing $dir (dotnet publish skipped)"
}

# Copies every file under $1 (a published app directory) to $2 (root:root, 0755), only where the bytes differ; prints
# the source files' SHA-256 either way, so a dry run and the apply that follows it can be compared. Also removes
# any file under $2 with no counterpart under $1 (and the empty directories that leaves), so $2 stays a mirror of
# $1 across publishes -- this assumes $2 (/opt/hartsyinference/<app>) holds only what this script put there, never
# anything an operator added by hand. Sets wrote=1 if anything changed.
put_tree() {
    local src=$1 dst=$2 file rel changed=0 count=0 stale=0
    [[ -d $src ]] || die "$src does not exist"
    wrote=0
    say "source SHA-256 ($src):"
    while IFS= read -r -d '' file; do
        rel=${file#"$src"/}
        printf '    %s  %s\n' "$(sha256sum -- "$file" | cut -d' ' -f1)" "$rel"
        count=$((count + 1))
    done < <(find "$src" -type f -print0 | sort -z)
    say "$count file(s) under $src"
    local stale_files=()
    if [[ -d $dst ]]; then
        while IFS= read -r -d '' file; do
            rel=${file#"$src"/}
            refuse_symlink "$dst/$rel"
            if [[ ! -f "$dst/$rel" ]] || ! cmp -s -- "$file" "$dst/$rel"; then
                changed=$((changed + 1))
            fi
        done < <(find "$src" -type f -print0 | sort -z)
        # A file (or symlink -- cp -a preserves one as a link rather than following it, so -type f alone would
        # never see it here) under $dst with no counterpart under $src: cp -a only ever adds/overwrites, so
        # without this a DLL dropped between publishes (or its .deps.json) would stay in /opt forever.
        while IFS= read -r -d '' file; do
            rel=${file#"$dst"/}
            [[ -e "$src/$rel" || -L "$src/$rel" ]] || stale_files+=("$rel")
        done < <(find "$dst" \( -type f -o -type l \) -print0 | sort -z)
        stale=${#stale_files[@]}
    else
        changed=$count
    fi
    if ((changed == 0 && stale == 0)); then
        say "$dst: already installed ($count file(s), unchanged)"
        return 0
    fi
    wrote=1
    if ((dry_run)); then
        say "would install $changed of $count file(s) into $dst (root:root, dirs and files 0755)"
        if ((stale > 0)); then
            say "would also remove $stale stale file(s) no longer under $src:"
            printf '    %s\n' "${stale_files[@]}"
        fi
    else
        install -d -o root -g root -m 0755 -- "$dst"
        # Mirrors the source tree (subdirectories too, e.g. Kokoro's voices/ is not part of this tree, but a plugin
        # or locale folder would be): cp -a preserves the publish output's own file modes, then we force root:root
        # 0755 uniformly, matching the task's ownership and the apphost binary's need for +x.
        cp -a -- "$src"/. "$dst"/
        chown -R root:root -- "$dst"
        find "$dst" -type d -exec chmod 0755 {} +
        find "$dst" -type f -exec chmod 0755 {} +
        if ((stale > 0)); then
            for rel in "${stale_files[@]}"; do
                rm -f -- "$dst/$rel"
                # Only each removed file's own now-maybe-empty parent chain, stopping at the first directory
                # that still has something in it (never a wholesale "any empty dir under $dst": a plugin or
                # locale folder the app ships empty, and that still exists under $src, must not be swept too).
                rmdir -p --ignore-fail-on-non-empty -- "$(dirname -- "$dst/$rel")" 2>/dev/null || true
            done
            say "removed $stale stale file(s) no longer under $src"
        fi
        say "installed $changed of $count file(s) into $dst"
    fi
}

# ---- secrets --------------------------------------------------------------------------------------------------

put_secret() {
    local name=$1
    local dst="$secrets_dir/$name" tmp
    refuse_symlink "$dst"
    if [[ -s $dst ]]; then
        say "$dst: already present"
        return 0
    fi
    if ((dry_run)); then
        if ((EUID == 0)); then
            say "would generate $dst (0600; value not printed)"
        else
            say "would generate $dst if missing (0600; cannot tell without root whether it already exists)"
        fi
        return 0
    fi
    # Via a temp file in the same directory, so a failure partway (disk full, killed) never leaves an empty $dst
    # that a later run would wrongly treat as "already present" (hence -s above, not -f: an empty file left by an
    # old run of this bug still needs to be regenerated).
    tmp=$(mktemp "$secrets_dir/.$name.XXXXXX")
    trap 'rm -f -- "$tmp"' EXIT
    (
        umask 077
        openssl rand -hex 32 >"$tmp"
    )
    chmod 0600 -- "$tmp"
    chown root:root -- "$tmp"
    mv -f -- "$tmp" "$dst"
    trap - EXIT
    say "generated $dst (0600; value not printed)"
}

# ---- configs ----------------------------------------------------------------------------------------------------

# Renders $1 (a template) to a temp file via $3 (a python filter: strips `//` comment lines, applies the production
# overrides) and installs it at $2 (0644) only if $2 does not exist; otherwise prints a diff and leaves it alone.
# Both example templates' defaults already match this box's production paths (/run/hartsyinference,
# /run/credentials/<unit>/<name>); only phone.json needs field overrides (see render_phone_json).
put_config() {
    local dst=$1 tmp=$2
    refuse_symlink "$dst"
    if [[ -f $dst ]]; then
        if cmp -s -- "$tmp" "$dst"; then
            say "$dst: already matches the LAN profile"
        else
            say "$dst already exists; leaving it alone. Diff (existing vs. the LAN profile this run would write):"
            diff -u -- "$dst" "$tmp" | sed 's/^/    /' || true
        fi
        rm -f -- "$tmp"
        return 0
    fi
    if ((dry_run)); then
        say "would write $dst (root:root, 0644):"
        sed 's/^/    /' <"$tmp"
        rm -f -- "$tmp"
    else
        install -o root -g root -m 0644 -- "$tmp" "$dst"
        rm -f -- "$tmp"
        say "wrote $dst"
    fi
}

render_voice_json() {
    local template="$repo_root/src/HartsyInference.VoiceHost/voice.example.json" tmp
    [[ -f $template ]] || die "missing $template"
    refuse_symlink "$etc_dir/voice.json"
    tmp=$(mktemp)
    trap 'rm -f -- "$tmp"' EXIT
    python3 - "$template" "$tmp" <<'PY'
import json, sys
template, out = sys.argv[1:3]
with open(template, encoding="utf-8") as f:
    text = "\n".join(line for line in f if line.strip()[:2] != "//")
cfg = json.loads(text)
# Otherwise the template's defaults (/run/hartsyinference/phone.sock,
# /run/credentials/hartsyinference-voice-host.service/phone-link-token, engine.cpuThreadCap 14, llmModel "qwen3",
# now resolved) already match this box's production units; only the denoiser is forced, matching
# VoiceAgentOptions' own default, in case the template ever carries a conflicting hard-coded value again.
cfg["models"]["denoise"] = True
with open(out, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
    f.write("\n")
PY
    trap - EXIT
    put_config "$etc_dir/voice.json" "$tmp"
}

render_phone_json() {
    local template="$repo_root/src/HartsyInference.PhoneGateway/phone.example.json" tmp
    [[ -f $template ]] || die "missing $template"
    refuse_symlink "$etc_dir/phone.json"
    tmp=$(mktemp)
    trap 'rm -f -- "$tmp"' EXIT
    python3 - "$template" "$tmp" <<'PY'
import json, sys
template, out = sys.argv[1:3]
with open(template, encoding="utf-8") as f:
    text = "\n".join(line for line in f if line.strip()[:2] != "//")
cfg = json.loads(text)
# The LAN profile: no registrar, no trunk, so no destination restriction (the `*File` paths already match this
# box's units' /run/credentials/<unit>/<name> convention and need no change). media.tickCpu pins the RTP tick
# thread to the hyperthread pair the gateway unit's AllowedCPUs=7,15 reserves for it.
cfg["sip"]["registrar"] = ""
cfg["sip"]["destinationPrefixes"] = []
cfg["sip"]["allowAnyDestination"] = True
cfg["media"]["tickCpu"] = 7
with open(out, "w", encoding="utf-8") as f:
    json.dump(cfg, f, indent=2)
    f.write("\n")
PY
    trap - EXIT
    put_config "$etc_dir/phone.json" "$tmp"
}

# ---- units ----------------------------------------------------------------------------------------------------

# Enables and starts (or restarts, if active but not enabled/changed) unit $1; mirrors install-host-tuning.sh's
# own unit logic, generalized to any number of units called in order.
enable_now_unit() {
    local unit=$1 needs_restart=${2:-0}
    if systemctl is-enabled --quiet "$unit" 2>/dev/null; then
        say "$unit: already enabled"
    else
        run systemctl enable "$unit"
    fi
    if ! systemctl is-active --quiet "$unit" 2>/dev/null; then
        if ! run systemctl start "$unit"; then
            die "$unit did not start; see: systemctl status $unit"
        fi
    elif ((needs_restart)); then
        say "$unit: already active, but its binaries or unit file changed; restarting so the change takes effect"
        if ! run systemctl restart "$unit"; then
            die "$unit did not restart; see: systemctl status $unit"
        fi
    else
        say "$unit: already active"
    fi
}

disable_now_unit() {
    local unit=$1
    wrote=0
    if [[ -e "$unit_dir/$unit" ]] || systemctl is-enabled --quiet "$unit" 2>/dev/null; then
        run systemctl disable --now "$unit"
    else
        say "$unit: not installed"
    fi
    # A dangling link here means the unit file was removed by hand, not by this script's own drop_file -- which
    # in revert() only runs *after* every disable_now_unit call, so "not already gone" is the common case the
    # first time through. Correct either way; just depends on that call order.
    if [[ ! -e "$unit_dir/$unit" && -L "$wants_dir/$unit" ]]; then
        run rm -f -- "$wants_dir/$unit"
        wrote=1
    fi
}

print_confirm_commands() {
    cat <<EOF
Confirm the install (runbook, Checklists/VOICE_AGENT_VERIFICATION.md, "Confirm the install"):
    systemctl status $host_unit $gateway_unit
    journalctl -u $host_unit -b | grep '\\[VoiceHost\\]'
    ls -l /run/hartsyinference/phone.sock
    journalctl -u $gateway_unit -b | grep -E 'gen0 budget|SCHED_FIFO'
    curl -s 127.0.0.1:9280/health
EOF
}

# ---- apply / revert -------------------------------------------------------------------------------------------

apply() {
    local voice_host_src="$repo_root/deploy/systemd/$host_unit" gateway_src="$repo_root/deploy/systemd/$gateway_unit"
    local server_src="$repo_root/deploy/systemd/$server_unit"
    [[ -f $voice_host_src && -f $gateway_src && -f $server_src ]] || die "deploy/systemd unit sources are missing; run from a checkout of the repository"

    local src_dir host_tree_changed=0 gateway_tree_changed=0 own_src_dir=0
    if [[ -n $publish_dir ]]; then
        verify_publish_dir "$publish_dir"
        src_dir=$publish_dir
    elif ((dry_run)); then
        say "would publish as \$SUDO_USER into a temp dir, then install it (see --publish-dir to skip this preview step)"
        src_dir=""
    else
        src_dir=$(publish_as_user)
        own_src_dir=1
        # Armed here, not after refuse_symlink/install -d on $opt_dir below: publish_as_user already succeeded
        # by this point, so the full tree exists in $src_dir -- a die() in either of those two lines (a
        # symlinked $opt_dir, say) would leak it otherwise, the same leak as publish_as_user's own, just one
        # call earlier. Covers put_tree() dying partway too (refuse_symlink, for one), which
        # publish_as_user()'s own EXIT trap (already cleared by the time we get here) does not -- that one only
        # covers a failed publish itself, not what happens to its output afterward.
        trap 'rm -rf -- "$src_dir"' EXIT
    fi

    refuse_symlink "$opt_dir"
    run install -d -o root -g root -m 0755 -- "$opt_dir"
    if [[ -n $src_dir ]]; then
        put_tree "$src_dir/$voice_host_app" "$opt_dir/$voice_host_app"
        host_tree_changed=$wrote
        put_tree "$src_dir/$phone_gateway_app" "$opt_dir/$phone_gateway_app"
        gateway_tree_changed=$wrote
        if ((own_src_dir)); then
            trap - EXIT
            rm -rf -- "$src_dir"
        fi
    fi

    refuse_symlink "$secrets_dir"
    run install -d -o root -g root -m 0700 -- "$secrets_dir"
    put_secret phone-link-token
    put_secret phone-admin-token
    put_secret sip-password

    refuse_symlink "$etc_dir"
    run install -d -o root -g root -m 0755 -- "$etc_dir"
    render_voice_json
    render_phone_json

    put_file "$voice_host_src" "$unit_dir/$host_unit" root:root 0644
    local host_changed=$wrote
    put_file "$gateway_src" "$unit_dir/$gateway_unit" root:root 0644
    local gateway_changed=$wrote
    local server_changed=0
    if [[ -e "$unit_dir/$server_unit" ]]; then
        put_file "$server_src" "$unit_dir/$server_unit" root:root 0644
        server_changed=$wrote
        if ((server_changed && !dry_run)); then
            say "note: $server_unit changed; restart it for the new settings to take effect: systemctl restart $server_unit"
        fi
    else
        say "$server_unit: not installed; leaving it alone (install it yourself first if you want it kept in sync here)"
    fi

    if ((host_changed || gateway_changed || server_changed)); then
        run systemctl daemon-reload
    fi
    enable_now_unit "$host_unit" "$((host_tree_changed || host_changed))"
    enable_now_unit "$gateway_unit" "$((gateway_tree_changed || gateway_changed))"

    if ((!dry_run)); then
        say "status:"
        systemctl --no-pager --no-legend status "$host_unit" "$gateway_unit" || true
        print_confirm_commands
    fi
}

revert() {
    disable_now_unit "$gateway_unit"
    local gateway_disabled=$wrote
    disable_now_unit "$host_unit"
    local host_disabled=$wrote
    drop_file "$unit_dir/$gateway_unit"
    local gateway_dropped=$wrote
    drop_file "$unit_dir/$host_unit"
    local host_dropped=$wrote
    if ((gateway_disabled || host_disabled || gateway_dropped || host_dropped)); then
        run systemctl daemon-reload
    fi
    if ((purge)); then
        say "--purge: removing binaries, configs and secrets"
        run rm -rf -- "$opt_dir/$voice_host_app" "$opt_dir/$phone_gateway_app" "$etc_dir/voice.json" "$etc_dir/phone.json" "$secrets_dir"
        # Only if that leaves them empty: a config this script doesn't know about (nothing today, but the
        # pattern this whole script follows is to never remove something it didn't itself put there) would
        # keep $etc_dir non-empty, and rmdir would just report that and leave it, harmlessly.
        if ((!dry_run)); then
            rmdir --ignore-fail-on-non-empty -- "$opt_dir" "$etc_dir" 2>/dev/null || true
        fi
    else
        say "binaries ($opt_dir), configs and secrets ($etc_dir) are left in place; --purge to remove them too"
    fi
}

main() {
    local arg action="" explicit_dry=0 self
    self=$(printf '%q' "$0")
    while [[ $# -gt 0 ]]; do
        arg=$1
        case "$arg" in
            --dry-run) explicit_dry=1; shift ;;
            --apply | --revert)
                if [[ -n $action && $action != "${arg#--}" ]]; then
                    die "choose one of --apply and --revert"
                fi
                action=${arg#--}
                shift
                ;;
            --purge) purge=1; shift ;;
            --publish-dir)
                [[ -n ${2:-} ]] || die "--publish-dir needs a directory"
                publish_dir=$2
                shift 2
                ;;
            -h | --help) usage; exit 0 ;;
            *) usage >&2; die "unknown option: $arg" ;;
        esac
    done
    if [[ -z $action ]]; then
        action=apply
        explicit_dry=1
    fi
    if ((purge)) && [[ $action != "revert" ]]; then
        die "--purge needs --revert"
    fi
    dry_run=$explicit_dry
    command -v systemctl >/dev/null || die "systemctl not found; this installs systemd units"
    command -v openssl >/dev/null || die "openssl not found; needed to generate secrets"
    command -v python3 >/dev/null || die "python3 not found; needed to render voice.json/phone.json"
    if ((!dry_run && EUID != 0)); then
        die "--$action changes system settings and must run as root: sudo bash $self --$action"
    fi
    if ((dry_run)); then
        say "dry run of --$action: nothing will be changed"
    fi
    case $action in
        apply) apply ;;
        revert) revert ;;
    esac
    if ((failures)); then
        die "$failures failure(s); see above"
    fi
    if ((dry_run)); then
        say "dry run only; to do this: sudo bash $self --$action"
    else
        say "done"
    fi
}

main "$@"
