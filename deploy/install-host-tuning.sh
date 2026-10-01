#!/usr/bin/env bash
# Host tuning for the phone-call voice agent (deploy/README.md, "Host tuning"). It installs two files:
#   /etc/systemd/system/hartsyinference-cpu-performance.service  a oneshot unit; at boot it sets every CPU's frequency
#                                                                 governor to "performance"
#   /etc/security/limits.d/hartsy-rt.conf                         rtprio 50 for the service user outside systemd
# Both are optimizations: every voice gate passes under the stock schedutil governor.
#
# Only the unit writes "performance"; only --revert writes "schedutil" back. With no option this is a dry run that
# prints exactly what --apply would write and change; it needs no root. --apply and --revert need root, and running
# either again changes nothing.

set -euo pipefail
shopt -s nullglob

deploy_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
readonly deploy_dir
readonly unit=hartsyinference-cpu-performance.service
readonly unit_src=$deploy_dir/systemd/$unit
readonly unit_dst=/etc/systemd/system/$unit
readonly limits_src=$deploy_dir/limits.d/hartsy-rt.conf
readonly limits_dst=/etc/security/limits.d/hartsy-rt.conf
readonly tuned=performance
readonly stock=schedutil
readonly governor_files=(/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_governor)

dry_run=1
wrote=0
failures=0

say() { printf '[host-tuning] %s\n' "$*"; }

die() {
    printf '[host-tuning] error: %s\n' "$*" >&2
    exit 1
}

usage() {
    cat <<'EOF'
Usage: deploy/install-host-tuning.sh [--dry-run] [--apply | --revert]
  (no option), --dry-run  print what --apply would write and change; changes nothing, needs no root
  --apply                 install the governor unit and the rtprio limits file, enable and start the unit
  --revert                disable and remove the unit, remove the limits file, set every CPU back to schedutil
  --dry-run --revert      print what --revert would do
--apply and --revert need root; running either again changes nothing.
EOF
}

# Runs a command and says so, or only prints it in a dry run.
run() {
    if ((dry_run)); then
        say "would run: $*"
    else
        say "running: $*"
        "$@"
    fi
}

# Stops before any change when a destination is a symlink: a root script must not write through one.
refuse_symlink() {
    if [[ -L $1 ]]; then
        die "$1 is a symlink; refusing to write through it (remove it, then run again)"
    fi
}

# Installs $1 as $2 (root:root, 0644) unless $2 already holds the same bytes; sets wrote=1 when it writes.
put_file() {
    local src=$1 dst=$2
    wrote=0
    refuse_symlink "$dst"
    if [[ -f $dst ]] && cmp -s -- "$src" "$dst"; then
        say "$dst: already installed"
        return 0
    fi
    wrote=1
    if ((dry_run)); then
        say "would write $dst (root:root, 0644):"
        sed 's/^/    /' -- "$src"
    else
        install -D -o root -g root -m 0644 -- "$src" "$dst"
        say "wrote $dst"
    fi
}

# Removes $1 if present; sets wrote=1 when it removes.
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

# cpu7 from /sys/devices/system/cpu/cpu7/cpufreq/scaling_governor.
cpu_of() {
    local path=${1#/sys/devices/system/cpu/}
    printf '%s' "${path%%/*}"
}

# True when the CPU whose governor file is $1 lists governor $2 as available.
offers() {
    local list=${1%/*}/scaling_available_governors available
    if [[ ! -r $list ]]; then
        return 1
    fi
    available=$(<"$list")
    [[ " $available " == *" $2 "* ]]
}

# How many CPUs offer governor $1 but run another one.
count_off() {
    local file n=0
    for file in "${governor_files[@]}"; do
        if [[ $(<"$file") != "$1" ]] && offers "$file" "$1"; then
            n=$((n + 1))
        fi
    done
    printf '%d\n' "$n"
}

# Lists the CPUs not on governor $1; after a real --apply each one that offers it counts as a failure.
check_governors() {
    local want=$1 file current on=0
    for file in "${governor_files[@]}"; do
        current=$(<"$file")
        if [[ $current == "$want" ]]; then
            on=$((on + 1))
        elif ! offers "$file" "$want"; then
            say "$(cpu_of "$file"): $want is not offered; stays $current"
        elif ((dry_run)); then
            say "$(cpu_of "$file"): $current; the unit would set $want"
        else
            say "$(cpu_of "$file"): still $current"
            failures=$((failures + 1))
        fi
    done
    say "$on of ${#governor_files[@]} CPUs on $want"
}

# Writes governor $1 to every CPU that offers it and runs another one.
set_governors() {
    local want=$1 file current kept=0
    for file in "${governor_files[@]}"; do
        current=$(<"$file")
        if [[ $current == "$want" ]]; then
            kept=$((kept + 1))
        elif ! offers "$file" "$want"; then
            say "$(cpu_of "$file"): $want is not offered; stays $current"
        elif ((dry_run)); then
            say "would set $(cpu_of "$file"): $current -> $want"
        elif printf '%s\n' "$want" >"$file"; then
            say "$(cpu_of "$file"): $current -> $want"
        else
            say "$(cpu_of "$file"): writing $want failed; stays $current"
            failures=$((failures + 1))
        fi
    done
    say "$kept of ${#governor_files[@]} CPUs were already on $want"
}

apply() {
    if [[ ! -f $unit_src || ! -f $limits_src ]]; then
        die "$unit_src or $limits_src is missing; run this script from a checkout of the repository"
    fi
    refuse_symlink "$unit_dst"
    refuse_symlink "$limits_dst"
    put_file "$unit_src" "$unit_dst"
    if ((wrote)); then
        run systemctl daemon-reload
    fi
    if systemctl is-enabled --quiet "$unit" 2>/dev/null; then
        say "$unit: already enabled"
    else
        run systemctl enable "$unit"
    fi
    if ! systemctl is-active --quiet "$unit" 2>/dev/null; then
        run systemctl start "$unit"
    elif (($(count_off "$tuned") > 0)); then
        # Active but a governor changed since boot: run the unit again, it only writes "performance".
        run systemctl restart "$unit"
    else
        say "$unit: already active"
    fi
    put_file "$limits_src" "$limits_dst"
    check_governors "$tuned"
}

revert() {
    if [[ -e $unit_dst ]] || systemctl is-enabled --quiet "$unit" 2>/dev/null; then
        run systemctl disable --now "$unit"
    else
        say "$unit: not installed"
    fi
    drop_file "$unit_dst"
    if ((wrote)); then
        run systemctl daemon-reload
    fi
    drop_file "$limits_dst"
    set_governors "$stock"
}

main() {
    local arg action="" explicit_dry=0
    for arg in "$@"; do
        case $arg in
            --dry-run) explicit_dry=1 ;;
            --apply | --revert)
                if [[ -n $action && $action != "${arg#--}" ]]; then
                    die "choose one of --apply and --revert"
                fi
                action=${arg#--}
                ;;
            -h | --help)
                usage
                exit 0
                ;;
            *)
                usage >&2
                die "unknown option: $arg"
                ;;
        esac
    done
    if [[ -z $action ]]; then
        action=apply
        explicit_dry=1
    fi
    dry_run=$explicit_dry
    command -v systemctl >/dev/null || die "systemctl not found; this installs a systemd unit"
    if ((!dry_run && EUID != 0)); then
        die "--$action changes system settings and must run as root: sudo $0 --$action"
    fi
    if ((${#governor_files[@]} == 0)); then
        say "no cpufreq governors under /sys/devices/system/cpu; the unit would change nothing here"
    fi
    if ((dry_run)); then
        say "dry run of --$action: nothing will be changed"
    fi
    case $action in
        apply) apply ;;
        revert) revert ;;
    esac
    if ((failures)); then
        die "$failures CPU(s) did not take the governor; see above"
    fi
    if ((dry_run)); then
        say "dry run only; to do this: sudo $0 --$action"
    else
        say "done"
    fi
}

main "$@"
