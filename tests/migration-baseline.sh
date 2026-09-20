#!/usr/bin/env bash
# Records what CUDA produces TODAY, so the migration onto the shared residency cache can be proved not to have
# changed it. Run once before the first CUDA change, then after each sub-step; the hashes must match exactly.
#
# Byte-identity, not a tolerance. The Vulkan half of this migration shipped three bugs that no single generation
# could see, and all three would have passed a tolerance gate.
#
#   tests/migration-baseline.sh --record                     # write the reference (once, on unmodified CUDA)
#   tests/migration-baseline.sh                              # compare against it
#   tests/migration-baseline.sh --backend vulkan --record    # the same gate for the Vulkan half
#   tests/migration-baseline.sh --filter sd15                # one case, for iterating on a change
#   tests/migration-baseline.sh --no-build                   # trust the CLI already on disk (+no-build in the stamp)
#
# It BUILDS the CLI before it measures anything, and that is not a convenience. The digest is evidence about a
# tree, and the only thing tying the two together is that the binary came from it — a gate run after building
# only the test projects measures whatever DLL was there before, reports `identical`, and the change under test
# never ran at all. That failure is silent and it looks exactly like success.
#
# One script for both backends on purpose: a digest is only evidence against a digest taken the same way, and the
# first Vulkan byte-identity checks were hashed by a different method than this, so they cannot be compared to
# anything produced here.
#
# An LLM case is here because image generation does not exercise the two mechanisms most likely to break:
# auto-promotion of a twice-uploaded weight, and the ambient state registry the decode loop resolves through.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLI="$REPO/src/HartsyInference.Cli/bin/Release/net10.0/HartsyInference.Cli.dll"
MODELS="${HARTSY_MODELS_ROOT:-/mnt/model-storage/Models}"
BASE="${HARTSY_MIGRATION_BASELINE:-$HOME/Desktop/migration-baselines}"
WORK="$(mktemp -d)"
MODE="compare"
BACKEND="cuda"
FILTER=""
BUILD=1
trap 'rm -rf "$WORK"' EXIT

while [ $# -gt 0 ]; do
    case "$1" in
        --record)  MODE="record";  shift ;;
        --backend) BACKEND="$2";   shift 2 ;;
        --filter)  FILTER="$2";    shift 2 ;;
        --no-build) BUILD=0;       shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
REF="$BASE/$BACKEND"

# What the run measured, so a digest can be traced back to a tree. A reference carries its own copy (written
# beside it at --record); a stale one is then visible as a commit nobody recognises instead of an unexplained
# CHANGED months later.
PROVENANCE="$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)"
if [ -n "$(git -C "$REPO" status --porcelain 2>/dev/null)" ]; then
    PROVENANCE="$PROVENANCE+dirty"
fi
if [ "$BUILD" = 0 ]; then
    # The commit is only evidence about the binary if this run built it. Under --no-build the CLI on disk may
    # have come from anywhere, so the stamp says it is a guess rather than quietly asserting a tree — which is
    # the misattribution this script exists to make impossible, one level up.
    PROVENANCE="$PROVENANCE+no-build"
fi

if [ "$BUILD" = 1 ]; then
    echo "building the CLI from $PROVENANCE..." >&2
    if ! dotnet build "$REPO/src/HartsyInference.Cli" -c Release -f net10.0 --nologo -v q >"$WORK/build.log" 2>&1; then
        echo "the CLI did not build; the gate has nothing to measure:" >&2
        tail -30 "$WORK/build.log" >&2
        exit 2
    fi
fi
[ -f "$CLI" ] || { echo "build the CLI first: dotnet build src/HartsyInference.Cli -c Release -f net10.0" >&2; exit 2; }
mkdir -p "$REF"

# case id | checkpoint (relative to MODELS) | command|positional|arguments
CASES=$(cat <<'MATRIX'
sd15	bench-cache/e9476a13728cd75d8279f6ec8bad753a66a1957ca375a1464dc63b37db6e3916/v1-5-pruned-emaonly-fp16.safetensors	image|a red apple on a table|--steps 8 --width 512 --height 512 --seed 42
krea2	Stable-Diffusion/Krea2/Turbo/krea2_turbo_fp8_scaled.safetensors	image|a red apple on a wooden table|-m krea2 --steps 8 --width 1024 --height 1024 --seed 42
llama32-1b	llm/llama32-1b/llama-3.2-1b-instruct-q8_0.gguf	text|Write a short story about a robot learning to paint.|--max-tokens 256 --temperature 0 --seed 42
MATRIX
)

# A digest is the only thing this script compares, so an unusable one must never reach the comparison. md5sum
# failing leaves the substitution empty, and an empty value is stable — record and compare would both produce it and
# the run would report `identical` having hashed nothing. Demand the exact shape of an MD5.
digest_of() {
    local value
    value="$(md5sum "$1" | cut -d' ' -f1)" || return 1
    case "$value" in
        [0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]) printf '%s' "$value" ;;
        *) return 1 ;;
    esac
}

status=0
confirmed=0
selected=0
printf 'case\tbackend\tresult\tdigest\tsource\n'

while IFS=$'\t' read -r id ckpt spec; do
    [ -z "$id" ] && continue
    # A filtered run reports only what it ran. Silence about a case is not a claim about it.
    if [ -n "$FILTER" ] && [ "$id" != "$FILTER" ]; then
        continue
    fi
    selected=$((selected + 1))
    IFS='|' read -r cmd positional args <<< "$spec"
    path="$MODELS/$ckpt"
    if [ ! -e "$path" ]; then
        # Recording what this machine has is reasonable; COMPARING and reporting success because the checkpoint was
        # missing is not. A case that should have been checked and was not is a failed gate, not a quiet note —
        # otherwise a machine holding only the image checkpoints proves the LLM path unchanged by never running it.
        printf '%s\t%s\tuntested\t%s\n' "$id" "$BACKEND" "$path"
        [ "$MODE" = compare ] && status=1
        continue
    fi

    out="$WORK/$id"
    mkdir -p "$out"
    # shellcheck disable=SC2086
    if ! timeout 1800 dotnet "$CLI" "$cmd" "$positional" --model-path "$path" -b "$BACKEND" -o "$out" $args \
        >"$WORK/$id.log" 2>&1 </dev/null; then
        printf '%s\t%s\tCRASH\tsee %s\n' "$id" "$BACKEND" "$WORK/$id.log"
        status=1
        continue
    fi

    # An image is hashed from its pixels, not its file: PNG metadata carries a timestamp, so two identical
    # generations differ in bytes. Text is hashed as-is.
    artifact="$(find "$out" -type f \( -name '*.png' -o -name '*.txt' \) | sort | head -1)"
    if [ -z "$artifact" ]; then
        printf '%s\t%s\tCRASH\tno artifact produced\n' "$id" "$BACKEND"
        status=1
        continue
    fi
    case "$artifact" in
        *.png)
            # Decode to a FILE and check it, rather than piping into md5sum. A pipeline reports the last command's
            # status, so a missing or failing ffmpeg left md5sum hashing empty stdin — and the empty digest is
            # stable, so recording and comparing would both produce it and the gate would report `identical`
            # without having looked at a single pixel. A gate that cannot fail is worse than no gate.
            raw="$WORK/$id.rgb24"
            if ! ffmpeg -nostdin -loglevel error -i "$artifact" -f rawvideo -pix_fmt rgb24 -y "$raw"                 >>"$WORK/$id.log" 2>&1 || [ ! -s "$raw" ]; then
                printf '%s\t%s\tCRASH\tcould not decode %s (see %s)\n' "$id" "$BACKEND" "$artifact" "$WORK/$id.log"
                status=1
                continue
            fi
            if ! digest="$(digest_of "$raw")"; then
                printf '%s\t%s\tCRASH\tcould not hash %s\n' "$id" "$BACKEND" "$raw"
                status=1
                continue
            fi
            ;;
        *)
            if [ ! -s "$artifact" ]; then
                printf '%s\t%s\tCRASH\tempty artifact %s\n' "$id" "$BACKEND" "$artifact"
                status=1
                continue
            fi
            if ! digest="$(digest_of "$artifact")"; then
                printf '%s\t%s\tCRASH\tcould not hash %s\n' "$id" "$BACKEND" "$artifact"
                status=1
                continue
            fi
            ;;
    esac

    if [ "$MODE" = record ]; then
        echo "$digest" > "$REF/$id.digest"
        printf '%s\t%s\n' "$PROVENANCE" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$REF/$id.source"
        cp "$artifact" "$REF/$id.${artifact##*.}"
        printf '%s\t%s\trecorded\t%s\t%s\n' "$id" "$BACKEND" "$digest" "$PROVENANCE"
        continue
    fi
    if [ ! -f "$REF/$id.digest" ]; then
        printf '%s\t%s\tno-reference\t%s\n' "$id" "$BACKEND" "$digest"
        status=1
        continue
    fi
    if [ "$digest" = "$(cat "$REF/$id.digest")" ]; then
        printf '%s\t%s\tidentical\t%s\n' "$id" "$BACKEND" "$digest"
        confirmed=$((confirmed + 1))
    else
        # Name both trees. "CHANGED" on its own sends the reader looking for a bug in the branch, when the
        # reference may simply predate a change nobody attributed — which is exactly what happened to krea2
        # on Vulkan between 2026-09-18 and 2026-09-20.
        recorded_at="unknown commit"
        [ -f "$REF/$id.source" ] && recorded_at="$(cut -f1 "$REF/$id.source") of $(cut -f2 "$REF/$id.source")"
        printf '%s\t%s\tCHANGED\t%s at %s (reference %s, recorded at %s)\n' \
            "$id" "$BACKEND" "$digest" "$PROVENANCE" "$(cat "$REF/$id.digest")" "$recorded_at"
        cp "$artifact" "$REF/$id.actual.${artifact##*.}"
        status=1
    fi
done <<< "$CASES"

if [ "$selected" -eq 0 ]; then
    printf 'no case matched --filter %s\n' "$FILTER" >&2
    exit 2
fi
if [ "$MODE" = compare ]; then
    # Say what was actually proved. A gate that reports success without naming a count invites reading an empty run
    # as a passing one.
    printf '%d of %d selected case(s) confirmed identical on %s\n' "$confirmed" "$selected" "$BACKEND" >&2
fi

exit $status
