#!/usr/bin/env bash
# Runs one test project and fails unless the tests it was asked for really executed: a run that self-skipped, hit a
# backend-unavailable guard or matched nothing must never come back green. tests/gpu-suite.sh and
# tests/run-multigpu-campaign.sh both go through it.
#
#   tests/require-executed.sh --project <dir|.csproj> [--filter EXPR] [--manifest FILE] [--out DIR] [--name NAME]
#                             [ENV=VAL ...] [-- <extra dotnet test args>]
#
# --manifest is a file of fully qualified test names, one per line (# comments and blank lines ignored; an empty one is
# an error). Every name must execute; a theory's cases count as its method. With no --filter the run is filtered to
# exactly those tests. HARTSY_REQUIRE_REAL_WEIGHTS=1 is set after the ENV=VAL pairs, so a missing checkpoint fails.
#
# Exit 1 on: a nonzero dotnet test exit; a SKIPPED line; a GUARD_LINES line; zero passed in the TRX; a manifest test
# the TRX shows not executed. Exit 2 on bad arguments. Writes NAME.log (detailed console), NAME.trx and
# NAME.executed.md (one "- `test` — Outcome" line per result, for a PR body) to --out, by default
# Output/require-executed/<UTC stamp>. A guard that returns without logging anything looks like a pass here too.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Guards that report a missing backend or device without a SKIPPED prefix, matched as fixed strings. Each is the
# literal the test writes; the interpolated nvidia-smi messages share their first two words.
GUARD_LINES=(
    "SKIP no PTX"                                                 # ThreeD CudaOpBisectTests
    "Skipped: no usable CUDA device."                             # Audio SheetSage2PipelineParityTests
    "skip (no CUDA)"                                              # Audio AttnBenchTest
    "factory duplicate-ordinal check skipped: CUDA unavailable"   # Cuda CollectiveCommTests
    "GPU dequant skipped (no PTX dir)"                            # Diffusion QwenGgufDequantTests
    "nvidia-smi unavailable"                                      # the sharding engine tests' VRAM-pooling checks
)

die() { echo "require-executed: $*" >&2; exit 2; }

PROJECT="" FILTER="" MANIFEST="" OUT="" NAME="" ENV_PAIRS=() DOTNET_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --project) PROJECT="$2"; shift 2 ;;  --filter) FILTER="$2"; shift 2 ;;  --manifest) MANIFEST="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;  --name) NAME="$2"; shift 2 ;;  --) shift; DOTNET_ARGS=("$@"); break ;;
        [A-Za-z_]*=*) ENV_PAIRS+=("$1"); shift ;;
        *) die "unknown argument $1" ;;
    esac
done
[ -n "${PROJECT}" ] || die "--project is required"
[ -n "${FILTER}${MANIFEST}" ] || die "--filter or --manifest is required"

EXPECTED=()
if [ -n "${MANIFEST}" ]; then
    [ -f "${MANIFEST}" ] || die "no manifest ${MANIFEST}"
    while IFS= read -r line; do
        EXPECTED+=("${line}")
    done < <(sed -e 's/\r$//' -e 's/#.*//' -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e '/^$/d' "${MANIFEST}")
    [ ${#EXPECTED[@]} -gt 0 ] || die "manifest ${MANIFEST} lists no tests"
    if [ -z "${FILTER}" ]; then
        # vstest filter values escape \ ( ) & | = ! ~ with a backslash.
        for test in "${EXPECTED[@]}"; do
            FILTER+="${FILTER:+|}FullyQualifiedName=$(printf '%s' "${test}" | sed 's/[\\()&|=!~]/\\&/g')"
        done
    fi
fi

if [ -z "${NAME}" ]; then
    if [ -n "${MANIFEST}" ]; then NAME="$(basename "${MANIFEST}")"; NAME="${NAME%.*}"
    else NAME="$(basename "${PROJECT%/}" .csproj)"; fi
fi
OUT="${OUT:-${ROOT}/Output/require-executed/$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "${OUT}"
LOG="${OUT}/${NAME}.log" TRX="${OUT}/${NAME}.trx" LIST="${OUT}/${NAME}.executed.md"
rm -f "${TRX}" "${LIST}"   # a stale TRX from an earlier run must never stand in for this one

env "${ENV_PAIRS[@]}" HARTSY_REQUIRE_REAL_WEIGHTS=1 dotnet test "${PROJECT}" --filter "${FILTER}" --nologo \
    --logger "console;verbosity=detailed" --logger "trx;LogFileName=${NAME}.trx" --results-directory "${OUT}" \
    "${DOTNET_ARGS[@]}" > "${LOG}" 2>&1
rc=$?

FAILED=0
fail() { echo "FAILED — $*"; FAILED=1; }

[ "${rc}" -eq 0 ] || fail "dotnet test exit ${rc}; see ${LOG}"
if grep -q "SKIPPED" "${LOG}"; then
    fail "a test self-skipped despite HARTSY_REQUIRE_REAL_WEIGHTS=1:"
    grep "SKIPPED" "${LOG}" | head -5
fi
GUARD_ARGS=()
for guard in "${GUARD_LINES[@]}"; do GUARD_ARGS+=(-e "${guard}"); done
if grep -qF "${GUARD_ARGS[@]}" "${LOG}"; then
    fail "a backend-unavailable guard fired:"
    grep -F "${GUARD_ARGS[@]}" "${LOG}" | head -5
fi

# One "outcome<TAB>name" row per result. .NET's XmlWriter keeps each start tag whole and escapes its attribute values.
RESULTS=""
if [ -f "${TRX}" ]; then
    RESULTS="$(tr '\r\n' '  ' < "${TRX}" | grep -o '<UnitTestResult [^>]*>' | awk '
        function attr(tag, key,   value) {
            if (!match(tag, " " key "=\"[^\"]*\"")) return ""
            value = substr(tag, RSTART + length(key) + 3, RLENGTH - length(key) - 4)
            gsub(/&quot;/, "\"", value); gsub(/&apos;/, "\047", value); gsub(/&lt;/, "<", value)
            gsub(/&gt;/, ">", value); gsub(/&amp;/, "\\&", value)
            return value
        }
        { print attr($0, "outcome") "\t" attr($0, "testName") }' | LC_ALL=C sort -t $'\t' -k2)"
else
    fail "no TRX written at ${TRX}"
fi
count() { printf '%s\n' "${RESULTS}" | grep -c "^$1"$'\t'; }
passed=$(count Passed) failed=$(count Failed) not_executed=$(count NotExecuted)
total=$(printf '%s\n' "${RESULTS}" | grep -c .)
[ "${passed}" -gt 0 ] || fail "zero tests passed; see ${LOG}"

# A manifest test executed when a result carries its name, alone or followed by a theory's "(args)", and ran.
MISSING=()
for test in "${EXPECTED[@]}"; do
    hit=0
    while IFS=$'\t' read -r outcome name; do
        if [ "${outcome}" != NotExecuted ] && { [ "${name}" = "${test}" ] || [[ "${name}" == "${test}("* ]]; }; then
            hit=1
            break
        fi
    done <<< "${RESULTS}"
    [ "${hit}" -eq 1 ] || MISSING+=("${test}")
done
if [ ${#MISSING[@]} -gt 0 ]; then
    fail "${#MISSING[@]} manifest test(s) did not execute:"
    printf '  %s\n' "${MISSING[@]}"
fi

{
    echo "\`${NAME}\`: ${passed} passed, ${failed} failed, ${not_executed} not executed (${NAME}.trx)"
    echo
    printf '%s\n' "${RESULTS}" | awk -F '\t' 'NF { printf "- `%s` — %s\n", $2, $1 }'
} > "${LIST}"
printf '%s\n' "${RESULTS}" | awk -F '\t' '$1 == "Failed" { print "  Failed " $2 }'
echo "${passed} passed, ${failed} failed, ${not_executed} not executed of ${total}; list: ${LIST}"
exit "${FAILED}"
