#!/usr/bin/env bash
# Checks tests/require-executed.sh without a GPU or a build: a stub dotnet on PATH replays a canned console log, TRX and
# exit code, and records the arguments and environment it was handed.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HELPER="${ROOT}/tests/require-executed.sh"
TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT
export FAKE="${TMP}/fake"
mkdir -p "${TMP}/bin" "${FAKE}"
failures=0

cat > "${TMP}/bin/dotnet" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$@" > "${FAKE}/args"
printf '%s\n' "${HARTSY_REQUIRE_REAL_WEIGHTS-unset}" "${SELFTEST_PAIR-unset}" > "${FAKE}/env"
dir="" trx=""
while [ $# -gt 0 ]; do
    case "$1" in
        --results-directory) dir="$2"; shift 2 ;;
        --logger) case "$2" in "trx;LogFileName="*) trx="${2#trx;LogFileName=}" ;; esac; shift 2 ;;
        *) shift ;;
    esac
done
cat "${FAKE}/console.log"
[ -f "${FAKE}/results.trx" ] && cp "${FAKE}/results.trx" "${dir}/${trx}"
exit "$(cat "${FAKE}/rc")"
EOF
chmod +x "${TMP}/bin/dotnet"
export PATH="${TMP}/bin:${PATH}"

trx() { # "Outcome|test name" rows → a TRX shaped like vstest's: BOM, CRLF, one start tag per line, escaped attributes
    printf '\xEF\xBB\xBF<?xml version="1.0" encoding="utf-8"?>\r\n<TestRun id="r" xmlns="http://microsoft.com/schemas/VisualStudio/TeamTest/2010">\r\n  <Results>\r\n'
    for row in "$@"; do
        name="$(printf '%s' "${row#*|}" | sed 's/&/\&amp;/g; s/"/\&quot;/g; s/</\&lt;/g; s/>/\&gt;/g')"
        printf '    <UnitTestResult executionId="e" testId="t" testName="%s" computerName="box" outcome="%s" testListId="l">\r\n' \
            "${name}" "${row%%|*}"
        printf '      <Output>\r\n        <StdOut>output</StdOut>\r\n      </Output>\r\n    </UnitTestResult>\r\n'
    done
    printf '  </Results>\r\n</TestRun>\r\n'
}

canned() { # dotnet exit code, console log text, then the TRX rows
    printf '%s' "$1" > "${FAKE}/rc"
    printf '%s\n' "$2" > "${FAKE}/console.log"
    shift 2
    trx "$@" > "${FAKE}/results.trx"
}

expect() { # case name, expected helper exit, text its output must contain, helper args...
    local name="$1" want="$2" text="$3" out got
    shift 3
    out="$(bash "${HELPER}" --project Fake.Tests --out "${TMP}/out/${name}" --name "${name}" "$@" 2>&1)"
    got=$?
    if [ "${got}" -eq "${want}" ] && printf '%s\n' "${out}" | grep -qF -- "${text}"; then
        echo "PASS ${name}"
    else
        echo "FAIL ${name}: exit ${got} (want ${want}), wanted output containing '${text}', got:"
        printf '%s\n' "${out}" | sed 's/^/    /'
        failures=$((failures + 1))
    fi
}

assert() { # description, then a command that must succeed
    local what="$1"
    shift
    if "$@"; then echo "PASS ${what}"; else echo "FAIL ${what}"; failures=$((failures + 1)); fi
}

printf '# stack manifest\r\n  Ns.SuiteTests.Plain  \r\n\r\nNs.SuiteTests.Theory # every case\r\n' > "${TMP}/manifest.txt"
printf '# nothing listed\n\n' > "${TMP}/empty.txt"
THEORY_CUDA='Ns.SuiteTests.Theory(kind: "cuda")'

canned 0 "Test Run Successful." "Passed|Ns.SuiteTests.Plain" "Passed|${THEORY_CUDA}" 'Passed|Ns.SuiteTests.Theory(kind: "vulkan")'
expect passes 0 "3 passed, 0 failed, 0 not executed of 3" --manifest "${TMP}/manifest.txt" \
    SELFTEST_PAIR=on HARTSY_REQUIRE_REAL_WEIGHTS=0
assert "manifest-filter" grep -qxF 'FullyQualifiedName=Ns.SuiteTests.Plain|FullyQualifiedName=Ns.SuiteTests.Theory' "${FAKE}/args"
assert "env-pairs-and-real-weights" [ "$(cat "${FAKE}/env")" = "$(printf '1\non')" ]
assert "executed-list" grep -qxF -- "- \`${THEORY_CUDA}\` — Passed" "${TMP}/out/passes/passes.executed.md"

canned 1 "Test Run Failed." "Passed|Ns.SuiteTests.Plain" "Failed|Ns.SuiteTests.Broken"
expect nonzero-exit 1 "dotnet test exit 1" --filter "FullyQualifiedName~SuiteTests"

canned 0 "  Passed Ns.SuiteTests.Plain [5 ms]
  Standard Output Messages:
 SKIPPED: cuda backend unavailable — no CUDA devices" "Passed|Ns.SuiteTests.Plain"
expect self-skip 1 "a test self-skipped" --filter "FullyQualifiedName~SuiteTests"
canned 0 " CUDA leg SKIPPED: CUDA unavailable" "Passed|Ns.SuiteTests.Plain"
expect self-skip-mid-line 1 "a test self-skipped" --filter "FullyQualifiedName~SuiteTests"

# Every guard, as the line its test prints. Both nvidia-smi wordings hit the one shared prefix.
GUARD_SAMPLES=(
    " SKIP no PTX"
    " Skipped: no usable CUDA device."
    " skip (no CUDA)"
    " factory duplicate-ordinal check skipped: CUDA unavailable"
    " GPU dequant skipped (no PTX dir)"
    " nvidia-smi unavailable — VRAM pooling assertion skipped (parity check still ran)."
    " nvidia-smi unavailable or ambiguous — VRAM pooling assertion will be skipped."
)
for i in "${!GUARD_SAMPLES[@]}"; do
    canned 0 "${GUARD_SAMPLES[i]}" "Passed|Ns.SuiteTests.Plain"
    expect "guard-$((i + 1))" 1 "a backend-unavailable guard fired" --filter "FullyQualifiedName~SuiteTests"
done

# The helper matches guards as fixed strings, so a reworded test message would stop matching without a sound.
guards=()
while IFS= read -r guard; do
    guards+=("${guard}")
done < <(sed -n '/^GUARD_LINES=(/,/^)/s/^ *"\([^"]*\)".*/\1/p' "${HELPER}")
assert "guard-lines-parsed (${#guards[@]})" [ "${#guards[@]}" -eq 6 ]
for guard in ${guards[@]+"${guards[@]}"}; do
    assert "guard-still-printed: ${guard}" grep -rqF --include=*.cs -- "${guard}" "${ROOT}/tests"
done

canned 0 "No test matches the given testcase filter \`FullyQualifiedName~Nothing\`"
expect zero-passed 1 "zero tests passed" --filter "FullyQualifiedName~Nothing"
canned 0 "" "NotExecuted|Ns.SuiteTests.Plain"
expect only-not-executed 1 "zero tests passed" --filter "FullyQualifiedName~SuiteTests"

# A stale TRX left in --out by an earlier run must not stand in for a run that wrote none.
canned 0 "Test Run Successful."
rm -f "${FAKE}/results.trx"
mkdir -p "${TMP}/out/no-trx"
trx "Passed|Ns.SuiteTests.Plain" > "${TMP}/out/no-trx/no-trx.trx"
expect no-trx 1 "no TRX written" --filter "FullyQualifiedName~SuiteTests"

# Neither manifest test ran: one only by a longer name, the other's only case was not executed.
canned 0 "" "Passed|Ns.SuiteTests.PlainTwo" "NotExecuted|${THEORY_CUDA}"
expect manifest-missing 1 "2 manifest test(s) did not execute" --filter "FullyQualifiedName~SuiteTests" \
    --manifest "${TMP}/manifest.txt"

expect empty-manifest 2 "lists no tests" --manifest "${TMP}/empty.txt"
expect no-filter 2 "--filter or --manifest is required"

if [ "${failures}" -ne 0 ]; then
    echo "${failures} check(s) failed"
    exit 1
fi
echo "all require-executed checks passed"
