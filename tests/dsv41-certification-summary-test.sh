#!/usr/bin/env bash
# Checks the certification runner's summary parser (tests/dsv41-certification-summary.awk) against the console logs it reads.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AWK_FILE="${ROOT}/tests/dsv41-certification-summary.awk"
TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT
failures=0

check() { # name, expected "runs total failed skipped", log text
    printf '%s' "$3" > "${TMP}/$1.log"
    got="$(awk -f "${AWK_FILE}" "${TMP}/$1.log")"
    if [ "${got}" = "$2" ]; then
        echo "PASS $1: ${got}"
    else
        echo "FAIL $1: expected '$2', got '${got}'"
        failures=$((failures + 1))
    fi
}

check classic-passed "1 5 0 0" "Passed!  - Failed:     0, Passed:     5, Skipped:     0, Total:     5, Duration: 2 s
"
check sdk-successful "1 5 0 0" "Test Run Successful.
Total tests: 5
     Passed: 5
 Total time: 2.1925 Hours
"
check sdk-failed "1 3 1 0" "Test Run Failed.
Total tests: 3
     Failed: 1
     Passed: 2
 Total time: 4.0000 Seconds
"
check sdk-skipped "1 4 0 1" "Test Run Successful.
Total tests: 4
     Skipped: 1
     Passed: 3
 Total time: 1.0000 Seconds
"
check sdk-stray-count-ignored "1 2 0 0" "Failed: 9
Total tests: 9
Test Run Successful.
Total tests: 2
     Passed: 2
 Total time: 1.0000 Seconds
"
check sdk-two-blocks "2 7 1 0" "Test Run Successful.
Total tests: 5
     Passed: 5
 Total time: 2.1925 Hours
Test Run Failed.
Total tests: 2
     Failed: 1
     Passed: 1
 Total time: 1.0000 Seconds
"
check sdk-unclosed-block-ignored "1 3 0 0" "Test Run Successful.
Total tests: 3
     Passed: 3
some test output
Failed: 9
"
check no-summary "0 0 0 0" "Starting test execution, please wait...
"

if [ "${failures}" -ne 0 ]; then
    echo "${failures} check(s) failed"
    exit 1
fi
echo "all summary checks passed"
