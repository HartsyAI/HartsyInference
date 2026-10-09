# Reads one dotnet test console log and prints "runs total failed skipped" for tests/dsv41-certification.sh.
# Two console-logger formats appear. The classic one ends each run with a line that starts "Passed!" or "Failed!",
# followed by "Total:", "Failed:" and "Skipped:" fields. The .NET 10 SDK instead prints a block that starts with
# "Test Run Successful." or "Test Run Failed." and then lines "Total tests: N", "Failed: N" and "Skipped: N", ending
# at "Total time:". Any other non-blank line also ends the block, so output after an unfinished summary is not counted.
# The SDK counts are read only inside a block, so a test that prints "Failed: 9" outside one is not counted.
# A log with neither format prints 0 0 0 0, which the runner records as "no test summary".
/^(Passed|Failed)!/ {
    n++
    for (i = 1; i < NF; i++) {
        if ($i == "Total:") t += $(i + 1)
        else if ($i == "Failed:") f += $(i + 1)
        else if ($i == "Skipped:") s += $(i + 1)
    }
}
/^Test Run (Successful|Failed)\./ { n++; block = 1; next }
block && /^ *Total tests: [0-9]+/ { t += $3; next }
block && /^ *Failed: [0-9]+/ { f += $2; next }
block && /^ *Skipped: [0-9]+/ { s += $2; next }
block && /^ *Passed: [0-9]+/ { next }
block && /^ *Total time:/ { block = 0; next }
block && NF > 0 { block = 0 }
END { printf "%d %d %d %d\n", n + 0, t + 0, f + 0, s + 0 }
