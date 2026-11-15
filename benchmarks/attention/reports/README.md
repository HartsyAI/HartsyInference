# PR 01 local evidence status

PR 01 remains **incomplete and draft**. The [3060 report](pr01-local-3060.json) contains exact raw-artifact hashes,
test outcomes, known baseline failures, telemetry identity results, descriptive operation pilots and outstanding
exit work. The [Kitchen dependency lock](k0-windows-3060.lock.json) describes a Python-only K0 cohort; it does not
establish native Sage, native decode or native Sol support. The [runtime index](k0-runtime-index.json) records the
individual probes; unexecuted source cells remain unexecuted.

No paired performance claim, primary workload speed claim or complete parity claim is made. The initial false
Sol D64 invalid classification is preserved with its correction to negative admission and a versioned matrix
amendment requiring review. It is not an upstream-correctness exception.

The CPU run has two failures reproduced on pristine B0 and one skip. GPU resource skips and two passed tests
whose own output says they lacked multiple GPUs are explicitly unvalidated. A successful test-process exit does
not turn those cases into regression evidence.

Continue with the [4090 handoff](../vps-handoff.md). Do not begin PR 02 until the complete PR 01 exit evidence has
been reviewed. Raw binaries, models, tensors, full locks and captures remain in the external artifact directory
identified by the report; retain it through the handoff and exit review.
