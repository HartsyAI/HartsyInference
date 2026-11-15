# PR 01 local evidence status

Subsequent semantic work is tracked separately in the [PR 02 progress report](pr02-semantic-progress.md).

PR 01 remains **incomplete and draft**. The [3060 report](pr01-local-3060.json) contains exact raw-artifact hashes,
test outcomes, known baseline failures, telemetry identity results, descriptive operation pilots and outstanding
exit work. The [Kitchen dependency lock](k0-windows-3060.lock.json) describes a Python-only K0 cohort; it does not
establish native Sage, native decode or native Sol support. The [runtime index](k0-runtime-index.json) records the
individual probes; unexecuted source cells remain unexecuted.

No paired performance claim, primary workload speed claim or complete parity claim is made. The initial false
Sol D64 invalid classification is preserved with its correction to negative admission and a versioned matrix
amendment requiring review. It is not an upstream-correctness exception.

The CPU run has two failures reproduced on pristine B0 and one skip. GPU resource skips and reported passes
whose own output says they lacked multiple GPUs are explicitly unvalidated. A successful test-process exit does
not turn those cases into regression evidence.

Continue with the [4090 handoff](../vps-handoff.md). The user's subsequent authorization to start PR 02 is recorded
in the [implementation queue](../proposed-queue.md); it does not certify the incomplete PR 01 evidence or create Bfix.
Raw binaries, models, tensors, full locks and captures remain in the external artifact directory
identified by the report; retain it through the handoff and exit review.
