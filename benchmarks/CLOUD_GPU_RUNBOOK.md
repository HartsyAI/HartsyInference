# Cloud GPU baseline runbook

Use [methodology](../docs/Research/PROFILING_METHODOLOGY.md) and the [benchmark harness](README.md). This is an execution checklist, not a claim that an unmeasured GPU is verified.

1. Define the hardware gap, workload, artifact destination, budget and shutdown time. Verify current availability/pricing when arranging a rental; old estimates are not a budget.
2. Pin the engine commit, reference revisions, checkpoint hashes and dependencies. Record nvidia-smi, driver, compiler, .NET, GPU topology and free memory. Ensure the driver can JIT the shipped PTX and required compute libraries resolve.
3. Build Release and prepare the baseline virtual environment using repository requirements. Preload necessary checkpoints; keep secrets out of logs and artifacts.
4. Run the smoke harness, inspect failures/skips, then the matched full workload. Record power/clock/thermal limits and other GPU processes; do not alter shared-host settings without authorization.
5. Verify raw C#/reference rows, matched shapes/dtypes, fingerprints, logs and quality results. Missing checkpoints or an aborted test run are incomplete coverage even if some output files exist.
6. Copy artifacts to the agreed durable destination, verify the copy, and update the canonical scoreboard with date/source. benchmarks/results is ignored locally; do not assume output was committed or uploaded.
7. Stop the workload and terminate the rented compute; check storage and other continuing charges. Confirm completion in the provider control plane.

PTX load failures require checking the actual emitted ISA, target and installed driver. Do not “fix” the header manually or assume an old toolkit version supports every shipped artifact.
