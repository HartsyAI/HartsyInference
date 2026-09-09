# Profiling methodology

Commands and setup: [benchmark guide](../../benchmarks/README.md). Results: [scoreboards](../../benchmarks/scoreboards/README.md). Scripts and their schemas are authoritative; the former Phase B pseudocode did not match the implemented analyzer.

## Comparable runs

- Pin source/reference revisions, checkpoint and input hashes, settings, dtype, shape, batch/sequence length, scheduler and step count. Archive an uncommitted diff if testing a working tree.
- Record GPU identity/topology, driver/toolchain/.NET/Python dependencies, CPU/governor, power/clocks, free memory and thermal/contention state. Do not assume a model fits from weight bytes alone.
- Separate load/JIT/cold capture from warm inference. The harness requests one warmup and five measured iterations by default; confirm that this reaches steady state for the workload. Keep warmups out of measured trials.
- Use identical saved inputs across runtimes. Match precision/quality before comparing speed. Explicitly label fallback paths, missing checkpoints and skipped workloads.
- Record total latency, phase times, memory and output quality. Synchronize the measurement boundary for asynchronous GPU work; avoid adding per-op synchronization that changes the workload.

## Find the bottleneck

Start with a whole-run Nsight Systems timeline: host gaps, transfers, allocation/synchronization, then kernel time. Verify NVTX is actually available; its optional native library is not guaranteed by a CUDA driver. Inspect NvtxRange and resolved profiling settings rather than relying on the old compile-time NVTX_ENABLED claim.

Use cudaProfilerApi capture ranges only if the chosen workload invokes profiler start/stop; otherwise the requested window may never be recorded. Nsight report names/options vary by installed tool version; inspect tool help rather than copying old version-specific commands.

Use Nsight Compute on representative hot launches to examine SM/DRAM utilization, eligible warps, cache traffic and tensor-core activity. A fixed launch-skip count does not establish steady state for every workload. Isolated-kernel estimated gains can disappear under end-to-end overlap or other bottlenecks.

Verify CUDA context/stream handling when moving work across threads. Managed Parallel.For is not process forking. Measure on the execution stream used by the backend. Check other GPU users; never kill unrelated processes as benchmark preparation.

## Statistics and artifacts

analyze.py joins operation/shape/dtype keys, reports means and 95% intervals, and marks a Welch p-value below 0.01 significant when available. Significance alone does not prove a useful speedup, and a non-significant result does not prove equality. Inspect effect direction, sample size, variance and matching conditions. The implemented report does not also enforce the old prose's non-overlapping-CI gate.

Preserve raw trials, logs, hardware/software/digest files and analysis outputs. Do not infer success from a best-effort analyzer: it can write a report with missing inputs. Read actual columns in analyze.py and python-baseline/_common.py rather than maintaining duplicate schema declarations.

benchmarks/results is a local output area; explicitly archive artifacts and link from the scoreboard. Redact credentials/private data while retaining enough metadata to reproduce. Hardware coverage is established by dated measurements, not a proposed rental matrix.
