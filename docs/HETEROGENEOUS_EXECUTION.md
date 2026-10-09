# Heterogeneous expert execution

How one layer's routed experts run across the GPU and the CPU. This document covers M5. Placement comes from
`ExpertScheduler.Plan` (see [MOE_ARCHITECTURE.md](MOE_ARCHITECTURE.md)); execution comes from `HeterogeneousExpertExecutor`.

## What is in place

- **Plan.** `ExpertScheduler.Plan` assigns each routed expert a placement (`Gpu` or `Cpu`) and a row count, and pins the
  resident experts so they cannot be evicted before they run. Planning never uploads.
- **Execute.** `HeterogeneousExpertExecutor.Execute` takes the plan and the expert-major rows (as `MoeBuildDispatch` lays
  them out) and runs each assignment on its side. GPU assignments go through `IExpertDeviceRunner`; CPU assignments go
  through the F32 reference (`ExpertProgramReference`). Output uses the input layout.
- **Contract.** A placement changes where an expert runs, not which rows it receives or where its output lands. The tests
  check this by running the same plan under all-CPU, all-GPU and mixed placements and requiring identical output. They use a
  reference device, so they verify placement and layout, not device numerics.

## What is not yet in place

The CUDA device runner is not built. `IExpertDeviceRunner` is synchronous: `Run` returns only when `y` is complete on the
host. The asynchronous protocol below is the contract the CUDA adapter must meet. It is not implemented or tested yet, and
the CUDA PR must not claim it is.

## Asynchronous handoff protocol (normative for the CUDA adapter)

Applies once a device run is queued rather than completed before `Run` returns. Each step is ordered on the stream named.

1. The GPU records an event after the activations and routing for the layer are written.
2. The CPU waits on that event, then computes its assignments into pinned host output. If pinning fails, use pageable memory
   with a synchronous copy; behavior does not change, only speed.
3. The CPU copies its output to the device on the compute stream, after the GPU's own writes to the same buffer.
4. The GPU combines the outputs (`MoeCombine`) only after the copy from step 3 has completed.

Every handoff carries a monotonically increasing sequence number. A buffer is reused only after the consumer has acknowledged
the sequence that last used it. Flags are not used: a boolean cannot tell a late acknowledgement from a current one.

Tests for the CUDA adapter must cover: a sequence number acknowledged twice, a reuse before acknowledgement, and a
cancellation while a split is in flight. None of these can pass on the CPU lane.

## Correctness

Heterogeneous output is compared to the all-CPU reference under the same rule as the rest of the MoE runtime: identical route
IDs and weights, and a bounded per-layer error. Greedy token identity is required only for configurations where placement is
fixed. Placement that depends on runtime state changes rounding and can change greedy output, as the Strata notes record.
