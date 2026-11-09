# Multi-node NCCL bootstrap — plan and status

Scope: letting one model run across processes on several datacenter nodes, each process owning one GPU. Design basis is
[MULTI_GPU_PARALLELISM.md §4.3](../Research/MULTI_GPU_PARALLELISM.md). Matrix entry: `dist.multinode`, still
`Unimplemented` until rented two-node hardware passes its acceptance test.

## Built (bootstrap layer)

| Piece | Location | Needs GPU |
|---|---|---|
| Rendezvous options (`WorldSize`, `Rank`, `MasterAddress`, `MasterPort`, `JobToken`, `Timeout`) | `src/HartsyInference.Core/Distributed/DistributedRendezvousOptions.cs` | No |
| TCP exchange of the 128-byte NCCL unique ID from rank 0 to all ranks | `src/HartsyInference.Core/Distributed/UniqueIdRendezvous.cs` | No |
| `IRankCollective` (one local rank; `AllReduceSum`, `AllGather`) | `src/HartsyInference.Core/Backends/IRankCollective.cs` | No |
| `NcclRankComm.JoinAsync` → `ncclCommInitRank` on the local `CudaBackend` | `src/HartsyInference.Cuda/NcclRankComm.cs` | Yes (libnccl) |

Rendezvous protocol (v1, little-endian): hello = `"HRV1"` · version · rank · world size · SHA-256(job token), 48 bytes.
Rank 0 validates each hello (magic, version, world size, rank range, token digest in constant time, no duplicate rank),
holds the accepted sockets until all ranks have joined, then sends `status · length · payload` to each. Rejected or stalled
hellos are dropped and the wait continues. Clients retry connect with backoff while rank 0 is not yet listening. The
deadline fails the exchange; rank 0 names the missing ranks.

Tests: `tests/HartsyInference.Core.Tests/Distributed/UniqueIdRendezvousTests.cs` (7 loopback tests, run on any host);
`tests/HartsyInference.Cuda.Tests/NcclRankCommTests.cs` (two rank threads, one per GPU, join → AllReduce → AllGather,
bit-exact; skips without CUDA, two GPUs or libnccl).

## Decisions

1. **Process per rank for multi-node.** `NcclComm` (`ncclCommInitAll`, all ranks in one process) is unchanged for
   single-node. Multi-node uses `NcclRankComm`, one rank per process, the usual NCCL model.
2. **Rendezvous lives in Core, not `Engine/Distributed/`.** The matrix's planned location would make `Cuda` depend on
   `Engine`, which is the wrong direction. The NCCL-specific join lives in Cuda.
3. **No host-staged fallback for multi-node.** A single process cannot stand in for other nodes; missing libnccl fails the
   join instead of computing a partial result.
4. **Security is a shared token on a private network.** Plain TCP, no TLS. Run the rendezvous port on the fabric network
   only; the unique ID grants membership in the communicator.
5. **The engine reads no environment variables.** Launchers (Slurm, Kubernetes, torchrun-style wrappers) map their
   environment to `DistributedRendezvousOptions` at the CLI/API layer. NCCL's own knobs (`NCCL_SOCKET_IFNAME`,
   `NCCL_IB_HCA`, `NCCL_IB_DISABLE`) are set by the launcher before the process starts.

## Not built yet

- **Engine wiring.** `TextService` builds tensor-parallel slots in-process. `TensorParallelTransformer` drives all ranks
  from one thread and loads every rank's weights into one process. Multi-node needs a per-process driver that runs only
  the local rank's partial loop and calls `IRankCollective.AllReduceSum` at the two per-layer seams, plus loading only the
  local rank's weight slice. The phase-split structure in that file is the intended seam.
- **Hang handling after join.** `ncclCommInitRank` blocks if a rank dies after the rendezvous. Needs
  `ncclCommGetAsyncError` polling and `ncclCommAbort` on timeout.
- **Multi-process smoke runner.** Launch N processes and compare against single-device output; the current test only covers
  two threads in one process.
- **AllToAll for expert parallel** (`dist.ep`) and pipeline send/recv across nodes.

## Verification gates

- Unit: rendezvous tests above (run on any host).
- GpuIntegration: `NcclRankCommTests` on a 2-GPU Linux host with libnccl. Not run here: this workstation is Windows and
  NCCL ships for Linux only.
- Acceptance (`dist.multinode`): exact-token parity with a single-device run over two rented nodes, via the `two-node` lane
  of `tests/dsv41-certification.sh` (see [DSV41_RENTAL_RUNBOOK.md](DSV41_RENTAL_RUNBOOK.md)).

## Open questions

- Which fabric is the target: InfiniBand/RoCE, or TCP-only Ethernet? This sets the NCCL transport settings to document
  and the minimum throughput to measure.
- Which first model: DeepSeek-V4.1 (`dist.multinode` in the matrix) or a dense model under tensor parallelism?
- Which launcher is the first target?
