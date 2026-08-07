using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

/// <summary>Rank-indexed collectives over per-rank tensors: entry i of every list belongs to rank i and lives
/// (or is staged) on rank i's backend. Single-process — one call on one thread drives all ranks.
/// Implementations: <c>NcclComm</c> (HartsyInference.Cuda; device-side) and <see cref="HostStagedComm"/>
/// (universal fallback, correct on any backend mix). Obtain one via <c>CollectiveComm.Create</c>.
/// <para>Post-call semantics: each rank's result is authoritative wherever the transport left it (device for
/// NCCL, host for host-staged) — the Tensor binding machinery reconciles on the next access; callers never
/// copy manually.</para></summary>
public interface ICollectiveComm : IDisposable
{
    /// <summary>Number of ranks this communicator was created over.</summary>
    int RankCount { get; }

    /// <summary>"nccl" or "host-staged" — the factory's transport decision, for logs and tests.</summary>
    string Transport { get; }

    /// <summary>In-place element-wise sum across ranks: every <paramref name="buffers"/>[i] ends holding
    /// Σ_j buffers[j]. All tensors must share element count and dtype.</summary>
    void AllReduceSum(IReadOnlyList<Tensor> buffers);

    /// <summary>Each rank's <paramref name="sendBuffers"/>[i] (N elements) is concatenated rank-major into
    /// every <paramref name="recvBuffers"/>[i] (N·<see cref="RankCount"/> elements): rank j's block lands at
    /// element offset j·N on every rank.</summary>
    void AllGather(IReadOnlyList<Tensor> sendBuffers, IReadOnlyList<Tensor> recvBuffers);
}
