using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

/// <summary>One process's half of a multi-node collective: this process is a single rank and passes only its own tensors. Every rank must issue the same collectives in the same order. Implementation: <c>NcclRankComm</c> (HartsyInference.Cuda).</summary>
// TODO(unsure): only NcclRankComm implements this. No host-staged or CPU variant exists, so CPU-only multi-node is unsupported.
public interface IRankCollective : IDisposable
{
    /// <summary>This process's rank.</summary>
    int Rank { get; }

    /// <summary>Total rank count across all processes.</summary>
    int WorldSize { get; }

    /// <summary>"nccl" — the transport decision, for logs and tests.</summary>
    string Transport { get; }

    /// <summary>In-place element-wise sum across all ranks; this rank's buffer ends holding the global sum.</summary>
    void AllReduceSum(Tensor buffer);

    /// <summary>This rank's send block (N elements) is gathered rank-major into every rank's receive buffer (N·<see cref="WorldSize"/> elements).</summary>
    void AllGather(Tensor sendBuffer, Tensor recvBuffer);
}
