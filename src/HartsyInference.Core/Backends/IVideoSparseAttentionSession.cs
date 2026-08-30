using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Backends;

/// <summary>Opaque generation-scoped sparse-attention state owned by one backend.</summary>
public interface IVideoSparseAttentionSession : IDisposable
{
    /// <summary>The immutable routing profile this session implements.</summary>
    VideoSparseAttentionProfileKind Profile { get; }

    /// <summary>Executes exact routed attention plus all-block attention over Q/K/V block means, broadcasts the
    /// coarse result to each query-block row, and writes <c>fine + gate * coarse</c>. Q/K/V, the vector gate, and
    /// output are <c>[1,H,S,D]</c>.</summary>
    void Execute(Tensor output, Tensor query, Tensor key, Tensor value, Tensor gate);
}
