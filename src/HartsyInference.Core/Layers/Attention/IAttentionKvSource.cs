using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Core.Layers.Attention;

/// <summary>The key/value store an attention call appends to and attends over; a cache implements it, and the layer holds no
/// state of its own.</summary>
public interface IAttentionKvSource
{
    /// <summary>Appends head-major K/V <c>[1, kvHeads, tNew, D]</c> to <paramref name="layer"/>; does not advance any committed length.</summary>
    void AppendKv(IBackend backend, int layer, Tensor k, Tensor v);

    /// <summary>The cache-owned K of <paramref name="layer"/>, <c>[1, kvHeads, &gt;= valid, D]</c>; the caller never disposes it.</summary>
    Tensor KeyBuffer(int layer);

    /// <summary>The cache-owned V of <paramref name="layer"/>, <c>[1, kvHeads, &gt;= valid, D]</c>; the caller never disposes it.</summary>
    Tensor ValueBuffer(int layer);
}
