using HartsyInference.Core.Backends;
using HartsyInference.Core.Layers.Attention;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Where a cached reference's keys and values go relative to a block's live ones.</summary>
public enum ReferenceKvOrder
{
    /// <summary>After the live tokens: the references follow the target, as the reference adapter and MOSS place them.</summary>
    Append,

    /// <summary>Before the live tokens: a cached text prefix, as Qwen-Image's text-conditioned prefix is laid out.</summary>
    Prepend,
}

/// <summary>Joins a cached reference's keys and values to a block's live ones, before the head permute, so every live query
/// also attends to the reference. The cache is borrowed and never disposed here; the live K and V are taken over.</summary>
public sealed class ReferenceKvRewrite(Tensor cachedKeys, Tensor cachedValues, ReferenceKvOrder order) : IKvRewrite
{
    /// <inheritdoc/>
    public (Tensor K, Tensor V) Rewrite(IBackend backend, Tensor k, Tensor v)
    {
        Tensor? keys = null;
        try
        {
            keys = Join(backend, order, k, cachedKeys);
            k.Dispose();
            Tensor values = Join(backend, order, v, cachedValues);
            v.Dispose();
            return (keys, values);
        }
        catch
        {
            keys?.Dispose();
            k.Dispose();
            v.Dispose();
            throw;
        }
    }

    private static Tensor Join(IBackend backend, ReferenceKvOrder order, Tensor live, Tensor cached) =>
        order == ReferenceKvOrder.Append
            ? ReferenceKvCache.JoinTokens(backend, live, cached)
            : ReferenceKvCache.JoinTokens(backend, cached, live);
}
