using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Denoisers.DiTBlocks;

/// <summary>Runs a reference through a model's blocks once, at the references' own timestep, and collects each block's keys and
/// values into a <see cref="ReferenceKvCache"/>. Every model that caches references runs this same loop; only the block call differs.</summary>
public static class ReferenceKvPrepass
{
    /// <summary>Runs block <paramref name="layer"/> over <paramref name="stream"/>. Returns the next stream and hands back the block's
    /// rotated keys and values, which the cache takes over.</summary>
    public delegate Tensor PrefixLayer(int layer, Tensor stream, out Tensor keys, out Tensor values);

    /// <summary>Builds the cache for <paramref name="layers"/> blocks, leaving the layers in <paramref name="skipLayers"/> uncached.
    /// <paramref name="embedded"/> is borrowed; the streams the loop creates are disposed as it goes.</summary>
    public static ReferenceKvCache Build(IBackend backend, int layers, IReadOnlyCollection<int> skipLayers, Tensor embedded,
        PrefixLayer prefix, long fingerprint, string family)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(prefix);
        using ReferenceKvCache.Builder builder = new(layers);
        Tensor stream = embedded;
        bool ownsStream = false;
        try
        {
            for (int i = 0; i < layers; i++)
            {
                if (skipLayers.Contains(i))
                    continue;
                Tensor next = prefix(i, stream, out Tensor keys, out Tensor values);
                try
                {
                    builder.Set(i, keys, values);
                }
                catch
                {
                    keys.Dispose();
                    values.Dispose();
                    next.Dispose();
                    throw;
                }
                if (ownsStream)
                    stream.Dispose();
                stream = next;
                ownsStream = true;
            }
        }
        finally
        {
            if (ownsStream)
                stream.Dispose();
        }
        return builder.Build(backend, fingerprint, family);
    }
}
