using HartsyInference.Core.Backends;
using HartsyInference.Core.Sparse;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Models.TrellisV2;

namespace HartsyInference.ThreeD.Pipelines.Stages;

/// <summary>Shape-VAE latent to dual-grid field: runs <see cref="ShapeVaeDecoder"/> and splits its head. It also serves TRELLIS.2's
/// cascade, which asks the decoder only where its first four levels put voxels.</summary>
public sealed class ShapeVaeDecoderStage : ISparseShapeDecoder, ISparseShapeUpsampler, IDisposable
{
    private readonly ShapeVaeDecoder _decoder;
    private bool _disposed;

    /// <param name="decoder">Built from the shape VAE decoder weights; this stage owns it.</param>
    public ShapeVaeDecoderStage(ShapeVaeDecoder decoder) => _decoder = decoder ?? throw new ArgumentNullException(nameof(decoder));

    /// <inheritdoc/>
    public DualGridField Decode(IBackend backend, SparseTensor latent)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(latent);
        SparseTensor head = _decoder.Decode(backend, latent);
        try
        {
            return ShapeVaeHead.Split(head);
        }
        finally
        {
            head.Feats.Dispose();
        }
    }

    /// <inheritdoc/>
    public int[] Upsample(IBackend backend, SparseTensor latent, int times) => _decoder.Upsample(backend, latent, times);

    /// <summary>Releases the decoder and its weights.</summary>
    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
        _decoder.Dispose();
    }
}
