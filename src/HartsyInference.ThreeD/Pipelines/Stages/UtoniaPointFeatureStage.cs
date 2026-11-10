using HartsyInference.Core.Backends;
using HartsyInference.Core.Sparse;
using HartsyInference.ThreeD.Models.Utonia;

namespace HartsyInference.ThreeD.Pipelines.Stages;

/// <summary>Utonia encoder features (<c>forward_level4</c>) for single-sample voxels; the output sits on the grid 16x coarser.</summary>
public sealed class UtoniaPointFeatureStage : IPointFeatureExtractor, IDisposable
{
    private readonly UtoniaEncoder _encoder;
    private readonly int _totalStride;
    private bool _disposed;

    /// <param name="encoder">Built from the Utonia weights; this stage owns it.</param>
    /// <param name="totalStride">Product of the encoder stage strides; the 1024 CoLa3D input grid maps to 64 at 16.</param>
    public UtoniaPointFeatureStage(UtoniaEncoder encoder, int totalStride)
    {
        _encoder = encoder ?? throw new ArgumentNullException(nameof(encoder));
        _totalStride = totalStride > 0 ? totalStride : throw new ArgumentOutOfRangeException(nameof(totalStride), totalStride, "Must be positive.");
    }

    /// <inheritdoc/>
    public SparseTensor Extract(IBackend backend, int[] coords, int resolution)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(coords);
        if (coords.Length == 0 || coords.Length % 4 != 0)
            throw new ArgumentException("Utonia coords must be non-empty rows of (batch, x, y, z).", nameof(coords));
        if (resolution % _totalStride != 0)
            throw new ArgumentException($"Utonia input resolution {resolution} must be a multiple of {_totalStride}.", nameof(resolution));

        UtoniaFeatures features = _encoder.ForwardLevel4(coords, 1);
        return new SparseTensor(features.Feats, SparseCoords.WithBatch(features.GridCoord), resolution / _totalStride);
    }

    /// <summary>Releases the encoder and its weights.</summary>
    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
        _encoder.Dispose();
    }
}
