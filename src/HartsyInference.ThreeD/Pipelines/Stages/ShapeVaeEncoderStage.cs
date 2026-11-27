using HartsyInference.Core.Backends;
using HartsyInference.Core.Sparse;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Geometry;
using HartsyInference.ThreeD.Models.TrellisV2;

namespace HartsyInference.ThreeD.Pipelines.Stages;

/// <summary>Dual grid to shape-VAE latent: builds the vertex-offset and intersection-flag inputs, then runs <see cref="ShapeVaeEncoder"/>.</summary>
public sealed class ShapeVaeEncoderStage : ISparseShapeEncoder, IDisposable
{
    private readonly ShapeVaeEncoder _encoder;
    private bool _disposed;

    /// <param name="encoder">Built from the shape VAE encoder weights; this stage owns it.</param>
    public ShapeVaeEncoderStage(ShapeVaeEncoder encoder) => _encoder = encoder ?? throw new ArgumentNullException(nameof(encoder));

    /// <inheritdoc/>
    public SparseTensor Encode(IBackend backend, DualGridVoxels voxels, int resolution)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(voxels);
        int n = voxels.Coords.Length / 3;
        if (n == 0)
            throw new ArgumentException("The dual grid holds no voxels.", nameof(voxels));

        int[] coords = SparseCoords.WithBatch(voxels.Coords);
        Tensor offsets = new(new TensorShape(n, 3), DType.F32);
        Tensor flags = new(new TensorShape(n, 3), DType.F32);
        Span<float> o = offsets.AsSpan<float>();
        Span<float> f = flags.AsSpan<float>();
        for (int i = 0; i < n * 3; i++)
        {
            o[i] = voxels.DualVertices[i] - voxels.Coords[i];
            f[i] = voxels.Intersected[i] ? 1f : 0f;
        }

        SparseTensor mean;
        try
        {
            mean = _encoder.Encode(backend, new SparseTensor(offsets, coords, resolution), new SparseTensor(flags, coords, resolution));
        }
        finally
        {
            offsets.Dispose();
            flags.Dispose();
        }

        // The encoder returns batch-first [1, M, C]; the latent contract is [M, C] with owned storage.
        try
        {
            Tensor latent = new(new TensorShape(mean.Count, mean.Channels), DType.F32);
            mean.Feats.AsReadOnlySpan<float>().CopyTo(latent.AsSpan<float>());
            return new SparseTensor(latent, mean.Coords, mean.Resolution);
        }
        finally
        {
            mean.Feats.Dispose();
        }
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
