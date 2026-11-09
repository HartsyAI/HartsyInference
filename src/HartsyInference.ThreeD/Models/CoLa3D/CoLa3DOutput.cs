using HartsyInference.Core.Sparse;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.CoLa3D;

/// <summary>Result of one <see cref="CoLa3DDecomposer.Forward"/> call. Owns every tensor it exposes.</summary>
public sealed class CoLa3DOutput : IDisposable
{
    private int _disposed;

    internal CoLa3DOutput(SparseTensor pointFeatures, Tensor maskLogits, Tensor queryFeatures)
    {
        PointFeatures = pointFeatures;
        MaskLogits = maskLogits;
        QueryFeatures = queryFeatures;
    }

    /// <summary>Point features after the backbone, <c>[N, C]</c>, on the latent's coordinates.</summary>
    public SparseTensor PointFeatures { get; }

    /// <summary>Mask logits <c>[N, Q]</c> over the latent voxels. Columns of invalid query slots are zero.</summary>
    public Tensor MaskLogits { get; }

    /// <summary>Query features <c>[Batch, Q, C]</c>. Rows of invalid query slots are zero.</summary>
    public Tensor QueryFeatures { get; }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        PointFeatures.Feats.Dispose();
        MaskLogits.Dispose();
        QueryFeatures.Dispose();
    }
}
