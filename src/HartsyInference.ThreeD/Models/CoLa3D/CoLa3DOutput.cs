using HartsyInference.Core.Sparse;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.CoLa3D;

/// <summary>Result of one <see cref="CoLa3DDecomposer.Forward"/> call. Owns every tensor it exposes.</summary>
/// <remarks>Zero is a real logit (p = 0.5), so a zero entry does not mark an empty slot. Consumers must test
/// <see cref="ActiveSlots"/> before reading a logit or query row as live.</remarks>
public sealed class CoLa3DOutput : IDisposable
{
    private readonly IReadOnlyList<bool> _activeSlots;
    private int _disposed;

    internal CoLa3DOutput(SparseTensor pointFeatures, Tensor maskLogits, Tensor queryFeatures, bool[] activeSlots)
    {
        PointFeatures = pointFeatures;
        MaskLogits = maskLogits;
        QueryFeatures = queryFeatures;
        _activeSlots = Array.AsReadOnly(activeSlots);
    }

    /// <summary>Point features after the backbone, <c>[N, C]</c>, on the latent's coordinates.</summary>
    public SparseTensor PointFeatures { get; }

    /// <summary>Mask logits <c>[N, Q]</c> over the latent voxels. An entry is written only when its slot is active for the
    /// voxel's sample; every other entry is zero and is not a live logit.</summary>
    public Tensor MaskLogits { get; }

    /// <summary>Query features <c>[Batch, Q, C]</c>. Rows of inactive slots are zero.</summary>
    public Tensor QueryFeatures { get; }

    /// <summary>Slot validity flat <c>[Batch·Q]</c>, indexed <c>sample · Q + slot</c> as <see cref="QueryFeatures"/> is. True for
    /// every slot when the prompts carry no valid mask.</summary>
    public IReadOnlyList<bool> ActiveSlots => _activeSlots;

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        PointFeatures.Feats.Dispose();
        MaskLogits.Dispose();
        QueryFeatures.Dispose();
    }
}
