using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Adapters;

/// <summary>Per-step output of <see cref="FluxControlNet.Forward"/>: one image-stream residual per ControlNet
/// double block and (when the checkpoint has them) one per single block, each <c>[1, imgSeqLen, hidden]</c>.
/// The base <c>FluxTransformer</c> adds them onto its own streams with diffusers' interval mapping — base block
/// <c>i</c> receives residual <c>i / ceil(baseDepth / count)</c>, so a 6-residual stack covers all 19 base
/// double blocks. Caller owns and disposes the tensors after the transformer forward consumes them.</summary>
public sealed record FluxControlNetResiduals
{
    /// <summary>Residuals added to the base transformer's double-block image stream, in ControlNet block order.</summary>
    public required Tensor[] DoubleBlockResiduals { get; init; }

    /// <summary>Residuals added to the image rows of the base transformer's single-block stream. Empty when the
    /// checkpoint has no single blocks (Union-Pro-2.0).</summary>
    public required Tensor[] SingleBlockResiduals { get; init; }

    /// <summary>Disposes every residual tensor.</summary>
    public void DisposeAll()
    {
        foreach (Tensor t in DoubleBlockResiduals) t.Dispose();
        foreach (Tensor t in SingleBlockResiduals) t.Dispose();
    }
}
