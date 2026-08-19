using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Adapters;

/// <summary>Per-step output of <see cref="QwenImageControlNet.Forward"/>: one image-stream residual per
/// ControlNet block, each <c>[1, imgSeqLen, hidden]</c>. The base <c>QwenImageTransformer</c> adds them onto
/// its image stream with diffusers' interval mapping — base block <c>i</c> receives residual
/// <c>i / ceil(baseDepth / count)</c>, so a 5-residual stack covers all 60 base blocks. Caller owns and
/// disposes the tensors after the transformer forward consumes them.</summary>
public sealed record QwenImageControlNetResiduals
{
    /// <summary>Residuals added to the base transformer's image stream, in ControlNet block order.</summary>
    public required Tensor[] BlockResiduals { get; init; }

    /// <summary>Disposes every residual tensor.</summary>
    public void DisposeAll()
    {
        foreach (Tensor t in BlockResiduals) t.Dispose();
    }
}
