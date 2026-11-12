using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.ThreeD.Models.Hunyuan3D.Omni;

/// <summary>The DiT context of Hunyuan3D-Omni for the two guidance branches (<c>encode_cond</c> of <c>Hunyuan3DOmniSiTFlowMatchingPipeline</c>): <c>[DINO tokens | control tokens | signal tokens]</c> for the conditional branch, and the same
/// with the DINO tokens replaced by zeros (<c>DinoImageEncoder.unconditional_embedding</c>) for the unconditional one; the control and signal tokens are kept in both.</summary>
internal static class OmniContext
{
    /// <summary>Builds both contexts, <c>[1, image + control, width]</c>. The caller disposes them; <paramref name="image"/> and <paramref name="controlTokens"/> stay the caller's.</summary>
    internal static (Tensor Cond, Tensor Uncond) Build(IBackend backend, Tensor image, Tensor controlTokens)
    {
        ArgumentNullException.ThrowIfNull(backend);
        if (image.Shape.Rank != 3 || controlTokens.Shape.Rank != 3 || image.Shape[0] != 1 || controlTokens.Shape[0] != 1 || image.Shape[2] != controlTokens.Shape[2])
            throw new ArgumentException($"The image tokens {image.Shape} and the control tokens {controlTokens.Shape} must be [1, tokens, width] with the same width.");
        TensorShape shape = new(1, image.Shape[1] + controlTokens.Shape[1], image.Shape[2]);
        Tensor cond = new(shape, DType.F32);
        Tensor? uncond = null;
        try
        {
            backend.Concat(cond, [image, controlTokens], 1);
            using Tensor zeros = new(image.Shape, DType.F32);
            uncond = new Tensor(shape, DType.F32);
            backend.Concat(uncond, [zeros, controlTokens], 1);
            return (cond, uncond);
        }
        catch
        {
            cond.Dispose();
            uncond?.Dispose();
            throw;
        }
    }
}
