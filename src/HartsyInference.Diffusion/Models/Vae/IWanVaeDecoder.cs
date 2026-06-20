using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>Common decode surface for the Wan VAEs (<see cref="Wan21VaeDecoder"/> z=16, <see cref="Wan22VaeDecoder"/>
/// z=48) so a single pipeline can drive any Wan variant. Both denormalize the latent internally.</summary>
public interface IWanVaeDecoder
{
    /// <summary>Decodes a normalized latent <c>[1, z, T, H, W]</c> to interleaved RGB <c>[1, 3, T_out, H_out, W_out]</c> in [-1, 1].</summary>
    Tensor Decode(IBackend backend, Tensor latent);

    /// <summary>Weights for bulk GPU preload/free.</summary>
    IEnumerable<Tensor> EnumerateWeights();
}
