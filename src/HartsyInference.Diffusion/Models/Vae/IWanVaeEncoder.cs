using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Models.Vae;

/// <summary>Common encode surface for the Wan VAEs (<see cref="Wan21VaeEncoder"/> z=16, <see cref="Wan22VaeEncoder"/>
/// z=48) so a single pipeline can build I2V conditioning latents for any Wan variant.</summary>
public interface IWanVaeEncoder
{
    /// <summary>Encodes one interleaved-RGB24 frame to the normalized first-frame latent <c>[1, z, 1, H/sp, W/sp]</c>.</summary>
    Tensor EncodeRgbFrame(IBackend backend, ReadOnlySpan<byte> rgb24, int width, int height);

    /// <summary>Weights for bulk GPU preload/free.</summary>
    IEnumerable<Tensor> EnumerateWeights();
}
