using HartsyInference.Core.Tensors;

namespace HartsyInference.World.Pipelines.SanaWm;

/// <summary>The LTX-2 VAE decode of a sampled latent. The decoder un-normalizes the latent itself.</summary>
public interface ISanaWmVideoDecoder
{
    /// <summary>Decodes a host F32 normalized latent <c>[1, 128, T, H, W]</c> to RGB <c>[1, 3, 8(T−1)+1, 32H, 32W]</c> in [-1, 1].
    /// The caller owns the result and still owns <paramref name="latent"/>.</summary>
    Tensor Decode(Tensor latent);
}
