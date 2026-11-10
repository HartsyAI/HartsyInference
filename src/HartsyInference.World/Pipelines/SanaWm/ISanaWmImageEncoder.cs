using HartsyInference.Core.Tensors;

namespace HartsyInference.World.Pipelines.SanaWm;

/// <summary>The seed-frame encoder: LTX-2 VAE encode, then the per-channel latent normalization the DiT samples in.</summary>
public interface ISanaWmImageEncoder
{
    /// <summary>Encodes a host F32 video <c>[1, 3, 1, H, W]</c> in [-1, 1] to the normalized latent <c>[1, 128, 1, H/32, W/32]</c>.
    /// The caller owns the result and still owns <paramref name="video"/>.</summary>
    Tensor Encode(Tensor video);
}
