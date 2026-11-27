using HartsyInference.Core.Tensors;

namespace HartsyInference.World.Pipelines.SanaWm;

/// <summary>The SANA-WM DiT as the sampler drives it: one forward over a guidance batch, each item conditioned on its branch's
/// caption and on the run's camera inputs.</summary>
public interface ISanaWmDenoiser
{
    /// <summary>Whether a forward may hold a single guidance branch, which sequential guidance needs.</summary>
    bool SupportsSingleBranch { get; }

    /// <summary>Predicts the flow velocity for host F32 latents <c>[B, 128, T, H, W]</c>.</summary>
    /// <param name="latents">Latents, one batch item per branch.</param>
    /// <param name="timesteps">Per batch item and latent frame, <c>B·T</c> values in batch-major order.</param>
    /// <param name="branches">The guidance branch of each batch item.</param>
    /// <returns>A host F32 tensor shaped like <paramref name="latents"/>, owned by the caller.</returns>
    Tensor Predict(Tensor latents, ReadOnlySpan<float> timesteps, ReadOnlySpan<SanaWmGuidanceBranch> branches);
}
