using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Sampling;

/// <summary>Independent seeded Gaussian per draw, ComfyUI's <c>default_noise_sampler</c> role.</summary>
public sealed class SeededNoiseSource(TensorShape shape, int seed) : INoiseSource
{
    /// <inheritdoc/>
    public Tensor Sample(int stepIndex, int subDraw, float sigma, float sigmaNext) =>
        SeedGenerator.CreateNoise(shape, SamplerOps.StepSeed(seed, stepIndex, subDraw));

    /// <inheritdoc/>
    public void Dispose()
    {
    }
}
