using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Sampling;

/// <summary>k-diffusion's <c>euler_ancestral</c> — after each deterministic move it re-injects fresh noise, so the
/// trajectory is stochastic rather than an ODE solve. With <c>dpmpp_2m</c>, one of the two most-used samplers in the
/// SD/SDXL community.
///
/// <para>Per step: denoise, split the jump into a deterministic part (<c>sigmaDown</c>) and a resampled part
/// (<c>sigmaUp</c>) via <see cref="SamplerMath.AncestralStep"/>, take the Euler move to <c>sigmaDown</c>, then add
/// <c>sigmaUp</c>-scaled Gaussian noise. On the final step <c>sigmaNext</c> is 0, <c>sigmaUp</c> is 0, and the method
/// degenerates to plain Euler — adding noise there would re-noise the finished image.</para></summary>
public sealed class EulerAncestralSampler : ISampler
{
    private readonly float[] _sigmas;
    private readonly int _seed;
    private readonly float _eta;
    private TensorShape _shape;

    /// <inheritdoc/>
    public string Name => "euler_ancestral";

    /// <inheritdoc/>
    public int StepCount => _sigmas.Length - 1;

    /// <summary>Creates the sampler. <paramref name="eta"/> scales how much of each step is resampled; 1.0 is the
    /// k-diffusion default and what ComfyUI's <c>euler_ancestral</c> uses.</summary>
    public EulerAncestralSampler(float[] sigmas, int seed, float eta = 1.0f)
    {
        ArgumentNullException.ThrowIfNull(sigmas);
        if (sigmas.Length < 2)
        {
            throw new ArgumentException($"Need at least 2 sigmas (one step plus the terminal zero); got {sigmas.Length}.",
                nameof(sigmas));
        }
        _sigmas = sigmas;
        _seed = seed;
        _eta = eta;
    }

    /// <inheritdoc/>
    public void Reset(TensorShape latentShape) => _shape = latentShape;

    /// <inheritdoc/>
    public void Step(IBackend backend, Tensor z, IDenoisePredictor predictor, int stepIndex)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(z);
        ArgumentNullException.ThrowIfNull(predictor);
        float sigma = _sigmas[stepIndex];
        float sigmaNext = _sigmas[stepIndex + 1];
        (float sigmaDown, float sigmaUp) = SamplerMath.AncestralStep(sigma, sigmaNext, _eta);

        using Tensor denoised = SamplerMath.PredictDenoised(backend, predictor, z, sigma, stepIndex);
        // d = (z − denoised)/sigma; z ← z + d·(sigmaDown − sigma), folded into one affine mix.
        float dt = (sigmaDown - sigma) / sigma;
        SamplerOps.MixInto(backend, z, denoised, 1.0f + dt, -dt);
        SamplerOps.AddNoise(backend, z, _shape, _seed, stepIndex, sigmaUp);
    }
}
