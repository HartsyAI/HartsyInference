using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Sampling;

/// <summary>k-diffusion's <c>dpm_2</c> (and, with <c>eta &gt; 0</c>, <c>dpm_2_ancestral</c>) — a second-order midpoint
/// method. Where <see cref="HeunSampler"/> corrects using the derivative at the step's END, this one steps to the
/// geometric MIDPOINT in sigma, re-evaluates there, and takes the whole step with that derivative.
///
/// <para>The midpoint is geometric — <c>exp((log σ + log σ_next)/2)</c>, not <c>(σ + σ_next)/2</c> — because the
/// diffusion ODE is near-linear in log-sigma, not in sigma. Using the arithmetic mean is a silent accuracy loss that
/// looks like a slightly soft image rather than an error.</para>
///
/// <para>Two model evaluations per step, same as heun. The final step (<c>sigmaNext == 0</c>) degrades to Euler.</para></summary>
public sealed class Dpm2Sampler : ISampler
{
    private readonly float[] _sigmas;
    private readonly int _seed;
    private readonly float _eta;
    private TensorShape _shape;

    /// <inheritdoc/>
    public string Name => _eta > 0f ? "dpm_2_ancestral" : "dpm_2";

    /// <inheritdoc/>
    public int StepCount => _sigmas.Length - 1;

    /// <summary>Creates the sampler. <paramref name="eta"/> above zero selects the ancestral variant, which resamples
    /// part of each step as noise instead of moving there deterministically.</summary>
    public Dpm2Sampler(float[] sigmas, int seed, float eta = 0f)
    {
        ArgumentNullException.ThrowIfNull(sigmas);
        if (sigmas.Length < 2)
        {
            throw new ArgumentException($"Need at least 2 sigmas; got {sigmas.Length}.", nameof(sigmas));
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
        (float sigmaDown, float sigmaUp) = _eta > 0f
            ? SamplerMath.AncestralStep(sigma, sigmaNext, _eta)
            : (sigmaNext, 0f);

        using Tensor denoised = SamplerMath.PredictDenoised(backend, predictor, z, sigma, stepIndex);
        using Tensor d = new Tensor(z.Shape, DType.F32);
        SamplerOps.SetMix(backend, d, z, denoised, 1.0f / sigma, -1.0f / sigma);

        if (sigmaDown <= 0f)
        {
            SamplerOps.MixInto(backend, z, d, 1.0f, sigmaDown - sigma);
            SamplerOps.AddNoise(backend, z, _shape, _seed, stepIndex, sigmaUp);
            return;
        }

        float sigmaMid = MathF.Exp(0.5f * (SamplerOps.LogSigma(sigma) + SamplerOps.LogSigma(sigmaDown)));
        using Tensor zMid = new Tensor(z.Shape, DType.F32);
        SamplerOps.SetMix(backend, zMid, z, d, 1.0f, sigmaMid - sigma);

        using Tensor denoisedMid = SamplerMath.PredictDenoised(backend, predictor, zMid, sigmaMid, stepIndex);
        using Tensor dMid = new Tensor(z.Shape, DType.F32);
        SamplerOps.SetMix(backend, dMid, zMid, denoisedMid, 1.0f / sigmaMid, -1.0f / sigmaMid);

        SamplerOps.MixInto(backend, z, dMid, 1.0f, sigmaDown - sigma);
        SamplerOps.AddNoise(backend, z, _shape, _seed, stepIndex, sigmaUp);
    }
}
