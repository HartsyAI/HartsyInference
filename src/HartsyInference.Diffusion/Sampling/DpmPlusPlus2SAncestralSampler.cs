using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Sampling;

/// <summary>DPM-Solver++(2S) ancestral — k-diffusion's <c>dpmpp_2s_ancestral</c>. Second order via a genuine
/// mid-step model evaluation (the "S" is single-step: unlike <see cref="DpmPlusPlus2MSampler"/> it carries no history
/// between steps and pays two forwards instead), with an ancestral noise injection at the end.
///
/// <para>Working in half-log-SNR with <c>σ(λ) = e^(−λ)</c>, the step to <c>σ_down</c> is split at its midpoint
/// <c>r = 1/2</c>:</para>
/// <code>
/// h  = λ_down − λ ;  s = λ + h/2
/// x₂ = (σ(s)/σ)·x       − (1 − e^(−h/2))·denoised
/// x  = (σ_down/σ)·x     − (1 − e^(−h))·denoised(x₂, σ(s))
/// x  = x + noise·σ_up
/// </code>
/// <para>Because it re-derives everything from the current step alone, this one is immune to the schedule-change and
/// first-step artefacts that history-carrying multistep methods have to special-case.</para></summary>
public sealed class DpmPlusPlus2SAncestralSampler : ISampler
{
    private readonly float[] _sigmas;
    private readonly int _seed;
    private readonly float _eta;
    private TensorShape _shape;

    /// <inheritdoc/>
    public string Name => "dpmpp_2s_ancestral";

    /// <inheritdoc/>
    public int StepCount => _sigmas.Length - 1;

    /// <summary>Creates the sampler over a resolved sigma array.</summary>
    public DpmPlusPlus2SAncestralSampler(float[] sigmas, int seed, float eta = 1.0f)
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
        (float sigmaDown, float sigmaUp) = SamplerMath.AncestralStep(sigma, sigmaNext, _eta);

        using Tensor denoised = SamplerMath.PredictDenoised(backend, predictor, z, sigma, stepIndex);

        if (sigmaDown <= 0f)
        {
            // Terminal (or fully-resampled) step: plain Euler onto the denoised estimate.
            float dtTerminal = (sigmaDown - sigma) / sigma;
            SamplerOps.MixInto(backend, z, denoised, 1.0f + dtTerminal, -dtTerminal);
            SamplerOps.AddNoise(backend, z, _shape, _seed, stepIndex, sigmaUp);
            return;
        }

        float lambda = SamplerOps.Lambda(sigma);
        float h = SamplerOps.Lambda(sigmaDown) - lambda;
        float sigmaMid = MathF.Exp(-(lambda + (0.5f * h)));

        using Tensor zMid = new Tensor(z.Shape, DType.F32);
        SamplerOps.SetMix(backend, zMid, z, denoised, sigmaMid / sigma, 1.0f - MathF.Exp(-0.5f * h));

        using Tensor denoisedMid = SamplerMath.PredictDenoised(backend, predictor, zMid, sigmaMid, stepIndex);
        SamplerOps.MixInto(backend, z, denoisedMid, sigmaDown / sigma, 1.0f - MathF.Exp(-h));
        SamplerOps.AddNoise(backend, z, _shape, _seed, stepIndex, sigmaUp);
    }
}
