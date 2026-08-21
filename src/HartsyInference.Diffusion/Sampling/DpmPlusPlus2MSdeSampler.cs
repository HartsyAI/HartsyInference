using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Sampling;

/// <summary>DPM-Solver++(2M) SDE — k-diffusion's <c>dpmpp_2m_sde</c>, the stochastic sibling of
/// <see cref="DpmPlusPlus2MSampler"/> and (usually paired with the Karras schedule, as
/// <c>dpmpp_2m_sde_karras</c>) one of the two most-recommended samplers in the SDXL community.
///
/// <para>Same one-evaluation-per-step second-order structure, but each step contracts toward the denoised estimate by
/// an extra <c>e^(−ηh)</c> and then re-injects the removed variance as noise. Per step with <c>h = λ_next − λ</c>:</para>
/// <code>
/// x ← (σ_next/σ)·e^(−ηh)·x + (1 − e^(−h−ηh))·D
/// D  = denoised + (1/2r)·(denoised − denoised_prev)     // midpoint solver
/// x ← x + noise·σ_next·sqrt(1 − e^(−2ηh))
/// </code>
/// <para>The midpoint solver is used rather than heun; ComfyUI exposes both and defaults to midpoint. As with every
/// ancestral method the terminal step is special: at <c>σ_next == 0</c> the result is the denoised estimate itself and
/// no noise is added, or the finished image would be re-noised.</para>
///
/// <para>Reproducibility bound as documented on <see cref="SamplerOps.AddNoise"/>: a seeded Gaussian, not ComfyUI's
/// BrownianTree, so same-seed images differ from Comfy's while remaining a correct SDE solve.</para></summary>
public sealed class DpmPlusPlus2MSdeSampler : ISampler
{
    private readonly float[] _sigmas;
    private readonly int _seed;
    private readonly float _eta;
    private Tensor? _previousDenoised;
    private float _previousH;
    private TensorShape _shape;

    /// <inheritdoc/>
    public string Name => "dpmpp_2m_sde";

    /// <inheritdoc/>
    public int StepCount => _sigmas.Length - 1;

    /// <summary>Creates the sampler. <paramref name="eta"/> controls how much of each step is resampled; 1.0 is the
    /// k-diffusion and ComfyUI default.</summary>
    public DpmPlusPlus2MSdeSampler(float[] sigmas, int seed, float eta = 1.0f)
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
    public void Reset(TensorShape latentShape)
    {
        _shape = latentShape;
        _previousDenoised?.Dispose();
        _previousDenoised = null;
        _previousH = 0f;
    }

    /// <inheritdoc/>
    public void Step(IBackend backend, Tensor z, IDenoisePredictor predictor, int stepIndex)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(z);
        ArgumentNullException.ThrowIfNull(predictor);
        float sigma = _sigmas[stepIndex];
        float sigmaNext = _sigmas[stepIndex + 1];

        Tensor denoised = SamplerMath.PredictDenoised(backend, predictor, z, sigma, stepIndex);
        try
        {
            if (sigmaNext <= 0f)
            {
                backend.Scale(z, denoised, 1.0f);
                return;
            }

            float h = SamplerOps.Lambda(sigmaNext) - SamplerOps.Lambda(sigma);
            float etaH = _eta * h;
            float xScale = (sigmaNext / sigma) * MathF.Exp(-etaH);
            float denoisedScale = 1.0f - MathF.Exp(-h - etaH);

            if (_previousDenoised is null || _previousH <= 0f)
            {
                SamplerOps.MixInto(backend, z, denoised, xScale, denoisedScale);
            }
            else
            {
                // Midpoint second-order term: D = denoised + (1/2r)·(denoised − denoised_prev).
                float r = _previousH / h;
                float correction = 0.5f / r;
                using Tensor extrapolated = new Tensor(z.Shape, DType.F32);
                SamplerOps.SetMix(backend, extrapolated, denoised, _previousDenoised, 1.0f + correction, -correction);
                SamplerOps.MixInto(backend, z, extrapolated, xScale, denoisedScale);
            }

            // Re-inject exactly the variance the e^(−ηh) contraction removed.
            float noiseScale = sigmaNext * MathF.Sqrt(MathF.Max(0f, 1.0f - MathF.Exp(-2.0f * etaH)));
            SamplerOps.AddNoise(backend, z, _shape, _seed, stepIndex, noiseScale);

            _previousDenoised?.Dispose();
            _previousDenoised = denoised;
            denoised = null!;
            _previousH = h;
        }
        finally
        {
            denoised?.Dispose();
        }
    }
}
