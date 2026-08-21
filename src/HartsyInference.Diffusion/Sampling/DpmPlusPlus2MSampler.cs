using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Sampling;

/// <summary>DPM-Solver++(2M) — k-diffusion's <c>dpmpp_2m</c>, and the single most-used sampler in the SD/SDXL
/// community. Second-order accuracy at ONE model evaluation per step, which it buys by reusing the previous step's
/// denoised estimate instead of evaluating twice like <see cref="HeunSampler"/> or <see cref="Dpm2Sampler"/>.
///
/// <para>Integrates in half-log-SNR <c>λ = −log σ</c>, where the probability-flow ODE has an exact exponential
/// solution. Per step with <c>h = λ_next − λ</c>:</para>
/// <code>
/// x ← (σ_next/σ)·x − expm1(−h)·D          // D = the (extrapolated) denoised estimate
/// D = (1 + 1/2r)·denoised − (1/2r)·denoised_prev,   r = h_last/h
/// </code>
/// <para>The first step has no previous estimate and the last has <c>σ_next == 0</c>; both fall back to the
/// first-order form, exactly as k-diffusion does. <c>−expm1(−h)</c> is computed as <c>1 − e^(−h)</c> via
/// <see cref="MathF"/>, which stays accurate for the small <c>h</c> that a fine schedule produces.</para></summary>
public sealed class DpmPlusPlus2MSampler : ISampler
{
    private readonly float[] _sigmas;
    private Tensor? _previousDenoised;
    private float _previousH;

    /// <inheritdoc/>
    public string Name => "dpmpp_2m";

    /// <inheritdoc/>
    public int StepCount => _sigmas.Length - 1;

    /// <summary>Creates the sampler over a resolved sigma array.</summary>
    public DpmPlusPlus2MSampler(float[] sigmas)
    {
        ArgumentNullException.ThrowIfNull(sigmas);
        if (sigmas.Length < 2)
        {
            throw new ArgumentException($"Need at least 2 sigmas; got {sigmas.Length}.", nameof(sigmas));
        }
        _sigmas = sigmas;
    }

    /// <inheritdoc/>
    public void Reset(TensorShape latentShape)
    {
        // History must not survive into a second generation — a stale denoised estimate from the previous image
        // would be extrapolated into this one's first step.
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
                // Terminal step: the denoised estimate IS the result.
                backend.Scale(z, denoised, 1.0f);
                return;
            }

            float lambda = SamplerOps.Lambda(sigma);
            float lambdaNext = SamplerOps.Lambda(sigmaNext);
            float h = lambdaNext - lambda;
            float sigmaRatio = sigmaNext / sigma;
            float negExpm1 = 1.0f - MathF.Exp(-h);

            if (_previousDenoised is null || _previousH <= 0f)
            {
                // First step of the run: no history to extrapolate from, so first-order.
                SamplerOps.MixInto(backend, z, denoised, sigmaRatio, negExpm1);
            }
            else
            {
                float r = _previousH / h;
                float currentWeight = 1.0f + (1.0f / (2.0f * r));
                float previousWeight = -1.0f / (2.0f * r);
                using Tensor extrapolated = new Tensor(z.Shape, DType.F32);
                SamplerOps.SetMix(backend, extrapolated, denoised, _previousDenoised, currentWeight, previousWeight);
                SamplerOps.MixInto(backend, z, extrapolated, sigmaRatio, negExpm1);
            }

            // PIN the history. A pipeline's mid-loop Backend.FreeActivations() reclaims every unpinned activation
            // WITHOUT a device-to-host sync and clears its GPU binding, so an unpinned denoised estimate would be
            // read back as stale host bytes on the next step — a silent corruption only multistep samplers can hit,
            // and only on the pipelines that reclaim per step (SD3, Flux, Qwen-Image, Z-Image and others do).
            // Pinning here fixes it once for every pipeline rather than per conversion.
            _previousDenoised?.Dispose();
            _previousDenoised = denoised;
            backend.PinActivation(_previousDenoised);
            denoised = null!;
            _previousH = h;
        }
        finally
        {
            denoised?.Dispose();
        }
    }
}
