using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;

namespace HartsyInference.Diffusion.Sampling;

/// <summary>Plain Euler — the default, and deliberately the one sampler that does NOT go through
/// <see cref="SamplerMath"/>.
///
/// <para>It collapses to the single fused device op every pipeline in the engine already calls:
/// <see cref="IBackend.CfgEulerStep"/>, which combines cond/uncond by CFG and applies <c>z += v·dt</c> in place on the
/// GPU-resident latent. That is why <see cref="IDenoisePredictor"/> returns the prediction PAIR — so this path stays
/// byte-for-byte the instruction sequence it was before the sampler abstraction existed.</para>
///
/// <para><b>This is the rollout's safety property.</b> A generation that names no sampler, or names <c>euler</c>, must
/// produce bit-identical output to the pre-refactor pipeline and cost the same. Every pipeline conversion is gated on a
/// test asserting exactly that, so the abstraction can be added family by family without a quality or perf review each
/// time.</para>
///
/// <para>Correct for <see cref="Schedulers.PredictionType.Epsilon"/> and
/// <see cref="Schedulers.PredictionType.FlowVelocity"/>, whose Euler updates are the same expression. V-prediction needs
/// the denoised conversion and is refused here rather than silently mis-integrated.</para></summary>
public sealed class EulerSampler : ISampler
{
    private readonly float[] _sigmas;

    /// <inheritdoc/>
    public string Name => "euler";

    /// <inheritdoc/>
    public int StepCount => _sigmas.Length - 1;

    /// <summary>Creates the sampler over a resolved sigma array (length <c>steps + 1</c>, ending at 0).</summary>
    public EulerSampler(float[] sigmas)
    {
        ArgumentNullException.ThrowIfNull(sigmas);
        if (sigmas.Length < 2)
        {
            throw new ArgumentException($"Need at least 2 sigmas (one step plus the terminal zero); got {sigmas.Length}.",
                nameof(sigmas));
        }
        _sigmas = sigmas;
    }

    /// <inheritdoc/>
    public void Reset(TensorShape latentShape)
    {
        // Stateless: no history, no per-run buffers.
    }

    /// <inheritdoc/>
    public void Step(IBackend backend, Tensor z, IDenoisePredictor predictor, int stepIndex)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(z);
        ArgumentNullException.ThrowIfNull(predictor);
        if (predictor.Prediction == Schedulers.PredictionType.VPrediction)
        {
            throw new NotSupportedException(
                "EulerSampler's fused update is the epsilon/flow expression z += v·dt; a v-prediction model needs the "
                + "denoised conversion. Use a sampler built on SamplerMath, or the family's own v-prediction scheduler.");
        }
        // NextDiT families predict −v, so the sign rides on the scalar delta. Exact in F32 and free, where negating
        // the latent-sized prediction would cost a kernel and an allocation on every step of every generation.
        float delta = _sigmas[stepIndex + 1] - _sigmas[stepIndex];
        if (predictor.Prediction == Schedulers.PredictionType.NegatedFlowVelocity)
        {
            delta = -delta;
        }
        using DenoisePrediction prediction = predictor.Predict(z, _sigmas[stepIndex], stepIndex);
        backend.CfgEulerStep(z, prediction.Cond, prediction.Uncond, prediction.Guidance, delta);
    }
}
