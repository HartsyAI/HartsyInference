using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Schedulers;

namespace HartsyInference.Diffusion.Sampling;

/// <summary>An <see cref="IDenoisePredictor"/> built from a lambda, so a pipeline adopts the sampler seam by wrapping
/// the loop body it already has instead of growing a nested class.
///
/// <para>This is what makes the flow-match rollout cheap. Those pipelines differ mostly in how they call their
/// transformer — Krea2 anchors CFG on the conditional branch, Flux.2 carries a distilled guidance embedding, Chroma
/// threads attention masks, Qwen-Image appends reference latents — and all of that stays inside the closure. The
/// sampler sees only <c>(x, sigma) → prediction pair</c>.</para>
///
/// <para><b>Guidance belongs to whichever side owns the combine.</b> A family that applies its own CFG inside the
/// closure (Krea2's cond-anchored variant, anything with rescale or tcfg) returns the already-combined velocity as both
/// halves of the pair and passes <c>guidanceScale: 1.0</c>, which makes the sampler's combine an identity. A family
/// that hands back raw cond/uncond passes its real scale and lets the fused device op do the combine. Getting this
/// backwards double-applies guidance, so the choice is explicit rather than inferred.</para></summary>
public sealed class DelegateDenoisePredictor : IDenoisePredictor
{
    private readonly Func<Tensor, float, int, DenoisePrediction> _predict;

    /// <summary>Wraps <paramref name="predict"/> as a predictor.</summary>
    /// <param name="prediction">What the closure's output means — <see cref="PredictionType.FlowVelocity"/> for the
    /// DiT fleet, <see cref="PredictionType.Epsilon"/> for the SD family.</param>
    /// <param name="guidanceScale">1.0 when the closure already applied CFG itself; the request's scale otherwise.</param>
    public DelegateDenoisePredictor(PredictionType prediction, float guidanceScale,
        Func<Tensor, float, int, DenoisePrediction> predict)
    {
        ArgumentNullException.ThrowIfNull(predict);
        Prediction = prediction;
        GuidanceScale = guidanceScale;
        _predict = predict;
    }

    /// <inheritdoc/>
    public PredictionType Prediction { get; }

    /// <inheritdoc/>
    public float GuidanceScale { get; }

    /// <inheritdoc/>
    public DenoisePrediction Predict(Tensor x, float sigma, int stepIndex) => _predict(x, sigma, stepIndex);
}
