using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Schedulers;

namespace HartsyInference.Diffusion.Sampling;

/// <summary>True CFG as two batch-1 passes of one forward, the conditional one first, paired at a fixed scale for the sampler's fused combine.</summary>
/// <typeparam name="TCondition">What one branch conditions on: its text features, or whatever the forward needs per branch.</typeparam>
public sealed class TwoPassCfgPredictor<TCondition> : IDenoisePredictor where TCondition : class
{
    private readonly Func<Tensor, float, int, TCondition, Tensor> _forward;
    private readonly TCondition _cond;
    private readonly TCondition? _uncond;
    private readonly float _guidance;

    /// <summary>Runs <paramref name="forward"/> as <c>(x, sigma, stepIndex, branch)</c>; a null <paramref name="uncond"/> runs the conditional pass alone, unguided.</summary>
    public TwoPassCfgPredictor(PredictionType prediction, Func<Tensor, float, int, TCondition, Tensor> forward, TCondition cond,
        TCondition? uncond, float guidance)
    {
        ArgumentNullException.ThrowIfNull(forward);
        ArgumentNullException.ThrowIfNull(cond);
        Prediction = prediction;
        _forward = forward;
        _cond = cond;
        _uncond = uncond;
        _guidance = guidance;
    }

    /// <inheritdoc/>
    public PredictionType Prediction { get; }

    /// <inheritdoc/>
    public DenoisePrediction Predict(Tensor x, float sigma, int stepIndex)
    {
        Tensor cond = _forward(x, sigma, stepIndex, _cond);
        if (_uncond is null)
            return new DenoisePrediction(cond, cond);
        Tensor uncond;
        try
        {
            uncond = _forward(x, sigma, stepIndex, _uncond);
        }
        catch
        {
            cond.Dispose();
            throw;
        }
        return new DenoisePrediction(cond, uncond, _guidance);
    }
}
