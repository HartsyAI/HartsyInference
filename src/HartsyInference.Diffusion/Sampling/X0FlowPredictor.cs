using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Sampling;

/// <summary>An <see cref="IDenoisePredictor"/> over a model that predicts the clean sample (x0) and is sampled as a flow:
/// each evaluation turns x0 into the velocity <c>v = (x − x0) / max(σ, floor)</c> that the flow-match Euler update steps on.
/// Chroma Radiance and Zeta-Chroma each carry the same conversion inline; this is the shared form, with one addition.
///
/// <para><b>Guidance.</b> An evaluation is guidance-free when the forward returns no unconditional prediction, or returns the
/// same tensor for both (how the rest of the engine signals it). Otherwise the scale applies, taken from the unconditional
/// prediction or from the conditional one per <see cref="GuidanceAnchor"/>; the caller decides whether to run the
/// unconditional pass at all, as Chroma Radiance does when its scale is not above 1.</para>
///
/// <para><b>Without a guided-x0 transform</b> the conversion runs per branch and the raw pair goes back to the sampler at
/// the CFG scale, so the sampler's fused op does the combine. This is the order those families use, and since the
/// conversion is affine in x0 it equals combining first.</para>
///
/// <para><b>With a guided-x0 transform</b> the CFG combine happens here, in x0 space, the transform (a clamp to a valid
/// range, a gamut clip) is applied to the guided prediction, and the velocity of the result goes back as both halves of the
/// pair at guidance 1.0, which makes the sampler's combine an identity. The transform has to see the combined x0, which is
/// why it cannot ride in the closure of a pair-returning predictor.</para>
///
/// <para><b>Samplers.</b> The σ floor under-corrects the last steps on purpose, so use this with Euler. Higher-order
/// samplers recover x0 as <c>x − σ·v</c> (<see cref="SamplerMath.ToDenoised"/>), which is not the model's x0 once σ is below
/// the floor.</para>
///
/// <para>What the model sees as its input and timestep (an input rescale, a calibrated timestep) stays in the
/// <see cref="ForwardX0"/> closure: the predictor knows only the sample and σ it was asked about.</para></summary>
public sealed class X0FlowPredictor : IDenoisePredictor
{
    /// <summary>Runs the model on <c>(x, σ)</c> and returns its x0 predictions, conditional and unconditional (null or the
    /// same tensor when the evaluation is guidance-free). The predictor owns and disposes them.</summary>
    public delegate (Tensor Cond, Tensor? Uncond) ForwardX0(Tensor x, float sigma, int stepIndex);

    /// <summary>Maps the guided x0 to the x0 the velocity is taken from. It returns a new tensor, or its input; the
    /// predictor disposes the input when the two differ. Receives the σ of the evaluation.</summary>
    public delegate Tensor GuidedX0Transform(Tensor guidedX0, float sigma);

    private readonly IBackend _backend;
    private readonly ForwardX0 _forward;
    private readonly float _guidance;
    private readonly float _sigmaFloor;
    private readonly GuidanceAnchor _anchor;
    private readonly GuidedX0Transform? _transform;

    /// <summary>Creates the predictor.</summary>
    /// <param name="guidanceScale">The CFG scale <c>w</c>, applied whenever the forward returns an unconditional prediction.</param>
    /// <param name="sigmaFloor">The floor under σ in <c>v = (x − x0) / max(σ, floor)</c>.</param>
    public X0FlowPredictor(IBackend backend, ForwardX0 forward, float guidanceScale, float sigmaFloor,
        GuidanceAnchor anchor = GuidanceAnchor.Uncond, GuidedX0Transform? guidedX0Transform = null)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(forward);
        if (!(sigmaFloor > 0f)) throw new ArgumentOutOfRangeException(nameof(sigmaFloor), "The σ floor must be positive.");
        _backend = backend;
        _forward = forward;
        _guidance = guidanceScale;
        _sigmaFloor = sigmaFloor;
        _anchor = anchor;
        _transform = guidedX0Transform;
    }

    /// <inheritdoc/>
    public PredictionType Prediction => PredictionType.FlowVelocity;

    /// <inheritdoc/>
    public DenoisePrediction Predict(Tensor x, float sigma, int stepIndex)
    {
        (Tensor cond, Tensor? uncond) = _forward(x, sigma, stepIndex);
        bool guided = uncond is not null && !ReferenceEquals(uncond, cond);
        // Every tensor this call still owns is held in one of these and cleared as ownership moves on, so the finally
        // releases exactly what an exception (or an early return) left behind.
        Tensor? condX0 = cond, uncondX0 = guided ? uncond : null;
        Tensor? x0 = null, transformed = null, condVelocity = null, uncondVelocity = null;
        try
        {
            if (_transform is null)
            {
                condVelocity = X0Prediction.ToVelocityDevice(_backend, condX0, x, sigma, _sigmaFloor);
                condX0.Dispose();
                condX0 = null;
                if (!guided) return Hand(ref condVelocity, ref condVelocity, 1f);
                uncondVelocity = X0Prediction.ToVelocityDevice(_backend, uncondX0!, x, sigma, _sigmaFloor);
                uncondX0!.Dispose();
                uncondX0 = null;
                return Hand(ref condVelocity, ref uncondVelocity, _anchor == GuidanceAnchor.Cond ? _guidance + 1f : _guidance);
            }

            if (guided)
            {
                x0 = CombineX0(condX0, uncondX0!);
                condX0.Dispose();
                uncondX0!.Dispose();
                condX0 = uncondX0 = null;
            }
            else
            {
                x0 = condX0;
                condX0 = null;
            }
            transformed = _transform(x0, sigma);
            if (ReferenceEquals(transformed, x0)) x0 = null;
            else
            {
                x0.Dispose();
                x0 = null;
            }
            condVelocity = X0Prediction.ToVelocityDevice(_backend, transformed, x, sigma, _sigmaFloor);
            return Hand(ref condVelocity, ref condVelocity, 1f);
        }
        finally
        {
            condX0?.Dispose();
            uncondX0?.Dispose();
            x0?.Dispose();
            transformed?.Dispose();
            condVelocity?.Dispose();
            uncondVelocity?.Dispose();
        }
    }

    // Moves the velocity tensors into the returned pair and clears the caller's references so the finally leaves them alone.
    private static DenoisePrediction Hand(ref Tensor? cond, ref Tensor? uncond, float guidance)
    {
        DenoisePrediction prediction = new(cond!, uncond!, guidance);
        cond = null;
        uncond = null;
        return prediction;
    }

    // cond·a + uncond·b with (a, b) = (w, 1 - w) or (1 + w, -w). The inputs stay the caller's.
    private Tensor CombineX0(Tensor cond, Tensor uncond)
    {
        (float a, float b) = _anchor == GuidanceAnchor.Cond ? (1f + _guidance, -_guidance) : (_guidance, 1f - _guidance);
        Tensor combined = new(cond.Shape, DType.F32);
        _backend.AffineMix(combined, cond, uncond, a, b);
        return combined;
    }
}
