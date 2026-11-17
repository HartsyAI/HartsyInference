using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Schedulers;
using HartsyInference.Diffusion.Utilities;

namespace HartsyInference.Diffusion.Sampling;

/// <summary>Which prediction the CFG difference is taken from. <see cref="Uncond"/> is <c>u + w·(c − u)</c>, the form the
/// sampler's fused op applies at guidance <c>w</c>; <see cref="Cond"/> is <c>c + w·(c − u)</c> (Krea2, Chroma Radiance), which
/// the same op applies at guidance <c>w + 1</c>.</summary>
public enum GuidanceAnchor
{
    /// <summary>Classifier-free guidance as <c>w·cond + (1 − w)·uncond</c>.</summary>
    Uncond,

    /// <summary>Guidance as <c>cond + w·(cond − uncond)</c>.</summary>
    Cond,
}

/// <summary>An <see cref="IDenoisePredictor"/> over a model that predicts the clean sample (x0) and is sampled as a flow:
/// each evaluation turns x0 into the velocity <c>v = (x − x0) / max(σ, floor)</c> that the flow-match Euler update steps on.
/// Chroma Radiance and Zeta-Chroma each carry the same conversion inline; this is the shared form, with one addition.
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
/// <para>What the model sees as its input and timestep (an input rescale, a calibrated timestep) stays in the
/// <see cref="ForwardX0"/> closure: the predictor knows only the sample and σ it was asked about.</para></summary>
public sealed class X0FlowPredictor : IDenoisePredictor
{
    /// <summary>Runs the model on <c>(x, σ)</c> and returns its x0 predictions, conditional and unconditional (null when
    /// the evaluation is guidance-free). The predictor owns and disposes both.</summary>
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
    /// <param name="guidanceScale">The CFG scale <c>w</c>. 1.0 runs guidance-free whatever the forward returns.</param>
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
        (Tensor condX0, Tensor? uncondX0) = _forward(x, sigma, stepIndex);
        bool guided = uncondX0 is not null && _guidance != 1f;
        if (!guided && uncondX0 is not null && !ReferenceEquals(uncondX0, condX0)) uncondX0.Dispose();

        if (_transform is null)
        {
            Tensor condV = X0Prediction.ToVelocityDevice(_backend, condX0, x, sigma, _sigmaFloor);
            condX0.Dispose();
            if (!guided) return new DenoisePrediction(condV, condV);
            Tensor uncondV = X0Prediction.ToVelocityDevice(_backend, uncondX0!, x, sigma, _sigmaFloor);
            uncondX0!.Dispose();
            return new DenoisePrediction(condV, uncondV, _anchor == GuidanceAnchor.Cond ? _guidance + 1f : _guidance);
        }

        Tensor x0 = guided ? CombineX0(condX0, uncondX0!) : condX0;
        Tensor transformed = _transform(x0, sigma);
        if (!ReferenceEquals(transformed, x0)) x0.Dispose();
        Tensor velocity = X0Prediction.ToVelocityDevice(_backend, transformed, x, sigma, _sigmaFloor);
        transformed.Dispose();
        return new DenoisePrediction(velocity, velocity);
    }

    // cond·a + uncond·b with (a, b) = (w, 1 - w) or (1 + w, -w); disposes both inputs.
    private Tensor CombineX0(Tensor cond, Tensor uncond)
    {
        (float a, float b) = _anchor == GuidanceAnchor.Cond ? (1f + _guidance, -_guidance) : (_guidance, 1f - _guidance);
        Tensor scaledCond = new(cond.Shape, DType.F32);
        Tensor scaledUncond = new(cond.Shape, DType.F32);
        Tensor combined = new(cond.Shape, DType.F32);
        _backend.Scale(scaledCond, cond, a);
        _backend.Scale(scaledUncond, uncond, b);
        _backend.Add(combined, scaledCond, scaledUncond);
        scaledCond.Dispose();
        scaledUncond.Dispose();
        cond.Dispose();
        uncond.Dispose();
        return combined;
    }
}

/// <summary>A signal-to-noise calibration for a model that was trained on one representation and is run on another that
/// carries <c>c</c> times as much signal per unit of noise (AsymFlow; PixVid's pixel patches against Wan2.1's VAE
/// latents). At noise level σ the pixel sample projects to <c>c·(1 − σ)·z + σ·ε</c>, which has the SNR of the latent flow at
/// <see cref="Sigma"/>, and scaling the input by <see cref="InputScale"/> matches its magnitude as well.
/// <para>Two calibrations compose by multiplying their scales, which is how a 2x2 average-pooled input (scale 2) stacks on a
/// patch-embedding calibration.</para></summary>
public readonly record struct SnrCalibration
{
    /// <summary>Creates a calibration with scale <paramref name="scale"/> (c). Must be positive.</summary>
    public SnrCalibration(float scale)
    {
        if (!(scale > 0f)) throw new ArgumentOutOfRangeException(nameof(scale), "The calibration scale must be positive.");
        Scale = scale;
    }

    /// <summary>The scale c. 1.0 is the identity.</summary>
    public float Scale { get; } = 1f;

    /// <summary>The noise level the model is conditioned on for sample noise level <paramref name="sigma"/>:
    /// <c>σ / (c·(1 − σ) + σ)</c>. 0 maps to 0 and 1 to 1.</summary>
    public float Sigma(float sigma) => sigma * InputScale(sigma);

    /// <summary>The factor the model input is multiplied by: <c>1 / (c·(1 − σ) + σ)</c>.</summary>
    public float InputScale(float sigma) => 1f / (EffectiveScale() * (1f - sigma) + sigma);

    /// <summary>The calibration that applies this one and then <paramref name="other"/>: the scales multiply.</summary>
    public SnrCalibration Compose(SnrCalibration other) => new(EffectiveScale() * other.EffectiveScale());

    // `default` has no scale set; treat it as the identity rather than a division by zero.
    private float EffectiveScale() => Scale > 0f ? Scale : 1f;
}
