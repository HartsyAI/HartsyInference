using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.Diffusion.Sampling;
using HartsyInference.Diffusion.Schedulers;

namespace HartsyInference.Diffusion.Utilities;

/// <summary>How a reference step's combined prediction is corrected.</summary>
public enum ReferenceCorrection
{
    /// <summary>The combined velocity is used as it is.</summary>
    None,

    /// <summary>Adaptive projected guidance: in x0 space, the part of the guided update parallel to the positive prediction is removed.</summary>
    Apg,

    /// <summary>In x0 space, the guided prediction is rescaled to the positive prediction's standard deviation, blended by the rescale phi.</summary>
    StdRescale,
}

/// <summary>What a step outside the reference window does.</summary>
public enum ReferenceOutside
{
    /// <summary>The reference stays in both branches, with no extra guidance on the reference. The reference adapter's default.</summary>
    KeepReference,

    /// <summary>Plain CFG on the base model, with no reference, outside the window.</summary>
    DropReference,
}

/// <summary>The model evaluations a reference step can need. "With reference" means the reference is joined and its LoRA is live.</summary>
public enum ReferenceBranch
{
    /// <summary>The positive prompt with the reference.</summary>
    PositiveWithReference,

    /// <summary>The negative prompt with the reference.</summary>
    NegativeWithReference,

    /// <summary>The positive prompt on the base model.</summary>
    PositiveWithoutReference,

    /// <summary>The negative prompt on the base model.</summary>
    NegativeWithoutReference,
}

/// <summary>Reference guidance settings. The window is a fraction of the denoising, as the reference adapter's own is.</summary>
/// <param name="Scale">The reference's guidance (the node's <c>ref_guidance</c>): 1 costs no extra call, higher pushes toward the reference.</param>
/// <param name="Correction">The correction applied after the combine.</param>
/// <param name="RescalePhi">The blend for <see cref="ReferenceCorrection.StdRescale"/>, in [0, 1].</param>
/// <param name="Start">The first denoising fraction with extra reference guidance, in [0, 1].</param>
/// <param name="End">The last denoising fraction with extra reference guidance, in [Start, 1].</param>
/// <param name="Outside">What steps outside the window do.</param>
public sealed class ReferenceGuidanceSettings
{
    /// <summary>Creates the settings and validates them.</summary>
    /// <param name="scale">The reference's guidance (the node's <c>ref_guidance</c>): 1 costs no extra call, higher pushes toward the reference.</param>
    /// <param name="correction">The correction applied after the combine.</param>
    /// <param name="rescalePhi">The blend for <see cref="ReferenceCorrection.StdRescale"/>, in [0, 1].</param>
    /// <param name="start">The first denoising fraction with extra reference guidance, in [0, 1].</param>
    /// <param name="end">The last denoising fraction with extra reference guidance, in [start, 1].</param>
    /// <param name="outside">What steps outside the window do.</param>
    public ReferenceGuidanceSettings(float scale, ReferenceCorrection correction, float rescalePhi = 0.5f, float start = 0f,
        float end = 1f, ReferenceOutside outside = ReferenceOutside.KeepReference)
    {
        if (!float.IsFinite(scale) || scale < 0f)
            throw new ArgumentOutOfRangeException(nameof(scale), scale, "The reference guidance scale must be finite and non-negative.");
        if (!float.IsFinite(rescalePhi) || rescalePhi < 0f || rescalePhi > 1f)
            throw new ArgumentOutOfRangeException(nameof(rescalePhi), rescalePhi, "The rescale phi must be in [0, 1].");
        if (!float.IsFinite(start) || start < 0f || start > 1f)
            throw new ArgumentOutOfRangeException(nameof(start), start, "The window start must be in [0, 1].");
        if (!float.IsFinite(end) || end < start || end > 1f)
            throw new ArgumentOutOfRangeException(nameof(end), end, "The window end must be in [start, 1].");
        Scale = scale;
        Correction = correction;
        RescalePhi = rescalePhi;
        Start = start;
        End = end;
        Outside = outside;
    }

    /// <summary>The reference's guidance scale.</summary>
    public float Scale { get; }

    /// <summary>The correction applied after the combine.</summary>
    public ReferenceCorrection Correction { get; }

    /// <summary>The blend for <see cref="ReferenceCorrection.StdRescale"/>.</summary>
    public float RescalePhi { get; }

    /// <summary>The first denoising fraction with extra reference guidance.</summary>
    public float Start { get; }

    /// <summary>The last denoising fraction with extra reference guidance.</summary>
    public float End { get; }

    /// <summary>What steps outside the window do.</summary>
    public ReferenceOutside Outside { get; }

    /// <summary>Whether the step at <paramref name="sigma"/> lies in the window. The bounds are the flow schedule's sigmas at
    /// <see cref="Start"/> and <see cref="End"/> (<paramref name="shift"/> is the schedule's flow shift), with the reference adapter's 1e-5 tolerance.</summary>
    public bool InWindow(float sigma, double shift)
    {
        // A window over the whole schedule covers every sigma, so it needs no percentage conversion (and no tolerance).
        if (Start <= 0f && End >= 1f)
            return true;
        double low = SamplerOptions.FlowPercentToSigma(End, shift);
        double high = SamplerOptions.FlowPercentToSigma(Start, shift);
        return sigma >= low - 1e-5 && sigma <= high + 1e-5;
    }
}

/// <summary>One denoising step with a reference joined, combined the way the reference adapter's post-CFG patch combines it.
/// Model-agnostic: the caller supplies the branch evaluations, so any model that joins a reference can use this.</summary>
public static class ReferenceGuidance
{
    /// <summary>Evaluates one branch of a step and returns its velocity. The caller owns the returned tensor.</summary>
    public delegate Tensor BranchPredictor(ReferenceBranch branch);

    /// <summary>Runs one step. The result is pre-combined: the same tensor on both sides at guidance 1, which the sampler uses as it is.</summary>
    /// <remarks><para>The combine is <c>base + Scale·(mid − base) + cfg·(positive − mid)</c>, where <c>mid</c> is the negative prediction
    /// with the reference (or the positive one when CFG is off) and <c>base</c> is the same branch without it. The branches it needs:</para>
    /// <list type="bullet">
    /// <item>In the window, with <see cref="ReferenceGuidanceSettings.Scale"/> not 1: positive and negative with the reference, plus the base (2 or 3 calls).</item>
    /// <item>In the window with Scale 1, or outside it with <see cref="ReferenceOutside.KeepReference"/>: positive and negative with the reference, and no base (1 or 2 calls).</item>
    /// <item>Outside the window with <see cref="ReferenceOutside.DropReference"/>: plain CFG on the base model (1 or 2 calls).</item>
    /// </list>
    /// <para>A correction is applied after the combine, in x0 space with the positive prediction's x0. Without one, the combine is done on
    /// velocities: the combine is affine and sums to one, so it equals the x0 combine.</para></remarks>
    public static DenoisePrediction Predict(IBackend backend, ReferenceGuidanceSettings settings, Tensor x, float sigma, double shift,
        float cfg, BranchPredictor predict)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(settings);
        ArgumentNullException.ThrowIfNull(predict);

        bool negative = cfg > 1.0f;
        bool windowed = settings.InWindow(sigma, shift);
        bool withBase = windowed && settings.Scale != 1.0f;
        bool withReference = windowed || settings.Outside == ReferenceOutside.KeepReference;

        Tensor? positive = null, negativeBranch = null, baseBranch = null;
        try
        {
            positive = predict(withReference ? ReferenceBranch.PositiveWithReference : ReferenceBranch.PositiveWithoutReference);
            if (negative)
                negativeBranch = predict(withReference ? ReferenceBranch.NegativeWithReference : ReferenceBranch.NegativeWithoutReference);
            if (withBase)
                baseBranch = predict(negative ? ReferenceBranch.NegativeWithoutReference : ReferenceBranch.PositiveWithoutReference);

            Tensor mid = negativeBranch ?? positive;
            // With no base the reference scale is 1 and the base is the mid branch itself, so base + 1·(mid − base) is mid.
            Tensor baseOrMid = baseBranch ?? mid;
            float innerScale = baseBranch is null ? 1f : settings.Scale;
            Tensor combined = CombineAndCorrect(backend, settings.Correction, settings.RescalePhi, x, sigma, positive, mid, baseOrMid,
                innerScale, cfg);
            return new DenoisePrediction(combined, combined, 1.0f);
        }
        finally
        {
            positive?.Dispose();
            negativeBranch?.Dispose();
            baseBranch?.Dispose();
        }
    }

    private static Tensor CombineAndCorrect(IBackend backend, ReferenceCorrection correction, float rescalePhi, Tensor x, float sigma,
        Tensor positive, Tensor mid, Tensor baseOrMid, float innerScale, float cfg)
    {
        if (correction == ReferenceCorrection.None)
        {
            Tensor velocity = new(x.Shape, DType.F32);
            try
            {
                CfgHelper.ApplyDualCfgDevice(backend, velocity, positive, mid, baseOrMid, innerScale, cfg);
            }
            catch
            {
                velocity.Dispose();
                throw;
            }
            return velocity;
        }

        using Tensor positiveX0 = new(x.Shape, DType.F32);
        using Tensor midX0 = new(x.Shape, DType.F32);
        using Tensor baseX0 = new(x.Shape, DType.F32);
        using Tensor guidedX0 = new(x.Shape, DType.F32);
        SamplerMath.ToDenoised(backend, positiveX0, x, positive, sigma, PredictionType.FlowVelocity);
        SamplerMath.ToDenoised(backend, midX0, x, mid, sigma, PredictionType.FlowVelocity);
        SamplerMath.ToDenoised(backend, baseX0, x, baseOrMid, sigma, PredictionType.FlowVelocity);
        CfgHelper.ApplyDualCfgDevice(backend, guidedX0, positiveX0, midX0, baseX0, innerScale, cfg);

        using Tensor correctedX0 = new(x.Shape, DType.F32);
        Correct(backend, correction, rescalePhi, guidedX0, positiveX0, correctedX0);
        return X0Prediction.ToVelocityDevice(backend, correctedX0, x, sigma);
    }

    /// <summary>Applies a correction to a guided prediction, given the positive prediction it is measured against. The reference
    /// adapter's <c>_correct</c>: APG or StdRescale, or the identity for <see cref="ReferenceCorrection.None"/>. Stateless: no
    /// momentum or threshold carries from one step to the next.</summary>
    /// <param name="guided">The guided prediction (x0 for the reference step; any tensor for the fixture checks).</param>
    /// <param name="positive">The positive prediction, in the same space.</param>
    /// <param name="output">Receives the corrected tensor, same shape as <paramref name="guided"/>.</param>
    public static void Correct(IBackend backend, ReferenceCorrection correction, float rescalePhi, Tensor guided, Tensor positive,
        Tensor output)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(guided);
        ArgumentNullException.ThrowIfNull(positive);
        ArgumentNullException.ThrowIfNull(output);
        if (!guided.Shape.Equals(positive.Shape) || !guided.Shape.Equals(output.Shape))
            throw new ArgumentException($"A correction needs matching shapes: guided {guided.Shape}, positive {positive.Shape}, output {output.Shape}.");
        switch (correction)
        {
            case ReferenceCorrection.None:
                backend.Scale(output, guided, 1f);
                break;
            case ReferenceCorrection.Apg:
                ApgCorrect(backend, output, guided, positive);
                break;
            case ReferenceCorrection.StdRescale:
                StdRescaleCorrect(backend, output, guided, positive, rescalePhi);
                break;
            default:
                throw new NotSupportedException($"Unhandled reference correction {correction}.");
        }
    }

    /// <summary>APG in x0 space: <c>guided − α·pos</c> with <c>α = ⟨guided − pos, pos⟩ / max(‖pos‖, ε)²</c>, which is the reference
    /// adapter's projection (its unit vector is <c>pos / max(‖pos‖, 1e-12)</c>). The result's inner product with <c>pos</c> is then ‖pos‖².</summary>
    private static void ApgCorrect(IBackend backend, Tensor output, Tensor guided, Tensor positive)
    {
        double dotGuidedPositive = Dot(guided, positive);
        double dotPositive = Dot(positive, positive);
        double norm = Math.Max(Math.Sqrt(dotPositive), 1e-12);
        double alpha = (dotGuidedPositive - dotPositive) / (norm * norm);
        backend.AffineMix(output, guided, positive, 1f, (float)-alpha);
    }

    /// <summary>StdRescale in x0 space: <c>(phi·std(pos)/std(guided) + (1 − phi))·guided</c>, with the unbiased standard deviation over
    /// every element, as the reference adapter computes it. A constant guided prediction has no spread to rescale from, so it passes
    /// through unchanged rather than dividing by zero.</summary>
    private static void StdRescaleCorrect(IBackend backend, Tensor output, Tensor guided, Tensor positive, float phi)
    {
        double stdGuided = StandardDeviation(guided.AsReadOnlySpan<float>());
        float factor = stdGuided > 0.0
            ? (float)(phi * (StandardDeviation(positive.AsReadOnlySpan<float>()) / stdGuided) + (1.0 - phi))
            : 1f;
        backend.Scale(output, guided, factor);
    }

    /// <summary>The unbiased standard deviation, taken around the mean in a second pass, so a large mean does not cancel the spread.</summary>
    private static double StandardDeviation(ReadOnlySpan<float> values)
    {
        double mean = 0;
        foreach (float v in values)
        {
            mean += v;
        }
        mean /= values.Length;
        double squares = 0;
        foreach (float v in values)
        {
            squares += (v - mean) * (v - mean);
        }
        return Math.Sqrt(squares / (values.Length - 1));
    }

    /// <summary>The inner product of two tensors of equal element count, summed in double on the host. The device GEMM runs F32 operands
    /// in BF16 or F16 (<c>CudaBackend.ResolveGemmDtype</c>), too coarse for the APG projection, so the sum reads its operands back: one
    /// copy per reduction, small next to a model call.</summary>
    private static double Dot(Tensor left, Tensor right)
    {
        ReadOnlySpan<float> a = left.AsReadOnlySpan<float>();
        ReadOnlySpan<float> b = right.AsReadOnlySpan<float>();
        double sum = 0;
        for (int i = 0; i < a.Length; i++)
        {
            sum += (double)a[i] * b[i];
        }
        return sum;
    }
}
