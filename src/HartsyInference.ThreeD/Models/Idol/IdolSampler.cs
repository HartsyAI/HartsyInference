using HartsyInference.Core.Backends;
using HartsyInference.Core.Tensors;
using HartsyInference.ThreeD.Sampling;

namespace HartsyInference.ThreeD.Models.Idol;

/// <summary>Idol's guided sampler (idol-sdk's <c>sample_flow</c>): Euler integration of the rectified flow from noise (t = 1) to the latent
/// (t = 0) on the time grid <c>r·t / (1 + (r − 1)·t)</c>, with classifier-free guidance <c>(1 + w)·v(cond) − w·v(zeros)</c>. The latent and the
/// camera integrate together, so they ride the shared <see cref="RectifiedFlowSampler"/> as one flat tensor that the prediction splits and joins;
/// the sampler's guidance <c>s·pos + (1 − s)·neg</c> with <c>s = 1 + w</c> is Idol's formula, applied over the whole interval [0, 1].</summary>
public static class IdolSampler
{
    /// <summary>Idol's defaults: 50 steps, guidance 3 and time shift 3.</summary>
    public const int DefaultSteps = 50;

    /// <summary>The default classifier-free guidance strength <c>w</c> (<c>--cfg_strength</c>).</summary>
    public const float DefaultGuidance = 3f;

    /// <summary>The default time shift <c>rescale_t</c>.</summary>
    public const double DefaultRescaleT = 3.0;

    /// <summary>Denoises <paramref name="noiseLatent"/> <c>[1, 8192, 16]</c> and <paramref name="noiseCamera"/> <c>[1, 1, 5]</c> (neither is modified) into the
    /// object latent and the input camera. <paramref name="positive"/> and <paramref name="negative"/> are the image streams of
    /// <see cref="IdolDenoiser.EmbedContext"/> for the picture and for zeros. A <paramref name="guidance"/> of 0 runs the conditional pass alone.
    /// <paramref name="passScope"/> opens a scope around each forward pass, given that pass's <c>1000·t</c> (Idol-turbo switches its phase expert there). The caller owns both results.</summary>
    public static (Tensor Latent, Tensor Camera) Sample(IBackend backend, IdolDenoiser model, Tensor positive, Tensor negative,
        Tensor noiseLatent, Tensor noiseCamera, int steps = DefaultSteps, float guidance = DefaultGuidance, double rescaleT = DefaultRescaleT,
        Action<int>? onStep = null, Func<float, IDisposable>? passScope = null)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(model);
        if (guidance < 0f || !float.IsFinite(guidance))
            throw new ArgumentOutOfRangeException(nameof(guidance), guidance, "Guidance must be finite and not negative.");
        Tensor packedNoise = Join(noiseLatent, noiseCamera, noiseLatent.Shape, noiseCamera.Shape);
        FlowSamplerParams parameters = new()
        {
            Steps = steps, GuidanceStrength = 1f + guidance, GuidanceRescale = 0f, IntervalLow = 0.0, IntervalHigh = 1.0, RescaleT = rescaleT,
        };
        Tensor sample;
        try
        {
            sample = RectifiedFlowSampler.Sample(backend, packedNoise, parameters, (x, tModel, conditional) =>
            {
                (Tensor latent, Tensor camera) = Split(x, noiseLatent.Shape, noiseCamera.Shape);
                using (latent)
                using (camera)
                {
                    using IDisposable? scope = passScope?.Invoke(tModel);
                    (Tensor vLatent, Tensor vCamera) = model.Forward(backend, latent, camera, tModel, conditional ? positive : negative);
                    using (vLatent)
                    using (vCamera)
                        return Join(vLatent, vCamera, vLatent.Shape, vCamera.Shape);
                }
            }, onStep);
        }
        finally
        {
            packedNoise.Dispose();
        }

        try
        {
            return Split(sample, noiseLatent.Shape, noiseCamera.Shape);
        }
        finally
        {
            sample.Dispose();
        }
    }

    /// <summary>The latent and the camera as one flat <c>[1, latent + camera]</c> tensor.</summary>
    internal static Tensor Join(Tensor latent, Tensor camera, TensorShape latentShape, TensorShape cameraShape)
    {
        Tensor packed = new(new TensorShape(1, latentShape.ElementCount + cameraShape.ElementCount), DType.F32);
        Span<float> p = packed.AsSpan<float>();
        latent.AsReadOnlySpan<float>().CopyTo(p);
        camera.AsReadOnlySpan<float>().CopyTo(p[(int)latentShape.ElementCount..]);
        return packed;
    }

    /// <summary>The inverse of <see cref="Join"/>: two new tensors of the given shapes.</summary>
    internal static (Tensor Latent, Tensor Camera) Split(Tensor packed, TensorShape latentShape, TensorShape cameraShape)
    {
        Tensor latent = new(latentShape, DType.F32);
        Tensor camera = new(cameraShape, DType.F32);
        ReadOnlySpan<float> p = packed.AsReadOnlySpan<float>();
        p[..(int)latentShape.ElementCount].CopyTo(latent.AsSpan<float>());
        p[(int)latentShape.ElementCount..].CopyTo(camera.AsSpan<float>());
        return (latent, camera);
    }
}
